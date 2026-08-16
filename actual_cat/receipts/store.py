"""Receipt persistent store — JSON-file-per-receipt across inbox/pending/done."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _store_root(store_path: str) -> Path:
    root = Path(store_path)
    for sub in ("inbox", "pending", "done"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root


# ---------------------------------------------------------------------------
# Write operations
# ---------------------------------------------------------------------------

def save_received(store_path: str, image_path: Path, source: str) -> str:
    """Record a newly received image in inbox. Returns the receipt ID."""
    receipt_id = uuid4().hex
    root = _store_root(store_path)
    meta: dict[str, Any] = {
        "id": receipt_id,
        "status": "received",
        "source": source,
        "input_kind": "image",
        "received_ts": _now_iso(),
        "image_path": str(image_path),
    }
    (root / "inbox" / f"{receipt_id}.json").write_text(json.dumps(meta, indent=2))
    return receipt_id


def save_received_text(
    store_path: str,
    text: str,
    source: str,
    extra: dict[str, Any] | None = None,
) -> str:
    """Record a newly received plain-text receipt in inbox. Returns the receipt ID.

    The text is stored inline in the meta (no sidecar file); the extractor reads
    meta["text"]. `extra` carries optional hints (e.g. hint_merchant/hint_date).
    """
    receipt_id = uuid4().hex
    root = _store_root(store_path)
    meta: dict[str, Any] = {
        "id": receipt_id,
        "status": "received",
        "source": source,
        "input_kind": "text",
        "received_ts": _now_iso(),
        "text": text,
        **(extra or {}),
    }
    (root / "inbox" / f"{receipt_id}.json").write_text(json.dumps(meta, indent=2))
    return receipt_id


def record_ocr_attempt(store_path: str, receipt_id: str) -> int:
    """Increment and persist the OCR attempt counter on an inbox receipt.

    Returns the new count. Call this BEFORE attempting OCR, not after: if the
    process is killed mid-OCR (a slow vision call outliving the service manager's
    timeout), nothing written after the attempt survives, so a count kept on the
    success path would never advance and the receipt would retry forever.

    Receipts written before this counter existed have no key and start at 0.
    """
    root = _store_root(store_path)
    inbox_json = root / "inbox" / f"{receipt_id}.json"
    meta = json.loads(inbox_json.read_text())
    attempts = int(meta.get("ocr_attempts", 0)) + 1
    meta["ocr_attempts"] = attempts
    meta["last_ocr_attempt_ts"] = _now_iso()
    inbox_json.write_text(json.dumps(meta, indent=2))
    return attempts


def save_pending(store_path: str, receipt_id: str, ocr_result: dict[str, Any]) -> None:
    """Move receipt from inbox to pending after successful OCR."""
    root = _store_root(store_path)
    inbox_json = root / "inbox" / f"{receipt_id}.json"
    meta = json.loads(inbox_json.read_text())
    meta["status"] = "pending"
    meta["ocr"] = ocr_result
    pending_json = root / "pending" / f"{receipt_id}.json"
    pending_json.write_text(json.dumps(meta, indent=2))
    inbox_json.unlink(missing_ok=True)


def save_done(
    store_path: str,
    receipt_id: str,
    final_status: str,
    matched_txn_id: str | None = None,
    splits: list[dict[str, Any]] | None = None,
    prior_category: str | None = None,
) -> None:
    """Move receipt from pending (or inbox) to done."""
    root = _store_root(store_path)
    pending_json = root / "pending" / f"{receipt_id}.json"
    inbox_json = root / "inbox" / f"{receipt_id}.json"
    src = pending_json if pending_json.exists() else inbox_json
    meta = json.loads(src.read_text())
    meta["status"] = final_status
    meta["resolved_ts"] = _now_iso()
    if matched_txn_id is not None:
        meta["matched_txn_id"] = matched_txn_id
    if splits is not None:
        meta["splits"] = splits
    if prior_category is not None:
        meta["prior_category"] = prior_category
    done_json = root / "done" / f"{receipt_id}.json"
    done_json.write_text(json.dumps(meta, indent=2))
    src.unlink(missing_ok=True)


def save_failed(store_path: str, receipt_id: str, error: str) -> None:
    save_done(store_path, receipt_id, "failed")
    # Append error detail to the done record
    root = _store_root(store_path)
    done_json = root / "done" / f"{receipt_id}.json"
    meta = json.loads(done_json.read_text())
    meta["error"] = error
    done_json.write_text(json.dumps(meta, indent=2))


# ---------------------------------------------------------------------------
# Read operations
# ---------------------------------------------------------------------------

def list_inbox(store_path: str) -> list[dict[str, Any]]:
    root = _store_root(store_path)
    return [
        json.loads(p.read_text())
        for p in sorted((root / "inbox").glob("*.json"))
    ]


def list_pending(store_path: str) -> list[dict[str, Any]]:
    root = _store_root(store_path)
    return [
        json.loads(p.read_text())
        for p in sorted((root / "pending").glob("*.json"))
    ]
