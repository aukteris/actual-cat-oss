"""Receipt matching and split pipeline."""

from __future__ import annotations

from datetime import date, timedelta
from typing import TYPE_CHECKING, Any

from ..sync_meta import is_pending
from ..tags import (
    append_tag,
    has_applied_split_marker,
    remove_tag,
    strip_category_tag,
)
from . import store as receipt_store

if TYPE_CHECKING:
    from ..audit import AuditLogger
    from ..config import Config
    from ..llm import LLMClient

_CONFIDENCE_RANK = {"high": 2, "medium": 1, "low": 0}


def _meets_threshold(confidence: str, threshold: str) -> bool:
    return _CONFIDENCE_RANK.get(confidence, 0) >= _CONFIDENCE_RANK.get(threshold, 2)


def find_receipt_match(
    session: Any,
    receipt_meta: dict[str, Any],
    window_days: int,
    defer_pending: bool = False,
) -> list[Any]:
    """Find eligible on-budget transactions matching the receipt's exact amount.

    Broader than find_uncategorized: receipt overrides prior categorization,
    so we scan ALL on-budget, non-transfer, non-split-parent, non-tombstoned
    transactions that haven't already been receipt-split (applied).

    Transactions bearing only #ai-suggested-split are still eligible — a new
    higher-confidence receipt submission can finalize the split (hybrid retry).

    Off-budget accounts excluded (consistent with all other pipelines).

    With defer_pending, rows the bank hasn't finalized are excluded: splitting
    against a pending authorization computes the split from the pre-tip total,
    which is exactly the discrepancy that produces the duplicate in the first
    place. The receipt stays pending and matches the posted row instead.
    """
    from actual.database import Transactions
    from actual.utils.conversions import date_to_int

    ocr = receipt_meta.get("ocr", {})
    total_cents: int = ocr.get("total_cents", 0)
    if total_cents <= 0:
        return []

    receipt_date_str: str | None = ocr.get("date")
    if receipt_date_str:
        try:
            receipt_date = date.fromisoformat(receipt_date_str)
        except ValueError:
            receipt_date = None
    else:
        receipt_date = None

    if receipt_date is None:
        # OCR couldn't read the date — fall back to the submission date so
        # a receipt posted promptly after purchase still matches within the window.
        received_ts = receipt_meta.get("received_ts", "")
        try:
            from datetime import datetime
            receipt_date = datetime.fromisoformat(received_ts).date()
        except (ValueError, TypeError):
            return []

    lo = date_to_int(receipt_date - timedelta(days=window_days))
    hi = date_to_int(receipt_date + timedelta(days=window_days))

    target_amount = -total_cents

    candidates = (
        session.query(Transactions)
        .filter(
            Transactions.amount == target_amount,
            Transactions.tombstone == 0,
            Transactions.transferred_id.is_(None),
            Transactions.is_parent == 0,
            Transactions.date >= lo,
            Transactions.date <= hi,
        )
        .all()
    )

    return [
        t for t in candidates
        if t.account is not None
        and not t.account.offbudget
        # Only exclude applied splits (#ai-receipt-split); suggested splits
        # (#ai-suggested-split) remain eligible for a retry submission.
        and not has_applied_split_marker(t.notes)
        and not (defer_pending and is_pending(t))
        # Also exclude the broad split marker for categorization pipeline
        # compatibility — but NOT for the receipt matcher. The receipt matcher
        # only cares about applied splits. However, has_split_marker also
        # matches #ai-suggested-split which we want to ALLOW here.
        # So we use only has_applied_split_marker above.
    ]


def _splits_from_ocr(ocr: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "description": item["description"],
            "amount_cents": item["amount_cents"],
            "category": item["category"],
        }
        for item in ocr.get("line_items", [])
        if isinstance(item.get("amount_cents"), int)  # skip unreadable (None) amounts
    ]


def process_receipt_splits(
    actual: Any,
    llm: "LLMClient",
    audit: "AuditLogger",
    cfg: "Config",
    schema_text: str,
    prompts: Any,
) -> None:
    """Process all pending receipts: match, suggest or apply splits, expire stale."""
    from datetime import datetime, timezone

    store_path = cfg.receipts_store_path
    window_days = cfg.receipts_match_window_days
    expiry_days = cfg.receipts_expiry_days
    mode = cfg.receipts_mode
    threshold = cfg.receipts_threshold

    pending = receipt_store.list_pending(store_path)

    for receipt_meta in pending:
        receipt_id = receipt_meta["id"]
        ocr = receipt_meta.get("ocr", {})

        # Expiry check
        received_ts_str = receipt_meta.get("received_ts", "")
        try:
            received_ts = datetime.fromisoformat(received_ts_str)
            age_days = (datetime.now(timezone.utc) - received_ts).days
        except (ValueError, TypeError):
            age_days = 0

        if age_days > expiry_days:
            receipt_store.save_done(store_path, receipt_id, "expired")
            audit._write({
                "event": "receipt_expired",
                "pipeline": "receipt",
                "receipt_id": receipt_id,
                "age_days": age_days,
            })
            continue

        candidates = find_receipt_match(
            actual.session, receipt_meta, window_days, cfg.duplicates_defer_pending
        )

        if len(candidates) == 0:
            # Still waiting for the bank transaction to post — leave as pending
            audit._write({
                "event": "receipt_no_match",
                "pipeline": "receipt",
                "receipt_id": receipt_id,
                "merchant": ocr.get("merchant"),
                "total_cents": ocr.get("total_cents"),
            })
            continue

        if len(candidates) > 1:
            # Ambiguous — hold rather than guess
            audit._write({
                "event": "receipt_ambiguous",
                "pipeline": "receipt",
                "receipt_id": receipt_id,
                "candidate_count": len(candidates),
                "merchant": ocr.get("merchant"),
                "total_cents": ocr.get("total_cents"),
            })
            continue

        txn = candidates[0]
        splits = _splits_from_ocr(ocr)
        prior_category = txn.category_id
        confidence = ocr.get("confidence", "high")

        # Apply only when mode is "apply" AND OCR confidence meets the threshold.
        # Low-confidence results fall back to suggest regardless of mode, so a
        # human can verify before the split is written — a re-submitted photo
        # of the same transaction will retry and can finalize it.
        should_apply = mode == "apply" and _meets_threshold(confidence, threshold)

        if should_apply:
            _apply_splits(actual.session, txn, splits, schema_text)
            receipt_store.save_done(
                store_path, receipt_id, "applied",
                matched_txn_id=txn.id,
                splits=splits,
                prior_category=prior_category,
            )
            audit._write({
                "event": "receipt_split",
                "pipeline": "receipt",
                "receipt_id": receipt_id,
                "transaction_id": txn.id,
                "action": "applied",
                "mode": mode,
                "confidence": confidence,
                "merchant": ocr.get("merchant"),
                "total_cents": ocr.get("total_cents"),
                "prior_category_id": prior_category,
                "split_count": len(splits),
            })
        else:
            # Suggest mode or low-confidence fallback: tag + audit only
            txn.notes = strip_category_tag(txn.notes)
            txn.notes = append_tag(txn.notes, "#ai-suggested-split")
            receipt_store.save_done(
                store_path, receipt_id, "suggested",
                matched_txn_id=txn.id,
                splits=splits,
                prior_category=prior_category,
            )
            audit._write({
                "event": "receipt_split",
                "pipeline": "receipt",
                "receipt_id": receipt_id,
                "transaction_id": txn.id,
                "action": "tagged",
                "mode": mode,
                "confidence": confidence,
                "merchant": ocr.get("merchant"),
                "total_cents": ocr.get("total_cents"),
                "prior_category_id": prior_category,
                "proposed_splits": splits,
            })


def _apply_splits(session: Any, txn: Any, splits: list[dict[str, Any]], schema_text: str) -> None:
    """Apply receipt splits to an existing transaction.

    Sets is_parent=1, clears category_id, strips any old AI category marker or
    prior suggested-split tag, and creates child splits for each line item.
    """
    from actual.queries import create_split

    from ..categorization import lookup_category_id

    txn.notes = strip_category_tag(txn.notes)
    txn.notes = remove_tag(txn.notes, "#ai-suggested-split")
    txn.notes = append_tag(txn.notes, "#ai-receipt-split")
    txn.category_id = None
    txn.is_parent = 1

    for item in splits:
        child = create_split(session, txn, -item["amount_cents"] / 100)
        child.notes = item.get("description", "")
        cat_id = lookup_category_id(session, item["category"])
        if cat_id:
            child.category_id = cat_id
