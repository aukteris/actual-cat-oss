"""Receipt HTTP receiver — minimal FastAPI service for iOS Shortcut POSTs.

Run as a standalone service:
    python -m actual_cat.receipts.receiver

Accepts multipart/form-data POST to /receipts/ with:
  - image: the receipt image file (required)
  - hint_merchant: optional merchant hint
  - hint_date: optional date hint (YYYY-MM-DD)

Bearer token auth via RECEIPT_RECEIVER_TOKEN env var.
Store path via RECEIPTS_STORE_PATH env var (default: "receipts").
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from . import store as receipt_store

app = FastAPI(title="actual-cat receipt receiver", docs_url=None, redoc_url=None)

_bearer = HTTPBearer()


def _get_token() -> str:
    token = os.environ.get("RECEIPT_RECEIVER_TOKEN", "")
    if not token:
        raise RuntimeError("RECEIPT_RECEIVER_TOKEN is not set")
    return token


def _verify_token(credentials: HTTPAuthorizationCredentials = Depends(_bearer)) -> str:
    expected = _get_token()
    if credentials.credentials != expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    return credentials.credentials


def _store_path() -> str:
    return os.environ.get("RECEIPTS_STORE_PATH", "receipts")


_ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/heic", "image/heif"}
_MAX_BYTES = 20 * 1024 * 1024  # 20 MB
_MAX_TEXT_BYTES = 1 * 1024 * 1024  # 1 MB — a receipt's text is tiny; cap abuse


@app.post("/receipts/", status_code=status.HTTP_201_CREATED)
async def receive_receipt(
    image: UploadFile = File(..., description="Receipt image"),
    hint_merchant: str | None = Form(default=None),
    hint_date: str | None = Form(default=None),
    _token: str = Depends(_verify_token),
) -> dict[str, str]:
    content_type = image.content_type or "application/octet-stream"
    if content_type not in _ALLOWED_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported image type: {content_type}",
        )

    image_bytes = await image.read()
    if len(image_bytes) > _MAX_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Image exceeds 20 MB limit",
        )

    store_path = _store_path()
    ext = _ext_for(content_type)

    # Write image file into inbox before creating the meta record
    receipt_store._store_root(store_path)  # ensure dirs exist
    receipt_id = __import__("uuid").uuid4().hex
    image_path = Path(store_path) / "inbox" / f"{receipt_id}{ext}"
    image_path.write_bytes(image_bytes)

    meta_extra: dict[str, str] = {}
    if hint_merchant:
        meta_extra["hint_merchant"] = hint_merchant
    if hint_date:
        meta_extra["hint_date"] = hint_date

    # Write meta JSON manually so we can embed the hints
    import json
    from datetime import datetime, timezone
    meta = {
        "id": receipt_id,
        "status": "received",
        "source": "ios",
        "input_kind": "image",
        "received_ts": datetime.now(timezone.utc).isoformat(),
        "image_path": str(image_path),
        **meta_extra,
    }
    (Path(store_path) / "inbox" / f"{receipt_id}.json").write_text(json.dumps(meta, indent=2))

    return {"receipt_id": receipt_id, "status": "received"}


@app.post("/receipts/text", status_code=status.HTTP_201_CREATED)
async def receive_receipt_text(
    text: str = Form(..., description="Plain-text receipt content"),
    hint_merchant: str | None = Form(default=None),
    hint_date: str | None = Form(default=None),
    _token: str = Depends(_verify_token),
) -> dict[str, str]:
    if not text.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Receipt text is empty",
        )
    if len(text.encode("utf-8")) > _MAX_TEXT_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Receipt text exceeds 1 MB limit",
        )

    extra: dict[str, str] = {}
    if hint_merchant:
        extra["hint_merchant"] = hint_merchant
    if hint_date:
        extra["hint_date"] = hint_date

    receipt_id = receipt_store.save_received_text(_store_path(), text, source="ios", extra=extra)
    return {"receipt_id": receipt_id, "status": "received"}


def _ext_for(content_type: str) -> str:
    return {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "image/heic": ".heic",
        "image/heif": ".heif",
    }.get(content_type, ".bin")


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("RECEIVER_HOST", "127.0.0.1")
    port = int(os.environ.get("RECEIVER_PORT", "8001"))
    uvicorn.run("actual_cat.receipts.receiver:app", host=host, port=port, log_level="info")
