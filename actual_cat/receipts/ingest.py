"""IMAP email ingestion — poll for unseen receipt emails, save attachments to inbox.

Called from the hourly worker (__main__.py) when email.enabled = true.
Credentials come from IMAP_PASSWORD env var; all other config from Config.

Supported attachment types mirror the receiver's _ALLOWED_TYPES.
"""

from __future__ import annotations

import email
import imaplib
import os
from email.message import Message
from pathlib import Path
from typing import TYPE_CHECKING

from . import store as receipt_store

if TYPE_CHECKING:
    from ..audit import AuditLogger
    from ..config import Config

_ALLOWED_EXTENSIONS = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/heic": ".heic",
    "image/heif": ".heif",
}


def _image_attachments(msg: Message) -> list[tuple[bytes, str]]:
    """Return (payload_bytes, ext) for each image attachment in the message."""
    results = []
    for part in msg.walk():
        ct = (part.get_content_type() or "").lower()
        if ct in _ALLOWED_EXTENSIONS:
            payload = part.get_payload(decode=True)
            if isinstance(payload, bytes):
                results.append((payload, _ALLOWED_EXTENSIONS[ct]))
    return results


def _text_body(msg: Message) -> str | None:
    """Return the message's plain-text body, or None if absent/blank.

    Used as a fallback when an email carries no image attachment — the body
    itself is treated as a plain-text receipt. Skips parts explicitly marked
    as attachments (e.g. an attached .txt file).
    """
    parts: list[str] = []
    for part in msg.walk():
        if (part.get_content_type() or "").lower() != "text/plain":
            continue
        if (part.get_content_disposition() or "").lower() == "attachment":
            continue
        payload = part.get_payload(decode=True)
        if not isinstance(payload, bytes):
            continue
        charset = part.get_content_charset() or "utf-8"
        try:
            parts.append(payload.decode(charset, errors="replace"))
        except (LookupError, ValueError):
            parts.append(payload.decode("utf-8", errors="replace"))
    body = "\n".join(parts).strip()
    return body or None


def poll_email(cfg: "Config", audit: "AuditLogger") -> int:
    """Poll the configured IMAP mailbox for unseen messages with image attachments.

    Returns the number of receipt images saved to inbox.
    """
    password = os.environ.get("IMAP_PASSWORD", "")
    if not password:
        raise ValueError("IMAP_PASSWORD not set")

    saved = 0
    try:
        with imaplib.IMAP4_SSL(cfg.email_imap_host) as imap:
            imap.login(cfg.email_user, password)
            imap.select(cfg.email_mailbox)

            _, msg_ids_raw = imap.search(None, "UNSEEN")
            msg_ids = msg_ids_raw[0].split() if msg_ids_raw[0] else []

            for msg_id in msg_ids:
                _, msg_data = imap.fetch(msg_id, "(RFC822)")
                raw = msg_data[0][1] if msg_data and msg_data[0] else None
                if not raw:
                    continue

                msg = email.message_from_bytes(raw)  # type: ignore[arg-type]
                attachments = _image_attachments(msg)
                subject = msg.get("Subject", "(no subject)")
                sender = msg.get("From", "(unknown)")
                email_extra = {"email_from": sender, "email_subject": subject}

                if attachments:
                    import json
                    import uuid
                    from datetime import datetime, timezone

                    for image_bytes, ext in attachments:
                        receipt_id = uuid.uuid4().hex
                        receipt_store._store_root(cfg.receipts_store_path)
                        image_path = Path(cfg.receipts_store_path) / "inbox" / f"{receipt_id}{ext}"
                        image_path.write_bytes(image_bytes)

                        meta = {
                            "id": receipt_id,
                            "status": "received",
                            "source": "email",
                            "input_kind": "image",
                            "received_ts": datetime.now(timezone.utc).isoformat(),
                            "image_path": str(image_path),
                            **email_extra,
                        }
                        (Path(cfg.receipts_store_path) / "inbox" / f"{receipt_id}.json").write_text(
                            json.dumps(meta, indent=2)
                        )

                        audit._write({
                            "event": "receipt_email_ingested",
                            "pipeline": "receipt",
                            "receipt_id": receipt_id,
                            "input_kind": "image",
                            **email_extra,
                        })
                        saved += 1
                else:
                    # No image attachment — fall back to the plain-text body as
                    # a text receipt. Emails with neither are just marked seen.
                    body = _text_body(msg)
                    if body:
                        receipt_id = receipt_store.save_received_text(
                            cfg.receipts_store_path, body, source="email", extra=email_extra
                        )
                        audit._write({
                            "event": "receipt_email_ingested",
                            "pipeline": "receipt",
                            "receipt_id": receipt_id,
                            "input_kind": "text",
                            **email_extra,
                        })
                        saved += 1

                imap.store(msg_id, "+FLAGS", "\\Seen")

    except imaplib.IMAP4.error as e:
        audit._write({
            "event": "receipt_email_error",
            "pipeline": "receipt",
            "error": str(e),
        })

    return saved
