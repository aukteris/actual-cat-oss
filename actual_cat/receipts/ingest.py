"""IMAP email ingestion — poll for unseen receipt emails, save attachments to inbox.

Called from the hourly worker (__main__.py) when email.enabled = true.
Credentials come from IMAP_PASSWORD env var; all other config from Config.

Supported image types mirror the receiver's _ALLOWED_TYPES. PDF is additionally
accepted here (the receiver does not accept it — see receiver.py:_ALLOWED_TYPES).
"""

from __future__ import annotations

import email
import imaplib
import json
import os
import uuid
from datetime import datetime, timezone
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
_PDF_CONTENT_TYPE = "application/pdf"
_PDF_EXTENSION = ".pdf"
# Body representations, not attachments: a multipart/alternative message
# carries the same body as both, and reporting the unused half as "skipped"
# would fire on ordinary mail and drown out a genuinely discarded attachment.
_BODY_CONTENT_TYPES = {"text/plain", "text/html"}


def _attachments(msg: Message) -> list[tuple[bytes, str, str]]:
    """Return (payload_bytes, ext, input_kind) for each image or PDF attachment.

    Matched on content type only, never Content-Disposition: emailed receipt
    PDFs (e.g. from an iOS share sheet) commonly arrive as "inline" rather than
    "attachment", and a disposition check would silently reject them exactly
    like the missing PDF content-type match once did.
    """
    results: list[tuple[bytes, str, str]] = []
    for part in msg.walk():
        ct = (part.get_content_type() or "").lower()
        if ct in _ALLOWED_EXTENSIONS:
            payload = part.get_payload(decode=True)
            if isinstance(payload, bytes):
                results.append((payload, _ALLOWED_EXTENSIONS[ct], "image"))
        elif ct == _PDF_CONTENT_TYPE:
            payload = part.get_payload(decode=True)
            if isinstance(payload, bytes):
                results.append((payload, _PDF_EXTENSION, "pdf"))
    return results


def _skipped_content_types(msg: Message) -> list[str]:
    """Content types of leaf parts that are neither a supported attachment nor
    a representation of the message body, in message order with duplicates
    removed.

    Used only on the body-fallback path, so a discarded attachment type
    announces itself instead of silently becoming the stored receipt.
    """
    skipped: list[str] = []
    for part in msg.walk():
        if part.is_multipart():
            continue
        ct = (part.get_content_type() or "").lower()
        if ct in _ALLOWED_EXTENSIONS or ct == _PDF_CONTENT_TYPE:
            continue
        if ct in _BODY_CONTENT_TYPES and (
            part.get_content_disposition() or ""
        ).lower() != "attachment":
            continue
        if ct not in skipped:
            skipped.append(ct)
    return skipped


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
    """Poll the configured IMAP mailbox for unseen messages with image/PDF attachments.

    Returns the number of receipt attachments (or text-body fallbacks) saved to inbox.
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
                attachments = _attachments(msg)
                subject = msg.get("Subject", "(no subject)")
                sender = msg.get("From", "(unknown)")
                email_extra = {"email_from": sender, "email_subject": subject}

                if attachments:
                    for payload_bytes, ext, kind in attachments:
                        receipt_id = uuid.uuid4().hex
                        receipt_store._store_root(cfg.receipts_store_path)
                        file_path = Path(cfg.receipts_store_path) / "inbox" / f"{receipt_id}{ext}"
                        file_path.write_bytes(payload_bytes)

                        path_key = "image_path" if kind == "image" else "pdf_path"
                        meta = {
                            "id": receipt_id,
                            "status": "received",
                            "source": "email",
                            "input_kind": kind,
                            "received_ts": datetime.now(timezone.utc).isoformat(),
                            path_key: str(file_path),
                            **email_extra,
                        }
                        (Path(cfg.receipts_store_path) / "inbox" / f"{receipt_id}.json").write_text(
                            json.dumps(meta, indent=2)
                        )

                        audit._write({
                            "event": "receipt_email_ingested",
                            "pipeline": "receipt",
                            "receipt_id": receipt_id,
                            "input_kind": kind,
                            **email_extra,
                        })
                        saved += 1
                else:
                    # No image/PDF attachment. Announce anything discarded before
                    # falling back to the plain-text body as a text receipt —
                    # otherwise a silently-dropped attachment type looks identical
                    # to someone genuinely emailing a bare signature.
                    skipped_types = _skipped_content_types(msg)
                    if skipped_types:
                        audit._write({
                            "event": "receipt_email_attachment_skipped",
                            "pipeline": "receipt",
                            "content_types": skipped_types,
                            **email_extra,
                        })

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
