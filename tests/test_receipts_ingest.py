"""Unit tests for IMAP email ingestion."""

import email
import imaplib
import json
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from actual_cat.receipts.ingest import _image_attachments, poll_email


def _make_image_email(image_bytes: bytes = b"\xff\xd8\xff\x00", content_type: str = "image/jpeg",
                      subject: str = "receipt", sender: str = "me@example.com") -> bytes:
    msg = MIMEMultipart()
    msg["From"] = sender
    msg["Subject"] = subject
    part = MIMEImage(image_bytes, _subtype=content_type.split("/")[1])
    msg.attach(part)
    return msg.as_bytes()


def _make_text_email(subject="no image", body="hello") -> bytes:
    msg = MIMEText(body)
    msg["From"] = "a@b.com"
    msg["Subject"] = subject
    return msg.as_bytes()


def _make_cfg(store_path, host="imap.example.com", user="u@example.com", mailbox="INBOX"):
    cfg = MagicMock()
    cfg.email_imap_host = host
    cfg.email_user = user
    cfg.email_mailbox = mailbox
    cfg.receipts_store_path = store_path
    return cfg


class TestImageAttachments:
    def test_extracts_jpeg_attachment(self):
        raw = _make_image_email(b"\xff\xd8\xff\xee", "image/jpeg")
        msg = email.message_from_bytes(raw)
        attachments = _image_attachments(msg)
        assert len(attachments) == 1
        _, ext = attachments[0]
        assert ext == ".jpg"

    def test_extracts_png_attachment(self):
        raw = _make_image_email(b"\x89PNG\x00", "image/png")
        msg = email.message_from_bytes(raw)
        attachments = _image_attachments(msg)
        assert len(attachments) == 1
        _, ext = attachments[0]
        assert ext == ".png"

    def test_no_attachments_for_text_email(self):
        msg = email.message_from_bytes(_make_text_email())
        assert _image_attachments(msg) == []


class TestPollEmail:
    def _imap(self, messages: dict[bytes, bytes], monkeypatch):
        """Build a minimal IMAP mock returning the given {msg_id: raw_bytes} map."""
        imap = MagicMock()
        msg_ids = list(messages.keys())
        search_result = b" ".join(msg_ids) if msg_ids else b""
        imap.search.return_value = (None, [search_result])

        def fetch_side_effect(msg_id, fmt):
            raw = messages.get(msg_id, b"")
            return (None, [(None, raw)])

        imap.fetch.side_effect = fetch_side_effect
        imap.__enter__ = lambda s: imap
        imap.__exit__ = MagicMock(return_value=False)
        return imap

    def test_saves_image_to_inbox(self, tmp_path, monkeypatch):
        monkeypatch.setenv("IMAP_PASSWORD", "secret")
        store_path = str(tmp_path / "receipts")
        cfg = _make_cfg(store_path)
        audit = MagicMock()
        audit._write = MagicMock()

        raw_email = _make_image_email(b"\xff\xd8\xff\x00", subject="Receipt")
        imap = self._imap({b"1": raw_email}, monkeypatch)

        with patch("imaplib.IMAP4_SSL", return_value=imap):
            count = poll_email(cfg, audit)

        assert count == 1
        inbox = Path(store_path) / "inbox"
        images = list(inbox.glob("*.jpg"))
        assert len(images) == 1
        jsons = list(inbox.glob("*.json"))
        assert len(jsons) == 1
        meta = json.loads(jsons[0].read_text())
        assert meta["source"] == "email"
        assert meta["status"] == "received"

    def test_marks_seen_after_processing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("IMAP_PASSWORD", "secret")
        store_path = str(tmp_path / "receipts")
        cfg = _make_cfg(store_path)
        audit = MagicMock()
        audit._write = MagicMock()

        raw_email = _make_image_email(b"\xff\xd8\xff\x00")
        imap = self._imap({b"1": raw_email}, monkeypatch)

        with patch("imaplib.IMAP4_SSL", return_value=imap):
            poll_email(cfg, audit)

        imap.store.assert_called_once_with(b"1", "+FLAGS", "\\Seen")

    def test_text_only_email_saved_as_text_receipt(self, tmp_path, monkeypatch):
        monkeypatch.setenv("IMAP_PASSWORD", "secret")
        store_path = str(tmp_path / "receipts")
        cfg = _make_cfg(store_path)
        audit = MagicMock()
        audit._write = MagicMock()

        raw = _make_text_email(subject="Receipt", body="TRADER JOE'S\nBananas 1.99\nTotal 1.99")
        imap = self._imap({b"1": raw}, monkeypatch)
        with patch("imaplib.IMAP4_SSL", return_value=imap):
            count = poll_email(cfg, audit)

        assert count == 1
        jsons = list((Path(store_path) / "inbox").glob("*.json"))
        assert len(jsons) == 1
        meta = json.loads(jsons[0].read_text())
        assert meta["source"] == "email"
        assert meta["input_kind"] == "text"
        assert "TRADER JOE'S" in meta["text"]
        assert "Bananas 1.99" in meta["text"]
        # No image file written for a text receipt
        assert list((Path(store_path) / "inbox").glob("*.jpg")) == []
        imap.store.assert_called_once_with(b"1", "+FLAGS", "\\Seen")

    def test_empty_body_email_marked_seen_no_inbox_entry(self, tmp_path, monkeypatch):
        monkeypatch.setenv("IMAP_PASSWORD", "secret")
        store_path = str(tmp_path / "receipts")
        cfg = _make_cfg(store_path)
        audit = MagicMock()
        audit._write = MagicMock()

        imap = self._imap({b"1": _make_text_email(body="   ")}, monkeypatch)
        with patch("imaplib.IMAP4_SSL", return_value=imap):
            count = poll_email(cfg, audit)

        assert count == 0
        assert list((Path(store_path) / "inbox").glob("*.json")) == []
        imap.store.assert_called_once()

    def test_no_messages_returns_zero(self, tmp_path, monkeypatch):
        monkeypatch.setenv("IMAP_PASSWORD", "secret")
        store_path = str(tmp_path / "receipts")
        cfg = _make_cfg(store_path)
        audit = MagicMock()
        audit._write = MagicMock()

        imap = self._imap({}, monkeypatch)
        with patch("imaplib.IMAP4_SSL", return_value=imap):
            count = poll_email(cfg, audit)

        assert count == 0

    def test_missing_password_raises(self, tmp_path, monkeypatch):
        monkeypatch.delenv("IMAP_PASSWORD", raising=False)
        cfg = _make_cfg(str(tmp_path / "receipts"))
        audit = MagicMock()
        with pytest.raises(ValueError, match="IMAP_PASSWORD"):
            poll_email(cfg, audit)

    def test_imap_error_logged_not_raised(self, tmp_path, monkeypatch):
        monkeypatch.setenv("IMAP_PASSWORD", "secret")
        store_path = str(tmp_path / "receipts")
        cfg = _make_cfg(store_path)
        audit = MagicMock()
        audit._write = MagicMock()

        with patch("imaplib.IMAP4_SSL", side_effect=imaplib.IMAP4.error("connection refused")):
            count = poll_email(cfg, audit)

        assert count == 0
        audit._write.assert_called_once()
        assert "error" in audit._write.call_args[0][0]
