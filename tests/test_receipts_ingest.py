"""Unit tests for IMAP email ingestion."""

import email
import imaplib
import json
from email.mime.application import MIMEApplication
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from actual_cat.receipts.ingest import _attachments, _skipped_content_types, poll_email


def _make_image_email(image_bytes: bytes = b"\xff\xd8\xff\x00", content_type: str = "image/jpeg",
                      subject: str = "receipt", sender: str = "me@example.com") -> bytes:
    msg = MIMEMultipart()
    msg["From"] = sender
    msg["Subject"] = subject
    part = MIMEImage(image_bytes, _subtype=content_type.split("/")[1])
    msg.attach(part)
    return msg.as_bytes()


def _make_pdf_email(pdf_bytes: bytes = b"%PDF-1.4 fake", subject: str = "receipt",
                    sender: str = "me@example.com", inline: bool = True,
                    extra_text_parts: tuple[str, ...] = ()) -> bytes:
    """Build a multipart/mixed message carrying a PDF part.

    Mirrors the real jsPDF receipts: PDF part disposition is "inline" (not
    "attachment") by default, since that's what the confirmed bug report found.
    """
    msg = MIMEMultipart()
    msg["From"] = sender
    msg["Subject"] = subject
    for text in extra_text_parts:
        msg.attach(MIMEText(text))
    part = MIMEApplication(pdf_bytes, _subtype="pdf")
    disposition = "inline" if inline else "attachment"
    part.add_header("Content-Disposition", disposition, filename="receipt.pdf")
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


class TestAttachments:
    def test_extracts_jpeg_attachment(self):
        raw = _make_image_email(b"\xff\xd8\xff\xee", "image/jpeg")
        msg = email.message_from_bytes(raw)
        attachments = _attachments(msg)
        assert len(attachments) == 1
        _, ext, kind = attachments[0]
        assert ext == ".jpg"
        assert kind == "image"

    def test_extracts_png_attachment(self):
        raw = _make_image_email(b"\x89PNG\x00", "image/png")
        msg = email.message_from_bytes(raw)
        attachments = _attachments(msg)
        assert len(attachments) == 1
        _, ext, kind = attachments[0]
        assert ext == ".png"
        assert kind == "image"

    def test_no_attachments_for_text_email(self):
        msg = email.message_from_bytes(_make_text_email())
        assert _attachments(msg) == []

    def test_extracts_pdf_attachment_regardless_of_content_disposition(self):
        """The confirmed bug: real jsPDF receipts arrive inline, not attachment."""
        for inline in (True, False):
            raw = _make_pdf_email(b"%PDF-1.4 fake", inline=inline)
            msg = email.message_from_bytes(raw)
            attachments = _attachments(msg)
            assert len(attachments) == 1
            payload, ext, kind = attachments[0]
            assert ext == ".pdf"
            assert kind == "pdf"
            assert payload == b"%PDF-1.4 fake"


class TestSkippedContentTypes:
    def test_pdf_not_reported_skipped(self):
        msg = email.message_from_bytes(_make_pdf_email())
        assert _skipped_content_types(msg) == []

    def test_body_text_plain_not_reported_skipped(self):
        msg = email.message_from_bytes(_make_text_email(body="hello"))
        assert _skipped_content_types(msg) == []

    def test_html_body_alternative_not_reported_skipped(self):
        """multipart/alternative carries the body twice — the unused HTML half
        is not a discarded attachment, and reporting it would fire this event on
        ordinary mail and drown out a genuine one."""
        msg = MIMEMultipart("alternative")
        msg["From"] = "a@b.com"
        msg["Subject"] = "receipt"
        msg.attach(MIMEText("Coffee $4.00", "plain"))
        msg.attach(MIMEText("<p>Coffee $4.00</p>", "html"))
        assert _skipped_content_types(email.message_from_bytes(msg.as_bytes())) == []

    def test_attached_html_file_reported_skipped(self):
        """An HTML *attachment* is a discarded receipt, unlike an HTML body."""
        msg = MIMEMultipart()
        msg["From"] = "a@b.com"
        msg["Subject"] = "receipt"
        msg.attach(MIMEText("Sent from my iPhone"))
        page = MIMEText("<p>total</p>", "html")
        page.add_header("Content-Disposition", "attachment", filename="receipt.html")
        msg.attach(page)
        assert _skipped_content_types(email.message_from_bytes(msg.as_bytes())) == ["text/html"]

    def test_unsupported_part_reported_skipped(self):
        msg = MIMEMultipart()
        msg["From"] = "a@b.com"
        msg["Subject"] = "receipt"
        msg.attach(MIMEText("Sent from my iPhone"))
        docx = MIMEApplication(b"docbytes", _subtype="vnd.openxmlformats-officedocument")
        docx.add_header("Content-Disposition", "inline", filename="receipt.docx")
        msg.attach(docx)
        assert _skipped_content_types(email.message_from_bytes(msg.as_bytes())) == [
            "application/vnd.openxmlformats-officedocument"
        ]


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

    def test_saves_pdf_to_inbox_with_pdf_input_kind(self, tmp_path, monkeypatch):
        monkeypatch.setenv("IMAP_PASSWORD", "secret")
        store_path = str(tmp_path / "receipts")
        cfg = _make_cfg(store_path)
        audit = MagicMock()
        audit._write = MagicMock()

        raw_email = _make_pdf_email(b"%PDF-1.4 fake pdf bytes", subject="Receipt")
        imap = self._imap({b"1": raw_email}, monkeypatch)

        with patch("imaplib.IMAP4_SSL", return_value=imap):
            count = poll_email(cfg, audit)

        assert count == 1
        inbox = Path(store_path) / "inbox"
        pdfs = list(inbox.glob("*.pdf"))
        assert len(pdfs) == 1
        assert pdfs[0].read_bytes() == b"%PDF-1.4 fake pdf bytes"
        jsons = list(inbox.glob("*.json"))
        assert len(jsons) == 1
        meta = json.loads(jsons[0].read_text())
        assert meta["source"] == "email"
        assert meta["input_kind"] == "pdf"
        assert meta["pdf_path"] == str(pdfs[0])
        assert "image_path" not in meta
        ingested_events = [
            c.args[0] for c in audit._write.call_args_list
            if c.args[0]["event"] == "receipt_email_ingested"
        ]
        assert len(ingested_events) == 1
        assert ingested_events[0]["input_kind"] == "pdf"

    def test_skipped_attachment_logged_before_text_fallback(self, tmp_path, monkeypatch):
        """The core of this bug: a discarded attachment must announce itself,
        even though the plain-text body still gets stored as a (likely bogus)
        receipt — same as an email with no attachment at all."""
        monkeypatch.setenv("IMAP_PASSWORD", "secret")
        store_path = str(tmp_path / "receipts")
        cfg = _make_cfg(store_path)
        audit = MagicMock()
        audit._write = MagicMock()

        msg = MIMEMultipart()
        msg["From"] = "a@b.com"
        msg["Subject"] = "receipt"
        msg.attach(MIMEText("Sent from my iPhone"))
        weird = MIMEApplication(b"data", _subtype="vnd.ms-excel")
        weird.add_header("Content-Disposition", "inline", filename="receipt.xls")
        msg.attach(weird)
        imap = self._imap({b"1": msg.as_bytes()}, monkeypatch)

        with patch("imaplib.IMAP4_SSL", return_value=imap):
            poll_email(cfg, audit)

        events = [c.args[0]["event"] for c in audit._write.call_args_list]
        assert "receipt_email_attachment_skipped" in events
        skipped = next(
            c.args[0] for c in audit._write.call_args_list
            if c.args[0]["event"] == "receipt_email_attachment_skipped"
        )
        assert skipped["content_types"] == ["application/vnd.ms-excel"]
        # Skip event fires strictly before the fallback ingest event.
        skip_idx = events.index("receipt_email_attachment_skipped")
        ingest_idx = events.index("receipt_email_ingested")
        assert skip_idx < ingest_idx

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
