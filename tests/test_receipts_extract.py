"""Unit tests for the receipt extractor dispatch registry."""

import base64
import io
from pathlib import Path
from types import SimpleNamespace

import pytest

from actual_cat.receipts.extract import extract_receipt

_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "pdf"

_PROMPTS = SimpleNamespace(
    RECEIPT_OCR_SYSTEM="ocr-system",
    RECEIPT_TEXT_SYSTEM="text-system",
)

_GOOD_RESPONSE = {
    "merchant": "Shop",
    "date": "2026-06-01",
    "total_cents": 100,
    "line_items": [{"description": "x", "amount_cents": 100, "category": "Food / Groceries"}],
}


def _make_text_pdf(text: str) -> bytes:
    """Build a minimal one-page PDF whose content stream is a text-showing op.

    Used to exercise the pypdf plumbing for PDFs that have a real text layer —
    distinct from the real jsPDF receipts (image-only, no text layer), which
    are covered by the real-fixture test below.
    """
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    page = writer.add_blank_page(width=200, height=200)

    stream = DecodedStreamObject()
    stream.set_data(f"BT /F1 24 Tf 10 100 Td ({text}) Tj ET".encode())
    stream_ref = writer._add_object(stream)

    font = DictionaryObject()
    font[NameObject("/Type")] = NameObject("/Font")
    font[NameObject("/Subtype")] = NameObject("/Type1")
    font[NameObject("/BaseFont")] = NameObject("/Helvetica")
    resources = DictionaryObject()
    fonts = DictionaryObject()
    fonts[NameObject("/F1")] = writer._add_object(font)
    resources[NameObject("/Font")] = fonts

    page[NameObject("/Contents")] = stream_ref
    page[NameObject("/Resources")] = resources

    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def _make_text_pdf_with_logo(text: str) -> bytes:
    """A one-page PDF carrying both a real text layer and a small JPEG logo.

    The shape of a merchant-generated PDF receipt, and the case where routing
    on "has an image" alone would OCR the logo and discard the text.
    """
    from PIL import Image
    from pypdf import PdfReader, PdfWriter

    logo_buf = io.BytesIO()
    Image.new("RGB", (16, 16), "red").save(logo_buf, format="PDF")
    logo_page = PdfReader(logo_buf).pages[0]

    writer = PdfWriter()
    writer.append(io.BytesIO(_make_text_pdf(text)))
    writer.pages[0].merge_page(logo_page)

    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def _make_blank_pdf() -> bytes:
    """A one-page PDF with no content stream — no image, no text."""
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


class MockLLM:
    def __init__(self, response: dict):
        self._response = response
        self.text_calls: list[tuple] = []
        self.vision_calls: list[tuple] = []

    def complete_json(self, system: str, user: str) -> dict:
        self.text_calls.append((system, user))
        return self._response

    def complete_json_vision(self, system, user, image_b64, media_type, *, timeout=None) -> dict:
        self.vision_calls.append((system, user, image_b64, media_type, timeout))
        return self._response


def test_text_kind_dispatches_to_text_parser():
    llm = MockLLM(_GOOD_RESPONSE)
    meta = {"input_kind": "text", "text": "TRADER JOE'S\nProduce 1.00\nTotal 1.00"}
    result = extract_receipt(meta, llm, _PROMPTS, "schema")
    assert result["merchant"] == "Shop"
    assert len(llm.text_calls) == 1
    assert len(llm.vision_calls) == 0
    assert llm.text_calls[0][0] == "text-system"


def test_image_kind_dispatches_to_ocr(tmp_path):
    image_path = tmp_path / "r.jpg"
    image_path.write_bytes(b"\xff\xd8\xff" + b"\x00" * 10)  # minimal JPEG magic
    llm = MockLLM(_GOOD_RESPONSE)
    meta = {"input_kind": "image", "image_path": str(image_path)}
    result = extract_receipt(meta, llm, _PROMPTS, "schema")
    assert result["merchant"] == "Shop"
    assert len(llm.vision_calls) == 1
    assert len(llm.text_calls) == 0
    assert llm.vision_calls[0][0] == "ocr-system"


def test_missing_input_kind_defaults_to_image(tmp_path):
    image_path = tmp_path / "r.jpg"
    image_path.write_bytes(b"\xff\xd8\xff" + b"\x00" * 10)
    llm = MockLLM(_GOOD_RESPONSE)
    meta = {"image_path": str(image_path)}  # legacy meta, no input_kind
    result = extract_receipt(meta, llm, _PROMPTS, "schema")
    assert result["merchant"] == "Shop"
    assert len(llm.vision_calls) == 1


def test_unknown_kind_returns_error():
    llm = MockLLM(_GOOD_RESPONSE)
    meta = {"input_kind": "csv", "payload": "..."}
    result = extract_receipt(meta, llm, _PROMPTS, "schema")
    assert "error" in result
    assert "csv" in result["error"]


def test_text_kind_missing_text_returns_error():
    llm = MockLLM(_GOOD_RESPONSE)
    result = extract_receipt({"input_kind": "text", "text": "   "}, llm, _PROMPTS, "schema")
    assert "error" in result


def test_image_kind_missing_path_returns_error():
    llm = MockLLM(_GOOD_RESPONSE)
    result = extract_receipt({"input_kind": "image"}, llm, _PROMPTS, "schema")
    assert "error" in result


def test_ocr_time_bounds_reach_the_vision_call(tmp_path):
    image_path = tmp_path / "r.jpg"
    image_path.write_bytes(b"\xff\xd8\xff" + b"\x00" * 10)
    llm = MockLLM(_GOOD_RESPONSE)
    meta = {"input_kind": "image", "image_path": str(image_path)}
    extract_receipt(
        meta, llm, _PROMPTS, "schema",
        request_timeout_seconds=30, budget_seconds=600,
    )
    # Request timeout is the smaller bound, so it's the one that applies.
    assert llm.vision_calls[0][4] == 30


def test_text_kind_ignores_ocr_time_bounds():
    llm = MockLLM(_GOOD_RESPONSE)
    meta = {"input_kind": "text", "text": "SHOP\nx 1.00\nTotal 1.00"}
    result = extract_receipt(
        meta, llm, _PROMPTS, "schema",
        request_timeout_seconds=30, budget_seconds=600,
    )
    assert result["merchant"] == "Shop"
    assert len(llm.text_calls) == 1


# ---------------------------------------------------------------------------
# PDF kind — ISSUE-001, emailed PDF receipts were silently discarded
# ---------------------------------------------------------------------------


def test_pdf_kind_missing_path_returns_error():
    llm = MockLLM(_GOOD_RESPONSE)
    result = extract_receipt({"input_kind": "pdf"}, llm, _PROMPTS, "schema")
    assert "error" in result


def test_pdf_kind_unreadable_file_returns_error(tmp_path):
    pdf_path = tmp_path / "r.pdf"
    pdf_path.write_bytes(b"not a pdf at all")
    llm = MockLLM(_GOOD_RESPONSE)
    meta = {"input_kind": "pdf", "pdf_path": str(pdf_path)}
    result = extract_receipt(meta, llm, _PROMPTS, "schema")
    assert "error" in result


def test_pdf_kind_no_image_no_text_returns_error(tmp_path):
    pdf_path = tmp_path / "r.pdf"
    pdf_path.write_bytes(_make_blank_pdf())
    llm = MockLLM(_GOOD_RESPONSE)
    meta = {"input_kind": "pdf", "pdf_path": str(pdf_path)}
    result = extract_receipt(meta, llm, _PROMPTS, "schema")
    assert "error" in result
    assert len(llm.vision_calls) == 0
    assert len(llm.text_calls) == 0


def test_pdf_kind_text_only_dispatches_to_text_parser(tmp_path):
    pdf_path = tmp_path / "r.pdf"
    pdf_path.write_bytes(_make_text_pdf("TRADER JOES"))
    llm = MockLLM(_GOOD_RESPONSE)
    meta = {"input_kind": "pdf", "pdf_path": str(pdf_path)}
    result = extract_receipt(meta, llm, _PROMPTS, "schema")
    assert result["merchant"] == "Shop"
    assert len(llm.text_calls) == 1
    assert len(llm.vision_calls) == 0
    assert llm.text_calls[0][0] == "text-system"
    assert "TRADER JOES" in llm.text_calls[0][1]


def _assert_routed_to_vision_as_jpeg(llm: "MockLLM") -> None:
    assert len(llm.vision_calls) == 1
    assert len(llm.text_calls) == 0
    assert llm.vision_calls[0][0] == "ocr-system"
    image_bytes = base64.b64decode(llm.vision_calls[0][2])
    assert image_bytes[:3] == b"\xff\xd8\xff"  # JPEG magic — the embedded /DCTDecode stream
    assert llm.vision_calls[0][3] == "image/jpeg"


def test_pdf_kind_with_embedded_image_dispatches_to_vision_ocr():
    """The bug's shape: a jsPDF share-sheet export — one page, no text layer,
    one embedded JPEG (/DCTDecode). Exactly the four receipts lost to ISSUE-001.

    The committed fixture is generated (see make_receipt_fixture.py); the real
    receipts are personal financial records and this repo is public.
    """
    llm = MockLLM(_GOOD_RESPONSE)
    meta = {"input_kind": "pdf", "pdf_path": str(_FIXTURE_DIR / "jspdf_image_only_receipt.pdf")}
    result = extract_receipt(meta, llm, _PROMPTS, "schema")

    assert result["merchant"] == "Shop"
    _assert_routed_to_vision_as_jpeg(llm)


@pytest.mark.parametrize("fixture", sorted(_FIXTURE_DIR.glob("real_*.pdf")), ids=lambda p: p.name)
def test_pdf_kind_real_receipts_dispatch_to_vision_ocr(fixture):
    """Same assertion against the genuine receipts, when they are present.

    These are gitignored, so this parametrization is empty in CI and on a fresh
    clone. Drop a real receipt into tests/fixtures/pdf/real_*.pdf to check the
    generated fixture has not drifted from what the mailbox actually delivers.
    """
    llm = MockLLM(_GOOD_RESPONSE)
    result = extract_receipt(
        {"input_kind": "pdf", "pdf_path": str(fixture)}, llm, _PROMPTS, "schema"
    )

    assert result["merchant"] == "Shop"
    _assert_routed_to_vision_as_jpeg(llm)


def test_pdf_kind_prefers_text_layer_over_an_embedded_logo(tmp_path):
    """A text-layer receipt with a logo must not OCR the logo.

    Merchant-generated PDF receipts are text plus a small image; routing those
    to vision would discard the merchant and total present in the text layer.
    """
    pdf_path = tmp_path / "r.pdf"
    pdf_path.write_bytes(
        _make_text_pdf_with_logo("TRADER JOES 123 MAIN ST  SUBTOTAL 42.10  TOTAL 45.99")
    )
    llm = MockLLM(_GOOD_RESPONSE)
    result = extract_receipt(
        {"input_kind": "pdf", "pdf_path": str(pdf_path)}, llm, _PROMPTS, "schema"
    )

    assert result["merchant"] == "Shop"
    assert len(llm.text_calls) == 1
    assert len(llm.vision_calls) == 0
    assert "TRADER JOES" in llm.text_calls[0][1]


def test_pdf_kind_trivial_text_with_image_still_dispatches_to_vision(tmp_path):
    """A caption-length text layer is not a receipt — the image is the content."""
    pdf_path = tmp_path / "r.pdf"
    pdf_path.write_bytes(_make_text_pdf_with_logo("scan"))
    llm = MockLLM(_GOOD_RESPONSE)
    extract_receipt({"input_kind": "pdf", "pdf_path": str(pdf_path)}, llm, _PROMPTS, "schema")

    _assert_routed_to_vision_as_jpeg(llm)
