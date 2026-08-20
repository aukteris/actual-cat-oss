"""Receipt extractor registry — dispatch raw input to the right parser.

Each inbox receipt carries an `input_kind` discriminator. This module maps that
kind to an extractor that turns the raw input into the canonical structured
receipt dict (`{merchant, date, total_cents, line_items, confidence}`) that the
matching pipeline consumes. Adding a new input format (e.g. PDF) means writing
one extractor and registering it in _EXTRACTORS — nothing downstream changes.

Legacy inbox metas written before input_kind existed have no such key; they
default to "image" so they keep processing unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from .ocr import DEFAULT_BUDGET_SECONDS, DEFAULT_REQUEST_TIMEOUT_SECONDS, ocr_receipt
from .text import parse_receipt_text

if TYPE_CHECKING:
    from ..llm import LLMClient

# A PDF text layer shorter than this is a caption or a stray label around an
# image, not a receipt — the shortest plausible receipt still names a merchant,
# a date and a total. Below the threshold the embedded image is the real
# content, so vision OCR gets it.
MIN_PDF_TEXT_CHARS = 40


def _extract_image(
    meta: dict[str, Any],
    llm: "LLMClient",
    prompts: Any,
    schema_text: str,
    *,
    request_timeout_seconds: float | None,
    budget_seconds: float | None,
    autocrop: bool,
) -> dict[str, Any]:
    image_path = meta.get("image_path", "")
    if not image_path:
        return {"error": "image receipt missing image_path"}
    image_bytes = Path(image_path).read_bytes()
    return ocr_receipt(
        image_bytes,
        llm,
        prompts.RECEIPT_OCR_SYSTEM,
        schema_text,
        request_timeout_seconds=request_timeout_seconds,
        budget_seconds=budget_seconds,
        autocrop=autocrop,
    )


def _extract_pdf(
    meta: dict[str, Any],
    llm: "LLMClient",
    prompts: Any,
    schema_text: str,
    *,
    request_timeout_seconds: float | None,
    budget_seconds: float | None,
    autocrop: bool,
) -> dict[str, Any]:
    """Route a PDF receipt to the text parser or to vision OCR.

    A real text layer wins over an embedded image. Merchant-generated PDF
    receipts are text plus a logo, and OCRing the logo would throw away the
    merchant and total sitting right there in the text. Only a PDF whose text
    is absent or too short to be a receipt — the jsPDF share-sheet exports that
    prompted this path are one page, one image, no text at all — falls through
    to the same vision OCR used for photographed receipts.

    pypdf (not a rasterizer) enumerates page images so both DCTDecode (already
    a JPEG, used as-is) and FlateDecode (re-encoded by pypdf) streams work.
    """
    pdf_path = meta.get("pdf_path", "")
    if not pdf_path:
        return {"error": "pdf receipt missing pdf_path"}

    from pypdf import PdfReader

    try:
        reader = PdfReader(pdf_path)
        image_bytes: bytes | None = None
        text_parts: list[str] = []
        for page in reader.pages:
            if image_bytes is None:
                for img in page.images:
                    image_bytes = img.data
                    break
            page_text = page.extract_text() or ""
            if page_text.strip():
                text_parts.append(page_text)
    except Exception as e:
        return {"error": f"failed to read PDF: {e}"}

    text = "\n".join(text_parts).strip()
    if len(text) >= MIN_PDF_TEXT_CHARS:
        return parse_receipt_text(text, llm, prompts.RECEIPT_TEXT_SYSTEM, schema_text)

    if image_bytes is not None:
        return ocr_receipt(
            image_bytes,
            llm,
            prompts.RECEIPT_OCR_SYSTEM,
            schema_text,
            request_timeout_seconds=request_timeout_seconds,
            budget_seconds=budget_seconds,
            autocrop=autocrop,
        )

    if text:
        return parse_receipt_text(text, llm, prompts.RECEIPT_TEXT_SYSTEM, schema_text)

    return {"error": "PDF receipt has no extractable image or text"}


def _extract_text(
    meta: dict[str, Any],
    llm: "LLMClient",
    prompts: Any,
    schema_text: str,
    *,
    request_timeout_seconds: float | None,  # noqa: ARG001 — uniform extractor signature
    budget_seconds: float | None,  # noqa: ARG001
    autocrop: bool,  # noqa: ARG001
) -> dict[str, Any]:
    text = meta.get("text", "")
    if not text.strip():
        return {"error": "text receipt missing or empty text"}
    return parse_receipt_text(text, llm, prompts.RECEIPT_TEXT_SYSTEM, schema_text)


_EXTRACTORS = {
    "image": _extract_image,
    "text": _extract_text,
    "pdf": _extract_pdf,
}


def extract_receipt(
    meta: dict[str, Any],
    llm: "LLMClient",
    prompts: Any,
    schema_text: str,
    *,
    request_timeout_seconds: float | None = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    budget_seconds: float | None = DEFAULT_BUDGET_SECONDS,
    autocrop: bool = True,
) -> dict[str, Any]:
    """Dispatch a received receipt to its extractor based on input_kind.

    Returns the canonical receipt dict, or {"error": "..."} on failure or an
    unknown kind. The two time bounds apply to extractors that call an LLM per
    image pass (currently only OCR); every extractor accepts them so the registry
    can stay a uniform dispatch.
    """
    kind = meta.get("input_kind", "image")
    extractor = _EXTRACTORS.get(kind)
    if extractor is None:
        return {"error": f"unknown receipt input_kind: {kind!r}"}
    return extractor(
        meta,
        llm,
        prompts,
        schema_text,
        request_timeout_seconds=request_timeout_seconds,
        budget_seconds=budget_seconds,
        autocrop=autocrop,
    )
