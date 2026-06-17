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

from .ocr import ocr_receipt
from .text import parse_receipt_text

if TYPE_CHECKING:
    from ..llm import LLMClient


def _extract_image(
    meta: dict[str, Any], llm: "LLMClient", prompts: Any, schema_text: str
) -> dict[str, Any]:
    image_path = meta.get("image_path", "")
    if not image_path:
        return {"error": "image receipt missing image_path"}
    image_bytes = Path(image_path).read_bytes()
    return ocr_receipt(image_bytes, llm, prompts.RECEIPT_OCR_SYSTEM, schema_text)


def _extract_text(
    meta: dict[str, Any], llm: "LLMClient", prompts: Any, schema_text: str
) -> dict[str, Any]:
    text = meta.get("text", "")
    if not text.strip():
        return {"error": "text receipt missing or empty text"}
    return parse_receipt_text(text, llm, prompts.RECEIPT_TEXT_SYSTEM, schema_text)


_EXTRACTORS = {
    "image": _extract_image,
    "text": _extract_text,
}


def extract_receipt(
    meta: dict[str, Any],
    llm: "LLMClient",
    prompts: Any,
    schema_text: str,
) -> dict[str, Any]:
    """Dispatch a received receipt to its extractor based on input_kind.

    Returns the canonical receipt dict, or {"error": "..."} on failure or an
    unknown kind.
    """
    kind = meta.get("input_kind", "image")
    extractor = _EXTRACTORS.get(kind)
    if extractor is None:
        return {"error": f"unknown receipt input_kind: {kind!r}"}
    return extractor(meta, llm, prompts, schema_text)
