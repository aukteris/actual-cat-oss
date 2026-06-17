"""Receipt parsing from unstructured plain text via the text LLM.

The text counterpart to ocr.py: instead of a vision call on image bytes, it
sends the raw receipt text to the text model and funnels the result through
the same shared validator (parse.validate_receipt). No rotation/normalization
applies to text, so this is a single LLM call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .parse import validate_receipt

if TYPE_CHECKING:
    from ..llm import LLMClient


def parse_receipt_text(
    text: str,
    llm: "LLMClient",
    system_prompt: str,
    schema_text: str,
) -> dict[str, Any]:
    """Parse a plain-text receipt into the canonical structured dict.

    Returns a dict with keys: merchant, date, total_cents, line_items, confidence.
    On failure returns {"error": "..."}.
    """
    user = (
        f"Budget category schema:\n\n{schema_text}\n\n"
        "Please extract the receipt data from the receipt text below.\n\n"
        "--- RECEIPT TEXT ---\n"
        f"{text}\n"
        "--- END RECEIPT TEXT ---"
    )

    raw = llm.complete_json(system_prompt, user)
    try:
        return validate_receipt(raw)
    except ValueError as e:
        return {"error": f"text parse error: {e}"}
