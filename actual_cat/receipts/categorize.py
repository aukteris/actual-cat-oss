"""Pass 2 of receipt processing: categorize transcribed line items.

Extraction (pass 1, ocr.py / text.py) now only transcribes line items
(description + amount). This medium-agnostic step assigns each item a budget
category, informed by how the same item description was categorized before
(history.item_hints). It runs on the canonical receipt dict regardless of which
extractor produced it, so image and text receipts share one categorization path.

On any failure (LLM error, malformed response, or a count mismatch between
returned categories and line items) every item is left as "Uncertain" — the
default validate_receipt already assigns — so the receipt still flows through
matching; lookup_category_id simply yields an uncategorized child for those.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .. import history as history_mod

if TYPE_CHECKING:
    from ..config import Config
    from ..llm import LLMClient


def _render_user(merchant: str, items: list[dict[str, Any]], hints: list[str]) -> str:
    lines = []
    for i, (item, hint) in enumerate(zip(items, hints), start=1):
        desc = item.get("description", "")
        amt = item.get("amount_cents")
        amt_str = f"{amt / 100:+.2f} USD" if isinstance(amt, int) else "(unreadable)"
        line = f"{i}. {desc} — {amt_str}"
        if hint:
            line += f"\n   previously categorized as: {hint}"
        lines.append(line)
    items_block = "\n".join(lines)
    return (
        f"Merchant (context): {merchant}\n\n"
        "Line items to categorize (return one category per item, in order):\n"
        f"{items_block}"
    )


def categorize_line_items(
    receipt: dict[str, Any],
    llm: "LLMClient",
    prompts: Any,
    schema_text: str,
    item_history: history_mod.ItemHistory,
    cfg: "Config",
) -> dict[str, Any]:
    """Write a 'category' onto each line item of `receipt`, in place. Returns it.

    No-ops (leaving items as their default "Uncertain") on any failure path.
    """
    items = receipt.get("line_items", [])
    if not items:
        return receipt

    descriptions = [item.get("description", "") for item in items]
    hints = (
        history_mod.item_hints(
            item_history, descriptions,
            top_n=cfg.history_item_top_n, min_count=cfg.history_min_count,
        )
        if cfg.history_enabled
        else ["" for _ in descriptions]
    )

    system = (
        f"{prompts.RECEIPT_CATEGORIZE_SYSTEM}\n\n"
        f"## Category schema\n\n{schema_text}"
    )
    user = _render_user(receipt.get("merchant", ""), items, hints)

    response = llm.complete_json(system, user)
    categories = response.get("categories")
    if not isinstance(categories, list) or len(categories) != len(items):
        # Error or count mismatch — leave items at their "Uncertain" default.
        return receipt

    for item, category in zip(items, categories):
        if isinstance(category, str) and category.strip():
            item["category"] = category.strip()

    return receipt
