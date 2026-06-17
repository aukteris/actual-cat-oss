"""Source-neutral receipt normalization and confidence scoring.

Every receipt extractor (vision OCR, text, future formats) funnels its raw
LLM JSON through `validate_receipt` to produce the canonical receipt dict
`{merchant, date, total_cents, line_items, confidence}`. This logic is not
image-specific — it operates on the already-parsed structure.
"""

from __future__ import annotations

from typing import Any

# Rounding-only tolerance: diffs within this many cents are nudged onto the
# largest item and scored as high confidence. Larger diffs signal missing/wrong
# items and are left untouched (low confidence).
TOLERANCE_CENTS = 3


def compute_confidence(items: list[dict[str, Any]], total_cents: int) -> tuple[str, int]:
    """Return (confidence, abs_diff).

    High confidence requires: all items have readable (int) amounts AND the sum
    is within TOLERANCE_CENTS of total_cents. Everything else is low.
    """
    readable = [
        i["amount_cents"] for i in items
        if isinstance(i.get("amount_cents"), int)
    ]
    if len(readable) < len(items):
        return "low", total_cents
    diff = total_cents - sum(readable)
    if abs(diff) <= TOLERANCE_CENTS:
        return "high", abs(diff)
    return "low", abs(diff)


def validate_receipt(data: dict[str, Any]) -> dict[str, Any]:
    """Validate required fields, score confidence, optionally nudge for rounding.

    - Negative amount_cents are allowed (discounts/credits).
    - Null/non-int amount_cents are kept as None (unreadable) — they force low confidence.
    - Raises ValueError only on structurally invalid responses.
    """
    if "error" in data:
        raise ValueError(data["error"])

    merchant = data.get("merchant")
    if not isinstance(merchant, str) or not merchant.strip():
        raise ValueError(f"missing or blank merchant: {data!r}")

    total_cents = data.get("total_cents")
    if not isinstance(total_cents, int) or total_cents <= 0:
        raise ValueError(f"total_cents must be a positive int: {data!r}")

    raw_items = data.get("line_items")
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError(f"line_items must be a non-empty list: {data!r}")

    items: list[dict[str, Any]] = []
    for item in raw_items:
        desc = item.get("description", "")
        amt = item.get("amount_cents")
        cat = item.get("category", "Uncertain")
        # Accept int (including negative for discounts); coerce anything else to None.
        if not isinstance(amt, int):
            amt = None
        items.append({"description": str(desc), "amount_cents": amt, "category": str(cat)})

    confidence, _ = compute_confidence(items, total_cents)

    # For high confidence (rounding-only diff), nudge the largest item to make
    # the sum exact. Never modify amounts for low-confidence results.
    if confidence == "high":
        readable_amounts = [
            i["amount_cents"] for i in items if isinstance(i.get("amount_cents"), int)
        ]
        diff = total_cents - sum(readable_amounts)
        if diff != 0:
            largest_idx = max(
                (i for i, it in enumerate(items) if isinstance(it["amount_cents"], int)),
                key=lambda i: items[i]["amount_cents"],
            )
            items[largest_idx]["amount_cents"] += diff

    return {
        "merchant": merchant.strip(),
        "date": data.get("date"),
        "total_cents": total_cents,
        "line_items": items,
        "confidence": confidence,
    }
