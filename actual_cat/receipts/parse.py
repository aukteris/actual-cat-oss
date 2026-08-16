"""Source-neutral receipt normalization and confidence scoring.

Every receipt extractor (vision OCR, text, future formats) funnels its raw
LLM JSON through `validate_receipt` to produce the canonical receipt dict
`{merchant, date, total_cents, line_items, confidence}`. This logic is not
image-specific — it operates on the already-parsed structure.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

import phonenumbers
import pyap
import pycountry
from dateutil import parser as date_parser
from phonenumbers import PhoneNumberMatcher

# Rounding-only tolerance: diffs within this many cents are nudged onto the
# largest item and scored as high confidence. Larger diffs signal missing/wrong
# items and are left untouched (low confidence).
TOLERANCE_CENTS = 3

# Date format conventions by country code. Extend both this and infer_country_code
# together as new countries are added.
_DATE_FORMAT_BY_COUNTRY = {
    # North America
    "US": "MDY",
    "CA": "MDY",
    "MX": "DMY",
    # Europe (major countries)
    "GB": "DMY",
    "IE": "DMY",
    "FR": "DMY",
    "DE": "DMY",
    "ES": "DMY",
    "IT": "DMY",
    "NL": "DMY",
    "BE": "DMY",
    "AT": "DMY",
    "CH": "DMY",
    "PT": "DMY",
    "PL": "DMY",
    "SE": "DMY",
    "NO": "DMY",
    "DK": "DMY",
    "FI": "DMY",
    # Oceania
    "AU": "DMY",
    "NZ": "DMY",
    # Asia
    "JP": "YMD",
}


def infer_country_code(location_raw: str | None) -> str | None:
    """Infer a country's ISO-3166 alpha-2 code from raw location text.

    Tries methods in order of reliability:
    1. Phone number region detection (most reliable when present)
    2. pyap address parsing (structured address patterns)
    3. Country name substring match (fallback, least reliable)

    Returns None if no country can be confidently identified.
    """
    if not location_raw:
        return None

    # Method 1: Phone number detection (most reliable)
    try:
        for match in PhoneNumberMatcher(location_raw, "US"):
            raw = match.raw_string.strip()
            if raw.startswith("+") or raw.startswith("00"):
                region = phonenumbers.region_code_for_number(match.number)
                if region and region != "ZZ":
                    return region
    except Exception:
        pass

    # Method 2: pyap address parsing (try major countries)
    for country in ["US", "CA", "GB", "IE", "AU", "NZ", "DE", "FR", "MX", "JP"]:
        try:
            if pyap.parse(location_raw, country=country):
                return country
        except Exception:
            continue

    # Method 3: Country name substring match
    lower = location_raw.lower()
    for entry in pycountry.countries:
        names = {
            entry.name.lower(),
            getattr(entry, "common_name", "").lower(),
            getattr(entry, "official_name", "").lower(),
        }
        if any(name and name in lower for name in names):
            return str(entry.alpha_2)

    return None


def resolve_receipt_date(
    date_raw: str | None, location_raw: str | None, received_ts: str
) -> tuple[str | None, bool]:
    """Resolve a raw printed date to ISO format.

    Primary signal: vendor location -> country -> date format convention.
    Falls back to dual-parse (dayfirst=True/False) + received_ts-nearest
    tiebreak when location wasn't reliably extracted.

    Cross-checks the location-derived date against the received_ts-nearest
    parse — disagreement signals one extraction is probably wrong, so we
    return None (triggering the received_ts fallback in match.py) rather
    than silently trusting either.

    Returns (iso_date_or_None, was_ambiguous).
    """
    if not date_raw:
        return None, False

    anchor = datetime.fromisoformat(received_ts).date()

    # Get the received_ts-nearest interpretation (fallback path)
    dual_candidates: list[Any] = []
    for dayfirst in (False, True):
        try:
            parsed = date_parser.parse(date_raw, dayfirst=dayfirst, fuzzy=True).date()
            # Filter out obviously invalid dates (far future or far past)
            if abs((parsed - anchor).days) <= 365:
                dual_candidates.append(parsed)
        except (ValueError, OverflowError):
            pass
    dual_candidates = list(dict.fromkeys(dual_candidates))  # preserve order, remove dups
    dual_best = (
        min(dual_candidates, key=lambda d: abs((d - anchor).days)) if dual_candidates else None
    )

    # Try location-derived format as primary signal
    country = infer_country_code(location_raw) if location_raw else None
    convention = _DATE_FORMAT_BY_COUNTRY.get(country) if country else None

    located: Any = None
    if convention is not None:
        try:
            located = date_parser.parse(
                date_raw,
                dayfirst=(convention == "DMY"),
                yearfirst=(convention == "YMD"),
                fuzzy=True,
            ).date()
            # Filter out obviously invalid dates
            if abs((located - anchor).days) > 365:
                located = None
        except (ValueError, OverflowError):
            located = None

    # Cross-check: if both signals exist and disagree, treat as ambiguous
    if located is not None and dual_best is not None and located != dual_best:
        # Disagreement — fall back to None to trigger received_ts handling
        return None, True

    # Return the location-derived date if available, otherwise the fallback
    if located is not None:
        return located.isoformat(), False
    if dual_best is not None:
        return dual_best.isoformat(), False
    return None, False


# Rows a two-column receipt prints alongside an already-discounted price. A model
# that emits both the discounted price and these subtracts every discount twice.
_SAVINGS_ROW_RE = re.compile(
    r"member\s+savings|store\s+coupon|department\s+savings|personalized|basket\s+savings"
    r"|savings\s+total|total\s+savings",
    re.IGNORECASE,
)


def drop_double_counted_savings(
    items: list[dict[str, Any]], total_cents: int
) -> tuple[list[dict[str, Any]], bool]:
    """Remove savings rows that were already applied to the prices beside them.

    Returns (items, dropped_any).

    Receipts with a pre-discount "Price" and a post-discount "You Pay" column also
    print per-item savings rows. Recording the "You Pay" amount *and* those rows
    counts every discount twice. The prompt asks the model not to do this, and it
    obeys on some receipts and not others — so the arithmetic is checked here
    instead of trusted there.

    Deliberately conservative: the rows are dropped ONLY when doing so makes the
    items reconcile with the printed total. That self-validates. On a receipt whose
    discount rows are genuine (a single price column, as Costco prints), removing
    them moves the sum away from the total, so nothing is dropped.
    """
    if not isinstance(total_cents, int):
        return items, False
    readable = [i["amount_cents"] for i in items if isinstance(i["amount_cents"], int)]
    if len(readable) < len(items):
        return items, False  # an unreadable amount makes the arithmetic meaningless
    if abs(total_cents - sum(readable)) <= TOLERANCE_CENTS:
        return items, False  # already reconciles; nothing to repair

    kept = [
        i for i in items
        if not (i["amount_cents"] < 0 and _SAVINGS_ROW_RE.search(i["description"] or ""))
    ]
    if len(kept) == len(items):
        return items, False
    if abs(total_cents - sum(i["amount_cents"] for i in kept)) <= TOLERANCE_CENTS:
        return kept, True
    return items, False


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

    items, dropped_savings = drop_double_counted_savings(items, total_cents)
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

    # Why the score came out the way it did. "low" conflates two very different
    # failures — the model couldn't read the amounts, versus it read them fine and
    # they simply don't reconcile — and callers (the rotation retry in ocr.py) have
    # to tell those apart. Computed after the nudge so the reported diff matches the
    # items actually returned. Additive keys; consumers read this dict by .get().
    unreadable_count = sum(1 for i in items if not isinstance(i["amount_cents"], int))
    amount_diff_cents = total_cents - sum(
        i["amount_cents"] for i in items if isinstance(i["amount_cents"], int)
    )

    return {
        "merchant": merchant.strip(),
        "date": data.get("date"),  # kept for back-compat; prefer date_raw
        "date_raw": data.get("date_raw"),
        "location_raw": data.get("location_raw"),
        "total_cents": total_cents,
        "line_items": items,
        "confidence": confidence,
        "dropped_double_counted_savings": dropped_savings,
        "unreadable_count": unreadable_count,
        "amount_diff_cents": amount_diff_cents,
    }
