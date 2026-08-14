"""Parsing of the bank-sync payload Actual stores on each imported transaction.

Kept separate from duplicates.py because find_uncategorized(), the receipt
matcher, and the backtest script all need is_pending() without pulling in the
pipeline.

The payload lives in Transactions.raw_synced_data as a JSON string:

    {"booked": false, "cleared": false, "date": "...", "payeeName": "...",
     "transactionId": "TRN-...", "transactedDate": "...", "amount": "-230.00"}
"""

from __future__ import annotations

import json
import re
from typing import Any

# Tokens that drift between the pending and the posted descriptor for the same
# purchase — state/country suffixes, corporate forms, processor prefixes — and
# so carry no matching signal.
_STOPWORDS = frozenset({"us", "or", "inc", "llc", "the", "tst", "sq", "com"})

_PUNCT_RE = re.compile(r"[^a-z0-9]+")


def parse_synced(txn: Any) -> dict[str, Any] | None:
    """The row's bank-sync payload, or None when absent or unparseable."""
    raw = getattr(txn, "raw_synced_data", None)
    if not raw or not isinstance(raw, str):
        return None
    try:
        payload = json.loads(raw)
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


def is_pending(txn: Any) -> bool:
    """True only for rows imported as pending that have not cleared since.

    Neither half is sufficient alone. `cleared` is also 0 on starting balances,
    manually entered adjustments, and some split children. `booked` is a
    snapshot taken at first import and never refreshed, so rows that arrived
    pending still report `booked: false` long after the bank finalized them.
    The conjunction is what makes this precise enough to gate a deletion on.

    Rows with no payload at all — manual entries, starting balances, split
    children — are never pending and never deletion candidates.
    """
    # cleared is None on some rows; only an explicit 0 counts as uncleared.
    if txn.cleared != 0:
        return False
    payload = parse_synced(txn)
    if payload is None:
        return False
    return payload.get("booked") is False


def bank_txn_id(txn: Any) -> str | None:
    """The bank's own id for this row, from the column or the payload."""
    if txn.financial_id:
        return str(txn.financial_id)
    payload = parse_synced(txn)
    if payload is None:
        return None
    txn_id = payload.get("transactionId")
    return str(txn_id) if txn_id else None


def descriptor_text(txn: Any) -> str:
    """The best available merchant descriptor for the row.

    Same precedence the transfer pipeline uses (imported descriptor first, since
    the friendly payee is often shared across unrelated rows), with the synced
    payeeName as a last resort.
    """
    if txn.imported_description:
        return str(txn.imported_description)
    if txn.payee is not None and txn.payee.name:
        return str(txn.payee.name)
    payload = parse_synced(txn)
    if payload is not None and payload.get("payeeName"):
        return str(payload["payeeName"])
    return ""


def descriptor_tokens(txn: Any) -> frozenset[str]:
    """Normalized token set for descriptor comparison.

    Lowercase, strip punctuation, drop stopwords and bare numbers. The posted
    row typically appends location detail (MERCHANT -> MERCHANT CITY) or swaps a
    domain suffix (Merchant -> Merchant.com), so exact string equality is
    useless as a matching key but token overlap is not.
    """
    tokens = _PUNCT_RE.sub(" ", descriptor_text(txn).lower()).split()
    return frozenset(t for t in tokens if t not in _STOPWORDS and not t.isdigit())


def descriptor_similarity(a: frozenset[str], b: frozenset[str]) -> float:
    """Overlap of two token sets relative to the smaller one: |A n B| / min(|A|, |B|).

    Chosen over Jaccard because the posted descriptor is usually a superset of
    the pending one; on the sampled true pairs this scored 1.00 every time.
    """
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))
