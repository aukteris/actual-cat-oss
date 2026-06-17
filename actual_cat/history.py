"""Historical-categorization lookups built from the live Actual budget.

Both categorization paths benefit from knowing how the user has categorized
similar things before — and the budget DB is the best source because it reflects
the user's own manual corrections, not just the AI's prior guesses.

Two lookups, both built in a single pass over `get_transactions` (which returns
all non-parent rows — standalone transactions *and* child splits):

  - payee history  : payee_id -> Counter(category_path)   (standalone txns)
  - item history   : normalized description -> Counter(category_path)  (child splits)

Item history is keyed on the item *description*, not the merchant: a category is
a property of the item ("BANANAS" is groceries anywhere), so pooling across
merchants gives more data per item and avoids a cold start at a new store.
Cryptic merchant-specific descriptions are already self-namespaced by their
string, so no explicit merchant dimension is needed.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from actual.queries import get_category_groups, get_transactions

PayeeHistory = dict[str, "Counter[str]"]
ItemHistory = dict[str, "Counter[str]"]

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace — the history key."""
    lowered = _PUNCT_RE.sub(" ", text.lower())
    return _WS_RE.sub(" ", lowered).strip()


def build_category_path_map(session: Any) -> dict[str, str]:
    """category_id -> 'Group / Category', matching lookup_category_id's format.

    Mirrors schema.build_schema_text's traversal; skips tombstoned groups/cats.
    """
    path_map: dict[str, str] = {}
    for group in get_category_groups(session):
        if group.tombstone:
            continue
        for cat in group.categories:
            if cat.tombstone:
                continue
            path_map[cat.id] = f"{group.name} / {cat.name}"
    return path_map


def build_histories(session: Any, path_map: dict[str, str]) -> tuple[PayeeHistory, ItemHistory]:
    """Build payee and item history in one pass over the budget's transactions.

    get_transactions(session) returns all is_parent==0 rows: standalone
    transactions (payee history) and child splits (item history), partitioned
    here by is_child.
    """
    payee_history: PayeeHistory = {}
    item_history: ItemHistory = {}

    for txn in get_transactions(session):
        if txn.tombstone:
            continue
        category_path = path_map.get(txn.category_id)
        if category_path is None:
            continue  # uncategorized or category no longer in the live schema

        if txn.is_child:
            desc = normalize(txn.notes or "")
            if desc:
                item_history.setdefault(desc, Counter())[category_path] += 1
        elif txn.payee_id:
            payee_history.setdefault(txn.payee_id, Counter())[category_path] += 1

    return payee_history, item_history


def _top(counter: "Counter[str]", top_n: int, min_count: int) -> list[tuple[str, int]]:
    return [(path, n) for path, n in counter.most_common(top_n) if n >= min_count]


def render_payee_hint(
    payee_history: PayeeHistory, payee_id: str | None, *, top_n: int, min_count: int
) -> str:
    """A short reference block of a payee's past categories (empty if none)."""
    if not payee_id:
        return ""
    counter = payee_history.get(payee_id)
    if not counter:
        return ""
    entries = _top(counter, top_n, min_count)
    if not entries:
        return ""
    lines = "\n".join(f"- {path} ({n}×)" for path, n in entries)
    return (
        "\nPast categorizations for this payee "
        "(for reference — verify it still fits):\n"
        f"{lines}\n"
    )


def item_hints(
    item_history: ItemHistory, descriptions: list[str], *, top_n: int, min_count: int
) -> list[str]:
    """Per-item history lines aligned to `descriptions` ('' where no history)."""
    hints: list[str] = []
    for desc in descriptions:
        counter = item_history.get(normalize(desc))
        entries = _top(counter, top_n, min_count) if counter else []
        if entries:
            hints.append(", ".join(f"{path} ({n}×)" for path, n in entries))
        else:
            hints.append("")
    return hints
