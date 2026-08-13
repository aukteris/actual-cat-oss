"""Categorization pipeline — new-payee / uncategorized transaction handling."""

from typing import TYPE_CHECKING, Any

from actual.queries import get_category_groups, get_transactions

from . import history
from .sync_meta import is_pending
from .tags import append_tag, has_ai_marker, has_split_marker, slugify_category

if TYPE_CHECKING:
    from .audit import AuditLogger
    from .config import Config
    from .llm import LLMClient


def find_uncategorized(session: Any, defer_pending: bool = False) -> list[Any]:
    """Transactions needing categorization after the rule engine has run.

    Spike-confirmed field names:
      isParent (int)  — split parents skipped
      transferred_id  — already-paired transfers skipped
      category        — category ID (None when uncategorized)
      tombstone       — int 0/1

    With defer_pending, rows the bank hasn't finalized are skipped entirely. All
    three of the rule-engine loop, transfer detection, and categorization go
    through here, so this one filter keeps every pipeline off a row that may yet
    turn out to be a duplicate of its own posted counterpart — a pending row
    paired as a transfer leaves its partner pointing at a tombstone once the
    duplicate is deleted. The cost is that pending charges stay uncategorized
    until they clear.
    """
    return [
        t for t in get_transactions(session)
        if t.category_id is None
        and not t.is_parent
        and t.transferred_id is None
        and not t.tombstone
        and not has_ai_marker(t.notes)
        and not has_split_marker(t.notes)
        and t.account is not None
        and not t.account.offbudget
        and not (defer_pending and is_pending(t))
    ]


def render_user_prompt(txn: Any, schema_text: str, history_hint: str = "") -> str:
    payee_name = txn.payee.name if txn.payee else None
    # Strip any existing AI tags from notes before sending to LLM — they're
    # our own markers, not meaningful merchant context.
    import re
    raw_notes = re.sub(r"\s*#ai-\S+", "", txn.notes or "").strip() or "(none)"
    return f"""Current category schema:
{schema_text}

Transaction to categorize:
- Payee (institution's friendly name): {payee_name or '(none)'}
- Raw descriptor (bank memo, may include address): {raw_notes}
- Amount: {txn.amount} cents ({txn.amount / 100:+.2f} USD)
- Date: {txn.get_date()}
- Account: {txn.account.name} ({'off-budget' if txn.account.offbudget else 'on-budget'})
{history_hint}
Categorize this transaction. The raw descriptor often contains the merchant's
legal name or payment processor prefix (SQ *, PAR*, APPLE.COM/BILL, etc.) —
use it together with the friendly payee name to identify the merchant.
If you cannot confidently determine the category, return
"category": "Uncertain" with confidence "low".
"""


def lookup_category_id(session: Any, category_path: str) -> str | None:
    """Resolve 'Group / Category' string to a category UUID.

    Returns None if the path doesn't match the live schema (guards against
    LLM hallucinating category names).
    """
    if " / " not in category_path:
        return None
    group_name, cat_name = category_path.split(" / ", 1)

    for group in get_category_groups(session):
        if group.name == group_name and not group.tombstone:
            for cat in group.categories:
                if cat.name == cat_name and not cat.tombstone:
                    return cat.id  # type: ignore[no-any-return]
    return None


def process_categorization(
    actual: Any,
    llm: "LLMClient",
    audit: "AuditLogger",
    cfg: "Config",
    schema_text: str,
    prompts: Any,
) -> None:
    txns = find_uncategorized(actual.session, cfg.duplicates_defer_pending)

    if cfg.history_enabled:
        path_map = history.build_category_path_map(actual.session)
        payee_history, _ = history.build_histories(actual.session, path_map)
    else:
        payee_history = {}

    for txn in txns:
        history_hint = history.render_payee_hint(
            payee_history, txn.payee_id,
            top_n=cfg.history_payee_top_n, min_count=cfg.history_min_count,
        ) if cfg.history_enabled else ""
        user_msg = render_user_prompt(txn, schema_text, history_hint)
        response = llm.complete_json(prompts.CATEGORIZATION_SYSTEM, user_msg)

        if "error" in response:
            audit.log_failure(txn, response["error"], pipeline="categorization")
            continue

        category = response.get("category", "")
        confidence = response.get("confidence", "low")
        tags = response.get("tags", [])

        if category == "Uncertain":
            txn.notes = append_tag(txn.notes, f"#ai:{slugify_category(category)}")
            audit.log(txn, response, mode="uncertain", action="tagged")
            continue

        cat_id = lookup_category_id(actual.session, category)
        if cat_id is None:
            audit.log_failure(
                txn, f"Invalid category from LLM: {category!r}", pipeline="categorization"
            )
            continue

        should_apply = (
            cfg.categorization_mode == "apply"
            and confidence == cfg.categorization_threshold
        )

        # Free-form tags applied first so the AI marker ends up at the front
        for tag in tags:
            if isinstance(tag, str) and tag.startswith("#") and not has_ai_marker(tag):
                txn.notes = append_tag(txn.notes, tag)

        if should_apply:
            txn.category_id = cat_id
            txn.notes = append_tag(txn.notes, "#ai-assisted")
            action = "applied"
        else:
            txn.notes = append_tag(txn.notes, f"#ai:{slugify_category(category)}")
            action = "suggested"

        audit.log(txn, response, mode=cfg.categorization_mode, action=action)
