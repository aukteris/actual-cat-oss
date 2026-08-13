"""Transfer detection pipeline — inverse-amount pair evaluation."""

from datetime import timedelta
from typing import TYPE_CHECKING, Any

from .sync_meta import is_pending
from .tags import append_tag, strip_category_tag

if TYPE_CHECKING:
    from .audit import AuditLogger
    from .config import Config
    from .llm import LLMClient


def find_transfer_candidates(
    session: Any, txn: Any, window_days: int, defer_pending: bool = False
) -> list[Any]:
    """Find inverse-amount transactions in other accounts within the date window.

    Spike-confirmed: transfer linkage field is transferred_id; account FK is acct.

    defer_pending excludes rows the bank hasn't finalized from the partner side
    too, not just the driving side: pairing a pending leg leaves its partner
    pointing at a tombstone if that leg later turns out to be a duplicate.
    """
    from actual.database import Transactions
    from actual.utils.conversions import date_to_int

    if txn.amount == 0:
        return []

    txn_date = txn.get_date()
    lo = date_to_int(txn_date - timedelta(days=window_days))
    hi = date_to_int(txn_date + timedelta(days=window_days))

    candidates = (
        session.query(Transactions)
        .filter(
            Transactions.amount == -txn.amount,
            Transactions.acct != txn.acct,
            Transactions.tombstone == 0,
            Transactions.transferred_id.is_(None),
            Transactions.date >= lo,
            Transactions.date <= hi,
            Transactions.id != txn.id,
        )
        .all()
    )

    if defer_pending:
        return [t for t in candidates if not is_pending(t)]
    return list(candidates)


def render_transfer_prompt(txn_a: Any, txn_b: Any) -> str:
    payee_a = txn_a.imported_description or (txn_a.payee.name if txn_a.payee else "(none)")
    payee_b = txn_b.imported_description or (txn_b.payee.name if txn_b.payee else "(none)")
    acct_a = txn_a.account
    acct_b = txn_b.account

    import re
    def clean_notes(t: Any) -> str:
        return re.sub(r"\s*#ai-\S+", "", t.notes or "").strip() or "(none)"

    return f"""Two uncategorized transactions with matching inverse amounts within
a few days. Evaluate whether they represent a transfer between accounts.

Transaction A:
- Account: {acct_a.name} ({'off-budget' if acct_a.offbudget else 'on-budget'})
- Date: {txn_a.get_date()}
- Payee (friendly name): {payee_a}
- Raw descriptor: {clean_notes(txn_a)}
- Amount: {txn_a.amount / 100:+.2f} USD

Transaction B:
- Account: {acct_b.name} ({'off-budget' if acct_b.offbudget else 'on-budget'})
- Date: {txn_b.get_date()}
- Payee (friendly name): {payee_b}
- Raw descriptor: {clean_notes(txn_b)}
- Amount: {txn_b.amount / 100:+.2f} USD

Is this a transfer between accounts? Consider account semantics
(checking -> credit card is usually a card payment, checking -> savings
is a transfer), payee/memo content, and date alignment.

Return JSON.
"""


def pair_as_transfer(txn_a: Any, txn_b: Any) -> None:
    """Link two existing transactions as a proper Actual transfer pair.

    A real transfer needs three things on each leg (mirrors actualpy's
    create_transfer, but for existing rows): the cross-referenced
    transferred_id, the *other* account's transfer payee (account.payee), and
    no spending category. Setting transferred_id alone produces a malformed pair
    that Actual doesn't recognize as a transfer.
    """
    txn_a.payee_id = txn_b.account.payee.id
    txn_b.payee_id = txn_a.account.payee.id
    txn_a.transferred_id = txn_b.id
    txn_b.transferred_id = txn_a.id
    txn_a.category_id = None
    txn_b.category_id = None


def process_transfers(
    actual: Any,
    llm: "LLMClient",
    audit: "AuditLogger",
    cfg: "Config",
    prompts: Any,
) -> None:
    from .categorization import find_uncategorized

    candidates_seen: set[tuple[str, str]] = set()
    txns = find_uncategorized(actual.session, cfg.duplicates_defer_pending)

    for txn in txns:
        candidates = find_transfer_candidates(
            actual.session, txn, cfg.transfer_window_days, cfg.duplicates_defer_pending
        )

        for partner in candidates:
            pair_id = tuple(sorted([txn.id, partner.id]))
            if pair_id in candidates_seen:
                continue
            candidates_seen.add(pair_id)

            response = llm.complete_json(
                prompts.TRANSFER_SYSTEM, render_transfer_prompt(txn, partner)
            )

            if "error" in response:
                audit.log_failure(txn, response["error"], pipeline="transfer")
                continue

            is_transfer = response.get("is_transfer", False)
            confidence = response.get("confidence", "low")

            if not is_transfer:
                audit.log(
                    txn, response, mode="transfer-rejected", action="none",
                    pipeline="transfer", extra={"partner_id": partner.id},
                )
                continue

            should_apply = (
                cfg.transfer_mode == "apply"
                and confidence == cfg.transfer_threshold
            )

            # A leg categorized on an earlier run (before its partner posted)
            # carries a stale #ai:<category> tag; drop it now that we've
            # concluded this is a transfer.
            txn.notes = strip_category_tag(txn.notes)
            partner.notes = strip_category_tag(partner.notes)

            if should_apply:
                pair_as_transfer(txn, partner)
                txn.notes = append_tag(txn.notes, "#ai-assisted")
                partner.notes = append_tag(partner.notes, "#ai-assisted")
                action = "paired"
            else:
                txn.notes = append_tag(txn.notes, "#ai-suggested-transfer")
                partner.notes = append_tag(partner.notes, "#ai-suggested-transfer")
                action = "tagged"

            audit.log(
                txn, response, mode=cfg.transfer_mode, action=action,
                pipeline="transfer", extra={"partner_id": partner.id},
            )
