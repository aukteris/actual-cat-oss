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


def expected_transfer_payee(partner: Any) -> str | None:
    """The payee a transfer leg must carry: the *other* account's transfer payee."""
    return partner.account.payee.id if partner.account and partner.account.payee else None


def compute_transfer_repair(txn: Any, partner: Any) -> dict[str, Any]:
    """Fields that must be restored on `txn` to satisfy the transfer invariant
    with `partner`. Empty if `txn` is already healthy.

    payee_id and category_id are corrected unconditionally — any transfer leg,
    AI-paired or user-made, must carry the partner account's transfer payee and
    no category. #ai-assisted is restored only when `partner` still carries it:
    plenty of live pairs are user-made and were never tagged, and relabeling
    those as agent output would misattribute them.
    """
    fields: dict[str, Any] = {}

    expected_payee = expected_transfer_payee(partner)
    if expected_payee is not None and txn.payee_id != expected_payee:
        fields["payee_id"] = expected_payee

    if txn.category_id is not None:
        fields["category_id"] = None

    if "#ai-assisted" in (partner.notes or "") and "#ai-assisted" not in (txn.notes or ""):
        fields["notes"] = append_tag(txn.notes, "#ai-assisted")

    return fields


def classify_transfer_leg(txn: Any, partner: Any | None) -> tuple[str, dict[str, Any]]:
    """Label `txn` relative to its transfer partner, alongside the repair (if any).

    partner is None for both a missing and a tombstoned partner — Transactions.transfer
    is already conditioned on the remote row's tombstone, so that distinction collapses
    to "orphaned" here, which is correct: either way there's nothing safe to repair.
    """
    if partner is None:
        return "orphaned", {}

    fields = compute_transfer_repair(txn, partner)
    if not fields:
        return "healthy", {}
    if "payee_id" in fields:
        return "payee_reset", fields
    if "category_id" in fields:
        return "categorized", fields
    return "marker_lost", fields


def apply_transfer_repair(txn: Any, fields: dict[str, Any]) -> None:
    for key, value in fields.items():
        setattr(txn, key, value)


def repair_transfer_pairs(actual: Any, audit: "AuditLogger") -> int:
    """Restore what bank-sync reconciliation strips off an already-paired leg.

    reconcile_transaction(update_existing=True) overwrites notes and payee_id on
    any row the bank re-reports. On a transfer leg that silently removes the
    transfer payee and the #ai-assisted marker while leaving transferred_id set,
    producing a pair Actual no longer renders as a transfer and that no pipeline
    can reach again — every row selector filters on transferred_id IS NULL.

    Runs over all paired rows rather than only the rows run_bank_sync returned,
    so it also heals damage from Actual's own server-side sync, which this
    process never sees.
    """
    from actual.database import Transactions

    rows = (
        actual.session.query(Transactions)
        .filter(Transactions.transferred_id.isnot(None), Transactions.tombstone == 0)
        .all()
    )

    repaired = 0
    for txn in rows:
        label, fields = classify_transfer_leg(txn, txn.transfer)

        if label == "orphaned":
            audit.log(
                txn, {}, mode="repair", action="orphaned", pipeline="transfer",
                extra={"partner_id": txn.transferred_id},
            )
            continue

        if not fields:
            continue

        apply_transfer_repair(txn, fields)
        repaired += 1
        audit.log(
            txn, {}, mode="repair", action="repaired", pipeline="transfer",
            extra={"partner_id": txn.transfer.id, "fields": sorted(fields)},
        )

    return repaired


def find_relink_partner(
    txn: Any,
    live_by_id: dict[str, Any],
    live_by_financial_id: dict[str, list[Any]],
    tombstoned_by_financial_id: dict[str, list[Any]],
) -> tuple[Any | None, str | None]:
    """The leg `txn` was paired with before Actual re-created it, if that can be
    established without guessing.

    A sync triggered from the Actual UI deletes and re-creates a transaction
    rather than reconciling it in place: the replacement row carries a new id but
    the *same* financial_id, and the tombstoned predecessor keeps the
    transferred_id it was paired with. That predecessor is a record of a
    conclusion already reached, so the link can be restored deterministically
    instead of spending another LLM call to re-derive it — and without depending
    on the model returning the same verdict a second time.

    Returns (partner, None) only when exactly one partner is implied and every
    safety condition holds, (None, reason) on a refusal worth recording, and
    (None, None) when there is simply nothing to relink. Ambiguity is always a
    refusal rather than a ranked guess, as in duplicates.propose_candidates.
    """
    if txn.is_parent or txn.is_child:
        return None, None
    if txn.reconciled:
        return None, None

    predecessors = [
        row for row in tombstoned_by_financial_id.get(txn.financial_id, [])
        if row.transferred_id is not None and row.acct == txn.acct
    ]
    if not predecessors:
        return None, None

    # More than one live row on the same financial_id means we cannot tell which
    # of them the predecessor was replaced by.
    if len(live_by_financial_id.get(txn.financial_id, [])) > 1:
        return None, "more than one live row shares this financial id"

    implied = {row.transferred_id for row in predecessors}
    if len(implied) > 1:
        return None, "tombstoned predecessors disagree on the prior partner"

    partner = live_by_id.get(implied.pop())
    if partner is None:
        return None, "prior partner is no longer live"
    if partner.transferred_id is not None and partner.transferred_id != txn.id:
        return None, "prior partner is already paired with another row"
    if partner.acct == txn.acct:
        return None, "prior partner is in the same account"
    if partner.amount != -txn.amount:
        return None, "amounts are no longer inverse"
    if partner.is_parent or partner.is_child or partner.reconciled:
        return None, "prior partner is a split or has been reconciled"

    return partner, None


def relink_recreated_transfers(actual: Any, audit: "AuditLogger") -> int:
    """Restore pairings that a UI-triggered sync broke by re-creating a leg.

    repair_transfer_pairs() cannot see this case: the replacement row and its
    former partner both have transferred_id NULL, so neither is a paired row any
    more. Left alone the pair is eventually rebuilt by process_transfers at the
    cost of a fresh LLM call, and only if the model again returns exactly the
    configured confidence — a "medium" verdict would instead tag the row
    #ai-suggested-transfer, which excludes it from find_uncategorized for good.
    Relinking deterministically removes both the cost and that failure mode.
    """
    from actual.database import Transactions

    rows = actual.session.query(Transactions).all()

    live_by_id: dict[str, Any] = {}
    live_by_financial_id: dict[str, list[Any]] = {}
    tombstoned_by_financial_id: dict[str, list[Any]] = {}
    for row in rows:
        if row.tombstone:
            if row.financial_id:
                tombstoned_by_financial_id.setdefault(row.financial_id, []).append(row)
            continue
        live_by_id[row.id] = row
        if row.financial_id:
            live_by_financial_id.setdefault(row.financial_id, []).append(row)

    relinked = 0
    for txn in live_by_id.values():
        if txn.transferred_id is not None or not txn.financial_id:
            continue

        partner, reason = find_relink_partner(
            txn, live_by_id, live_by_financial_id, tombstoned_by_financial_id
        )
        if partner is None:
            if reason:
                audit.log(
                    txn, {}, mode="repair", action="relink-refused",
                    pipeline="transfer", extra={"reason": reason},
                )
            continue

        pair_as_transfer(txn, partner)
        # Reuse the shared rules for the marker so relink and repair cannot
        # disagree about when #ai-assisted may be added.
        apply_transfer_repair(txn, compute_transfer_repair(txn, partner))
        apply_transfer_repair(partner, compute_transfer_repair(partner, txn))
        relinked += 1
        audit.log(
            txn, {}, mode="repair", action="relinked", pipeline="transfer",
            extra={"partner_id": partner.id, "via_financial_id": txn.financial_id},
        )

    return relinked


def partner_state(partner: Any) -> dict[str, Any]:
    """Partner-leg fields worth recording alongside a transfer decision.

    audit.log() otherwise records only the driving transaction; the partner
    appears solely as partner_id and can never be reconstructed from the log
    afterward. That blind spot is what let a real defect (payee/marker
    stripped off a paired leg by bank-sync reconciliation) be mis-diagnosed as
    a missing row, purely from absence of a log line that could never exist.
    """
    return {
        "partner_payee_id": partner.payee_id,
        "partner_category_id": partner.category_id,
        "partner_notes": partner.notes,
    }


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
                    pipeline="transfer",
                    extra={"partner_id": partner.id, **partner_state(partner)},
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
                pipeline="transfer",
                extra={"partner_id": partner.id, **partner_state(partner)},
            )
