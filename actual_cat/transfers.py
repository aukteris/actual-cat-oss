"""Transfer detection pipeline — inverse-amount pair evaluation."""

import re
from datetime import timedelta
from string import ascii_uppercase
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


def _describe_transaction(txn: Any) -> str:
    """The bullet block both transfer prompts use to show one transaction."""
    payee = txn.imported_description or (txn.payee.name if txn.payee else "(none)")
    acct = txn.account
    notes = re.sub(r"\s*#ai-\S+", "", txn.notes or "").strip() or "(none)"
    return f"""- Account: {acct.name} ({'off-budget' if acct.offbudget else 'on-budget'})
- Date: {txn.get_date()}
- Payee (friendly name): {payee}
- Raw descriptor: {notes}
- Amount: {txn.amount / 100:+.2f} USD"""


def render_transfer_prompt(txn_a: Any, txn_b: Any) -> str:
    return f"""Two uncategorized transactions with matching inverse amounts within
a few days. Evaluate whether they represent a transfer between accounts.

Transaction A:
{_describe_transaction(txn_a)}

Transaction B:
{_describe_transaction(txn_b)}

Is this a transfer between accounts? Consider account semantics (a card
payment from checking to a credit card tracked in this budget is a transfer,
as is checking -> savings; paying an off-budget loan or an external merchant
is not), payee/memo content, and date alignment.

Return JSON.
"""


def render_transfer_choice_prompt(txn: Any, candidates: list[Any]) -> str:
    """One driving transaction against every candidate at once, lettered A, B, ...

    Judging each pair in isolation is what chained two same-day transfers across
    banks (ISSUE-028): any two of the user's own accounts with "transfer" in the
    descriptor look like a transfer, so whichever candidate came first won.
    """
    blocks = "\n\n".join(
        f"Candidate {letter}:\n{_describe_transaction(candidate)}"
        for letter, candidate in zip(ascii_uppercase, candidates)
    )
    return f"""One uncategorized transaction has {len(candidates)} transactions in other
accounts with the matching inverse amount within a few days. Money moved once,
so at most one of them is the other leg of a transfer.

Transaction:
{_describe_transaction(txn)}

{blocks}

Which candidate, if any, is the other leg of this transfer? Return JSON.
"""


def pair_as_transfer(txn_a: Any, txn_b: Any) -> None:
    """Link two existing transactions as a proper Actual transfer pair.

    A real transfer needs three things on each leg (mirrors actualpy's
    create_transfer, but for existing rows): the cross-referenced
    transferred_id, the *other* account's transfer payee (account.payee), and
    no spending category. Setting transferred_id alone produces a malformed pair
    that Actual doesn't recognize as a transfer.

    Refuses to overwrite a link to a third row. Doing so silently is how one run
    chained two same-day transfers into A→B→C→D instead of two pairs, which a
    later UI save in Actual turned into a duplicated row (ISSUE-028).
    """
    for leg, other in ((txn_a, txn_b), (txn_b, txn_a)):
        if leg.transferred_id is not None and leg.transferred_id != other.id:
            raise ValueError(
                f"{leg.id} is already linked to {leg.transferred_id}; "
                f"refusing to re-pair it with {other.id}"
            )

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

    # A one-way link names the wrong partner for at least one leg, so
    # "repairing" against it would assert the wrong transfer payee — and every
    # run would re-assert it, as happened for three days before ISSUE-028 was
    # found. Nothing safe can be inferred; report it like an orphan.
    if partner.transferred_id != txn.id:
        return "non_reciprocal", {}

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

        if label == "non_reciprocal":
            audit.log(
                txn, {}, mode="repair", action="non-reciprocal", pipeline="transfer",
                extra={
                    "partner_id": txn.transferred_id,
                    "partner_transferred_id": txn.transfer.transferred_id,
                },
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


def _choice_index(choice: Any, count: int) -> int | None:
    """The candidate a lettered choice names, or None if it names none of them."""
    letter = str(choice).strip().upper()
    if len(letter) == 1 and letter in ascii_uppercase[:count]:
        return ascii_uppercase.index(letter)
    return None


def _conclude_transfer(
    txn: Any,
    partner: Any,
    response: dict[str, Any],
    audit: "AuditLogger",
    cfg: "Config",
    handled: set[str],
    extra: dict[str, Any] | None = None,
) -> None:
    """Pair or tag txn and partner once the model has said they are a transfer."""
    confidence = response.get("confidence", "low")
    should_apply = (
        cfg.transfer_mode == "apply"
        and confidence == cfg.transfer_threshold
    )

    if should_apply:
        try:
            pair_as_transfer(txn, partner)
        except ValueError as e:
            audit.log_failure(txn, str(e), pipeline="transfer")
            return

    # A leg categorized on an earlier run (before its partner posted)
    # carries a stale #ai:<category> tag; drop it now that we've
    # concluded this is a transfer.
    txn.notes = strip_category_tag(txn.notes)
    partner.notes = strip_category_tag(partner.notes)

    if should_apply:
        txn.notes = append_tag(txn.notes, "#ai-assisted")
        partner.notes = append_tag(partner.notes, "#ai-assisted")
        action = "paired"
    else:
        txn.notes = append_tag(txn.notes, "#ai-suggested-transfer")
        partner.notes = append_tag(partner.notes, "#ai-suggested-transfer")
        action = "tagged"

    handled.update((txn.id, partner.id))
    audit.log(
        txn, response, mode=cfg.transfer_mode, action=action,
        pipeline="transfer",
        extra={"partner_id": partner.id, **partner_state(partner), **(extra or {})},
    )


def _evaluate_pair(
    txn: Any,
    partner: Any,
    llm: "LLMClient",
    audit: "AuditLogger",
    cfg: "Config",
    prompts: Any,
    handled: set[str],
) -> None:
    response = llm.complete_json(
        prompts.TRANSFER_SYSTEM, render_transfer_prompt(txn, partner)
    )

    if "error" in response:
        audit.log_failure(txn, response["error"], pipeline="transfer")
        return

    if not response.get("is_transfer", False):
        audit.log(
            txn, response, mode="transfer-rejected", action="none",
            pipeline="transfer",
            extra={"partner_id": partner.id, **partner_state(partner)},
        )
        return

    _conclude_transfer(txn, partner, response, audit, cfg, handled)


def _evaluate_choice(
    txn: Any,
    candidates: list[Any],
    llm: "LLMClient",
    audit: "AuditLogger",
    cfg: "Config",
    prompts: Any,
    handled: set[str],
) -> None:
    """Ask once which of several candidates is txn's partner, and hold on ambiguity.

    "ambiguous" — or a letter naming no candidate — tags every row involved
    #ai-suggested-transfer rather than ranking a guess, as
    duplicates.propose_candidates does. The tag also keeps categorization from
    booking these rows as spending while they wait for review.
    """
    candidate_ids = [c.id for c in candidates]

    if len(candidates) > len(ascii_uppercase):
        audit.log_failure(
            txn, f"{len(candidates)} transfer candidates, too many to compare",
            pipeline="transfer",
        )
        return

    response = llm.complete_json(
        prompts.TRANSFER_CHOICE_SYSTEM, render_transfer_choice_prompt(txn, candidates)
    )

    if "error" in response:
        audit.log_failure(txn, response["error"], pipeline="transfer")
        return

    # str() is deliberately not applied: a missing choice would become "None",
    # which lowercases to a rejection instead of a hold.
    raw_choice = response.get("choice")
    choice = raw_choice.strip().lower() if isinstance(raw_choice, str) else ""

    if choice == "none":
        audit.log(
            txn, response, mode="transfer-rejected", action="none",
            pipeline="transfer", extra={"candidate_ids": candidate_ids},
        )
        return

    index = _choice_index(choice, len(candidates))
    if index is None:
        for row in (txn, *candidates):
            row.notes = append_tag(strip_category_tag(row.notes), "#ai-suggested-transfer")
            handled.add(row.id)
        audit.log(
            txn, response, mode="transfer-ambiguous", action="tagged",
            pipeline="transfer", extra={"candidate_ids": candidate_ids},
        )
        return

    _conclude_transfer(
        txn, candidates[index], response, audit, cfg, handled,
        extra={"candidate_ids": candidate_ids},
    )


def process_transfers(
    actual: Any,
    llm: "LLMClient",
    audit: "AuditLogger",
    cfg: "Config",
    prompts: Any,
) -> None:
    from .categorization import find_uncategorized

    candidates_seen: set[tuple[str, str]] = set()
    # Rows paired or tagged earlier in this run. The driving list is computed
    # once up front and never refreshed, so without this a row linked as some
    # earlier row's partner would still be driven later and re-paired.
    handled: set[str] = set()
    txns = find_uncategorized(actual.session, cfg.duplicates_defer_pending)

    for txn in txns:
        if txn.id in handled or txn.transferred_id is not None:
            continue

        candidates = [
            c for c in find_transfer_candidates(
                actual.session, txn, cfg.transfer_window_days, cfg.duplicates_defer_pending
            )
            if c.id not in handled and c.transferred_id is None
        ]

        if len(candidates) > 1:
            # Several candidates are judged together, even if some pairs were
            # already seen from the other side: seeing them one at a time is
            # exactly what lets the first plausible one win.
            for c in candidates:
                candidates_seen.add(tuple(sorted([txn.id, c.id])))
            _evaluate_choice(txn, candidates, llm, audit, cfg, prompts, handled)
            continue

        for partner in candidates:
            pair_id = tuple(sorted([txn.id, partner.id]))
            if pair_id in candidates_seen:
                continue
            candidates_seen.add(pair_id)
            _evaluate_pair(txn, partner, llm, audit, cfg, prompts, handled)
