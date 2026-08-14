"""Pending-duplicate resolution — removes the pending authorization row left
behind when its posted counterpart imports as a separate transaction.

Runs first, before every other pipeline: deletion is the most destructive
operation in the project, and going first means no LLM call, tag, transfer
pairing, or split has been spent on a row that is about to be removed.

Stage 1 (clustering + rule matching) is deliberately pure over row lists rather
than session queries, so scripts/backtest_duplicates.py can replay it over the
budget's tombstoned history without touching the live pipeline.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from datetime import timedelta
from itertools import combinations
from typing import TYPE_CHECKING, Any

from .sync_meta import (
    bank_txn_id,
    descriptor_similarity,
    descriptor_text,
    descriptor_tokens,
    is_pending,
)
from .tags import append_tag, has_duplicate_marker

if TYPE_CHECKING:
    from .audit import AuditLogger
    from .config import Config
    from .llm import LLMClient

# Descriptor overlap required to consider two rows the same merchant. Not
# exposed in config: it is a property of the normalization in sync_meta, not a
# knob worth turning per install.
SIMILARITY_THRESHOLD = 0.5

# Ceiling on subset-sum enumeration. Real clusters are 2-3 rows; this only
# bounds the pathological case of many same-merchant pendings in one window.
MAX_SUBSET_SIZE = 4

SUGGEST_TAG = "#ai-suggested-duplicate"
REVIEW_TAG = "#ai-duplicate-review"

_CONFIDENCE_RANK = {"high": 2, "medium": 1, "low": 0}


def _meets_threshold(confidence: str, threshold: str) -> bool:
    """Rank comparison, as in receipts/match.py — a "high" verdict satisfies a
    "medium" threshold. (The transfer pipeline's exact-equality check would
    reject it; for a pipeline that deletes, the intent of the setting matters.)
    """
    return _CONFIDENCE_RANK.get(confidence, 0) >= _CONFIDENCE_RANK.get(threshold, 2)


@dataclass(frozen=True)
class Candidate:
    """A proposed resolution: pending rows superseded by one booked row.

    rule is "R1".."R5", possibly compounded ("R3+R4" when an authorization probe
    joins a tip-uplift match), or "ambiguous" for a held candidate set.
    booked is None for R5 (reversal pair, no survivor) and for holds.
    """

    rule: str
    pending: tuple[Any, ...]
    booked: Any | None = None
    delta_pct: float | None = None
    alternatives: int = 0


# ---------------------------------------------------------------------------
# Stage 1a — row selection (the only session-touching functions)
# ---------------------------------------------------------------------------


def find_pending(session: Any) -> list[Any]:
    """Live pending rows on on-budget accounts, oldest first."""
    from actual.database import Transactions

    rows = session.query(Transactions).filter(Transactions.tombstone == 0).all()
    selected = [
        t for t in rows
        if not t.is_child
        and t.account is not None
        and not t.account.offbudget
        and is_pending(t)
    ]
    return sorted(selected, key=lambda t: (t.date, t.id))


def find_booked_candidates(session: Any, cluster: list[Any], window_days: int) -> list[Any]:
    """Booked rows in the cluster's account that could be the posted counterpart.

    The window is symmetric: a posted row is sometimes dated a day *earlier*
    than the pending one, because it carries the transaction date where the
    pending row carried the authorization date.
    """
    from actual.database import Transactions
    from actual.utils.conversions import date_to_int

    if not cluster:
        return []

    dates = [t.get_date() for t in cluster]
    lo = date_to_int(min(dates) - timedelta(days=window_days))
    hi = date_to_int(max(dates) + timedelta(days=window_days))
    pending_ids = {t.id for t in cluster}

    rows = (
        session.query(Transactions)
        .filter(
            Transactions.acct == cluster[0].acct,
            Transactions.tombstone == 0,
            Transactions.date >= lo,
            Transactions.date <= hi,
        )
        .all()
    )

    return sorted(
        (
            t for t in rows
            if t.id not in pending_ids
            and not t.is_child
            # Re-checked in Python so the window holds regardless of what the
            # query returned.
            and lo <= t.date <= hi
            and not is_pending(t)
            and matches_cluster(cluster, t)
        ),
        key=lambda t: (t.date, t.id),
    )


def matches_cluster(cluster: list[Any], booked: Any) -> bool:
    """True when the booked row shares a descriptor with any cluster member."""
    booked_tokens = descriptor_tokens(booked)
    return any(
        descriptor_similarity(descriptor_tokens(row), booked_tokens) >= SIMILARITY_THRESHOLD
        for row in cluster
    )


# ---------------------------------------------------------------------------
# Stage 1b — clustering and rule matching (pure)
# ---------------------------------------------------------------------------


def cluster_pending(rows: list[Any]) -> list[list[Any]]:
    """Group pending rows by account, then single-link on descriptor similarity.

    Clusters matter because the duplicate is not always one row: a grocery
    pickup authorizes and then adjusts, and an authorization probe rides along
    with the real charge. Only the cluster as a whole sums to the posted row.
    """
    by_account: dict[str, list[Any]] = {}
    for row in rows:
        by_account.setdefault(row.acct, []).append(row)

    clusters: list[list[Any]] = []
    for acct in sorted(by_account):
        remaining = sorted(by_account[acct], key=lambda t: (t.date, t.id))
        while remaining:
            cluster = [remaining.pop(0)]
            grew = True
            while grew:
                grew = False
                for row in list(remaining):
                    row_tokens = descriptor_tokens(row)
                    if any(
                        descriptor_similarity(row_tokens, descriptor_tokens(member))
                        >= SIMILARITY_THRESHOLD
                        for member in cluster
                    ):
                        cluster.append(row)
                        remaining.remove(row)
                        grew = True
            clusters.append(cluster)
    return clusters


def _delta_pct(pending_cents: int, booked_cents: int) -> float | None:
    """How far the booked amount sits above (+) or below (-) the pending one."""
    if not pending_cents:
        return None
    return (abs(booked_cents) / abs(pending_cents) - 1) * 100


def _same_sign(a: int, b: int) -> bool:
    return (a < 0) == (b < 0)


def propose_candidates(cluster: list[Any], booked: list[Any], cfg: "Config") -> list[Candidate]:
    """Apply R1-R5 to one cluster. Each pending row lands in at most one candidate.

    Rules run strongest-first so a certain match claims its rows before a
    heuristic one can. Where a rule finds more than one booked counterpart the
    result is a hold, never a ranked guess.
    """
    candidates: list[Candidate] = []
    # Rows are tracked by id, never by object identity: SQLModel rows inherit a
    # field-wise __eq__, so two same-amount rows would compare equal.
    by_id: dict[str, Any] = {row.id: row for row in cluster}
    unmatched: list[str] = [row.id for row in cluster]
    claimed: dict[str, int] = {}  # booked id -> index into candidates

    def claim(rule: str, ids: list[str], booked_row: Any) -> None:
        rows = [by_id[i] for i in ids]
        candidates.append(
            Candidate(rule, tuple(rows), booked_row, _delta_pct(rows[0].amount, booked_row.amount))
        )
        claimed[booked_row.id] = len(candidates) - 1
        for i in ids:
            unmatched.remove(i)

    def hold(ids: list[str], alternatives: int) -> None:
        candidates.append(
            Candidate("ambiguous", tuple(by_id[i] for i in ids), None, None, alternatives)
        )
        for i in ids:
            unmatched.remove(i)

    def available() -> list[Any]:
        return [b for b in booked if b.id not in claimed]

    # R1 — same bank id on both rows. Certain when it happens.
    for row_key in list(unmatched):
        row_id = bank_txn_id(by_id[row_key])
        if row_id is None:
            continue
        hits = [b for b in available() if bank_txn_id(b) == row_id]
        if not hits:
            continue
        if len(hits) > 1:
            hold([row_key], len(hits))
        else:
            claim("R1", [row_key], hits[0])

    # R2 — a subset of the cluster sums exactly to a booked amount. Sign-agnostic,
    # so a refund or adjustment leg is pulled in alongside the charge. Larger
    # subsets first: an exact multi-row sum is a stronger signal than a single
    # row that happens to match, and it keeps the adjustment leg from being
    # orphaned. The single-element case covers plain exact duplicates.
    for size in range(min(len(unmatched), MAX_SUBSET_SIZE), 0, -1):
        for combo in combinations(list(unmatched), size):
            if any(key not in unmatched for key in combo):
                continue  # claimed by an earlier combo this pass
            total = sum(by_id[key].amount for key in combo)
            hits = [b for b in available() if b.amount == total]
            if not hits:
                continue
            if len(hits) > 1:
                hold(list(combo), len(hits))
            else:
                claim("R2", list(combo), hits[0])

    # R3 — tip band: the booked counterpart is a few percent to a third above
    # (occasionally slightly below) a single pending row.
    lo = 1 - cfg.duplicates_max_reduction_pct / 100
    hi = 1 + cfg.duplicates_max_uplift_pct / 100
    for row_key in list(unmatched):
        row = by_id[row_key]
        if not row.amount:
            continue
        hits = [
            b for b in available()
            if b.amount and _same_sign(row.amount, b.amount)
            and lo <= abs(b.amount) / abs(row.amount) <= hi
        ]
        if not hits:
            continue
        if len(hits) > 1:
            hold([row_key], len(hits))
        else:
            claim("R3", [row_key], hits[0])

    # R4 — authorization probe: a dollar-ish pending row beside the real charge,
    # both superseded by the same posted row. It deliberately looks at booked
    # rows already claimed above and merges into that candidate, so the probe and
    # the charge are adjudicated and removed together.
    for row_key in list(unmatched):
        row = by_id[row_key]
        if not row.amount or abs(row.amount) > cfg.duplicates_auth_hold_max_cents:
            continue
        hits = [b for b in booked if b.amount and _same_sign(row.amount, b.amount)]
        if len(hits) != 1:
            if len(hits) > 1:
                hold([row_key], len(hits))
            continue
        target = hits[0]
        if target.id in claimed:
            index = claimed[target.id]
            existing = candidates[index]
            candidates[index] = dataclasses.replace(
                existing,
                rule=f"{existing.rule}+R4",
                pending=existing.pending + (row,),
            )
            unmatched.remove(row_key)
        else:
            claim("R4", [row_key], target)

    # R5 — reversal pair: a pending charge and an equal pending credit with no
    # posted survivor. Both rows go, so there is nothing left to check the
    # decision against; resolve_candidate never deletes these automatically.
    for a_key, b_key in combinations(list(unmatched), 2):
        if a_key not in unmatched or b_key not in unmatched:
            continue
        a, b = by_id[a_key], by_id[b_key]
        if a.amount and a.amount + b.amount == 0:
            candidates.append(Candidate("R5", (a, b)))
            unmatched.remove(a_key)
            unmatched.remove(b_key)

    return candidates


# ---------------------------------------------------------------------------
# Stage 2 — adjudication
# ---------------------------------------------------------------------------


def _describe(txn: Any, label: str) -> str:
    payee = txn.payee.name if txn.payee else "(none)"
    return f"""{label}:
- Date: {txn.get_date()}
- Payee (friendly name): {payee}
- Raw descriptor: {descriptor_text(txn) or '(none)'}
- Amount: {txn.amount / 100:+.2f} USD
- Account: {txn.account.name}"""


_RULE_EXPLANATIONS = {
    "R1": "both rows carry the same bank transaction id",
    "R2": "the pending row(s) sum exactly to the posted amount",
    "R3": "the posted amount falls within the tip-adjustment band above the pending one",
    "R4": "the pending row is small enough to be an authorization probe",
    "R5": "two pending rows cancel out with no posted row to match against",
}


def render_duplicate_prompt(candidate: Candidate) -> str:
    explanation = " and ".join(
        _RULE_EXPLANATIONS.get(part, part) for part in candidate.rule.split("+")
    )
    pending_block = "\n\n".join(
        _describe(row, f"Pending row {i}" if len(candidate.pending) > 1 else "Pending row")
        for i, row in enumerate(candidate.pending, start=1)
    )

    if candidate.booked is None:
        return f"""Two pending rows in the same account cancel each other out, with no
posted transaction to match them against. Evaluate whether they are a charge and
its reversal for a single purchase — meaning neither belongs in the budget.

{pending_block}

Matched because {explanation}.

Is this the same purchase (a charge and its own reversal)?

Return JSON.
"""

    delta = (
        f"{candidate.delta_pct:+.1f}%" if candidate.delta_pct is not None else "n/a"
    )
    total = sum(row.amount for row in candidate.pending)
    totals_line = (
        f"\nPending rows total {total / 100:+.2f} USD.\n"
        if len(candidate.pending) > 1
        else ""
    )

    return f"""A bank feed sometimes imports one purchase twice: once as a pending
authorization and again, under a different bank id and often a different amount,
once it posts. Evaluate whether the rows below are that same purchase.

{pending_block}

{_describe(candidate.booked, 'Posted row')}
{totals_line}
Matched because {explanation}. The posted amount is {delta} relative to the pending one.

Is this the same purchase?

Return JSON.
"""


# ---------------------------------------------------------------------------
# Stage 3 — action
# ---------------------------------------------------------------------------


def _refusal_reason(rows: tuple[Any, ...]) -> str | None:
    """Why this candidate must not be acted on automatically, if it must not.

    A pending row already paired as a transfer cannot be deleted without
    orphaning its partner; a split parent carries receipt children; a reconciled
    row has been checked against a statement by hand.
    """
    for row in rows:
        if row.is_parent:
            return "pending row is a split parent"
        if row.reconciled:
            return "pending row is reconciled"
        if row.transferred_id is not None:
            return "pending row is paired as a transfer"
    return None


def _extra(candidate: Candidate, reason: str | None = None) -> dict[str, Any]:
    extra: dict[str, Any] = {
        "rule": candidate.rule,
        "pending_ids": [row.id for row in candidate.pending],
        "pending_amounts_cents": [row.amount for row in candidate.pending],
        "booked_id": candidate.booked.id if candidate.booked is not None else None,
        "booked_amount_cents": (
            candidate.booked.amount if candidate.booked is not None else None
        ),
        "booked_date": (
            str(candidate.booked.get_date()) if candidate.booked is not None else None
        ),
        "delta_pct": round(candidate.delta_pct, 2) if candidate.delta_pct is not None else None,
    }
    if candidate.alternatives:
        extra["booked_candidate_count"] = candidate.alternatives
    if reason:
        extra["reason"] = reason
    return extra


def _tag_all(rows: tuple[Any, ...], tag: str) -> None:
    for row in rows:
        row.notes = append_tag(row.notes, tag)


def resolve_candidate(
    candidate: Candidate,
    llm: "LLMClient",
    audit: "AuditLogger",
    cfg: "Config",
    prompts: Any,
) -> None:
    anchor = candidate.pending[0]

    # Ambiguous sets are held, never resolved by picking the closest match.
    if candidate.rule == "ambiguous":
        _tag_all(candidate.pending, REVIEW_TAG)
        audit.log(
            anchor, {}, mode=cfg.duplicates_mode, action="held", pipeline="duplicates",
            extra=_extra(candidate, reason="more than one booked candidate"),
        )
        return

    # Refuse before spending a call — these can't be acted on either way.
    refusal = _refusal_reason(candidate.pending)
    if refusal:
        _tag_all(candidate.pending, REVIEW_TAG)
        audit.log(
            anchor, {}, mode=cfg.duplicates_mode, action="refused", pipeline="duplicates",
            extra=_extra(candidate, reason=refusal),
        )
        return

    response = llm.complete_json(prompts.DUPLICATE_SYSTEM, render_duplicate_prompt(candidate))

    if "error" in response:
        audit.log_failure(anchor, response["error"], pipeline="duplicates")
        return

    if not response.get("is_same_purchase", False):
        audit.log(
            anchor, response, mode="duplicate-rejected", action="none",
            pipeline="duplicates", extra=_extra(candidate),
        )
        return

    should_apply = (
        cfg.duplicates_mode == "apply"
        and _meets_threshold(response.get("confidence", "low"), cfg.duplicates_threshold)
        # R5 removes both rows with no posted survivor to check the decision
        # against, so it stays suggest-only whatever the mode says.
        and candidate.rule != "R5"
    )

    if should_apply:
        for row in candidate.pending:
            # Soft delete: sets tombstone = 1 and cascades to child splits, the
            # same operation the Actual UI performs.
            row.delete()
        action = "deleted"
    else:
        _tag_all(candidate.pending, SUGGEST_TAG)
        action = "tagged"

    audit.log(
        anchor, response, mode=cfg.duplicates_mode, action=action,
        pipeline="duplicates", extra=_extra(candidate),
    )


def process_duplicates(
    actual: Any,
    llm: "LLMClient",
    audit: "AuditLogger",
    cfg: "Config",
    prompts: Any,
) -> None:
    # A row tagged on an earlier run is left alone: the suggestion is already in
    # front of the user, and re-adjudicating it would spend a call per run.
    pending = [row for row in find_pending(actual.session) if not has_duplicate_marker(row.notes)]

    for cluster in cluster_pending(pending):
        booked = find_booked_candidates(actual.session, cluster, cfg.duplicates_window_days)
        for candidate in propose_candidates(cluster, booked, cfg):
            resolve_candidate(candidate, llm, audit, cfg, prompts)
