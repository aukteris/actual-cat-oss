"""Unit tests for the pending-duplicate pipeline.

Row fixtures mirror the shapes observed in the reference budget: tip uplift, an
authorization plus adjustment summing to the posted row, a $1 authorization
probe, a reversal pair, and the two false-positive shapes (monthly subscription,
second visit to the same merchant).
"""

import json
from datetime import date
from unittest.mock import MagicMock, patch

from actual_cat.duplicates import (
    REVIEW_TAG,
    SUGGEST_TAG,
    Candidate,
    cluster_pending,
    find_booked_candidates,
    process_duplicates,
    propose_candidates,
    render_duplicate_prompt,
    resolve_candidate,
)


def make_txn(
    id="p1",
    amount=-9625,
    *,
    pending=True,
    day=8,
    acct="acct-cc",
    descriptor="CORNER CANTINA",
    notes=None,
    financial_id=None,
    transaction_id=None,
    is_parent=0,
    is_child=0,
    reconciled=0,
    transferred_id=None,
    account_name="Credit Card",
) -> MagicMock:
    txn = MagicMock()
    txn.id = id
    txn.amount = amount
    txn.acct = acct
    # `date` is the integer form Actual stores (yyyymmdd); get_date() is the
    # date object the pipelines actually render and window on.
    txn.date = 20260800 + day
    txn.get_date.return_value = date(2026, 8, day)
    txn.notes = notes
    txn.imported_description = descriptor
    txn.payee = None
    txn.financial_id = financial_id
    txn.cleared = 0 if pending else 1
    txn.raw_synced_data = json.dumps(
        {"booked": not pending, "transactionId": transaction_id or f"TRN-{id}"}
    )
    txn.is_parent = is_parent
    txn.is_child = is_child
    txn.reconciled = reconciled
    txn.transferred_id = transferred_id
    txn.tombstone = 0
    txn.account = MagicMock()
    txn.account.name = account_name
    txn.account.offbudget = False
    return txn


def make_booked(id="b1", amount=-11625, **kwargs) -> MagicMock:
    return make_txn(id, amount, pending=False, **kwargs)


def make_cfg(mode="suggest", threshold="high") -> MagicMock:
    cfg = MagicMock()
    cfg.duplicates_mode = mode
    cfg.duplicates_threshold = threshold
    cfg.duplicates_window_days = 7
    cfg.duplicates_max_uplift_pct = 40
    cfg.duplicates_max_reduction_pct = 5
    cfg.duplicates_auth_hold_max_cents = 200
    return cfg


class MockLLM:
    def __init__(self, responses: list[dict]):
        self.responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def complete_json(self, system: str, user: str) -> dict:
        self.calls.append((system, user))
        return self.responses.pop(0)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


class TestClusterPending:
    def test_same_merchant_clusters_together(self):
        rows = [
            make_txn("a", -10547, descriptor="GREEN GROCER #1234"),
            make_txn("b", -147, descriptor="GREEN GROCER #1234 ANYTOWN"),
        ]
        clusters = cluster_pending(rows)
        assert len(clusters) == 1
        assert {r.id for r in clusters[0]} == {"a", "b"}

    def test_different_merchants_stay_apart(self):
        rows = [
            make_txn("a", descriptor="GREEN GROCER"),
            make_txn("b", descriptor="NETFLIX"),
        ]
        assert len(cluster_pending(rows)) == 2

    def test_same_merchant_in_different_accounts_stays_apart(self):
        rows = [
            make_txn("a", descriptor="GREEN GROCER", acct="acct-cc"),
            make_txn("b", descriptor="GREEN GROCER", acct="acct-visa"),
        ]
        assert len(cluster_pending(rows)) == 2


# ---------------------------------------------------------------------------
# Stage 1 — rules
# ---------------------------------------------------------------------------


class TestProposeCandidates:
    def test_r1_same_bank_id(self):
        pending = make_txn("p", -232855, financial_id="FIN-9")
        booked = make_booked("b", -232855, financial_id="FIN-9")
        [candidate] = propose_candidates([pending], [booked], make_cfg())
        assert candidate.rule == "R1"
        assert [r.id for r in candidate.pending] == ["p"]
        assert candidate.booked.id == "b"

    def test_r2_exact_duplicate(self):
        pending = make_txn("p", -17500, descriptor="Tumblewell")
        booked = make_booked("b", -17500, descriptor="Tumblewell.com")
        [candidate] = propose_candidates([pending], [booked], make_cfg())
        assert candidate.rule == "R2"
        assert candidate.delta_pct == 0

    def test_r2_cluster_sums_to_posted(self):
        # Grocery pickup: authorization plus adjustment, together the posted row.
        cluster = [make_txn("p1", -10547), make_txn("p2", -147)]
        booked = make_booked("b", -10694)
        [candidate] = propose_candidates(cluster, [booked], make_cfg())
        assert candidate.rule == "R2"
        assert {r.id for r in candidate.pending} == {"p1", "p2"}

    def test_r2_subset_sum_with_sign_flipped_member(self):
        # A refund leg comes along with the charge: -82.03 + 1.42 = -80.61.
        cluster = [make_txn("p1", -8203), make_txn("p2", 142)]
        booked = make_booked("b", -8061)
        [candidate] = propose_candidates(cluster, [booked], make_cfg())
        assert candidate.rule == "R2"
        assert {r.id for r in candidate.pending} == {"p1", "p2"}

    def test_r3_tip_uplift(self):
        [candidate] = propose_candidates(
            [make_txn("p", -9625)], [make_booked("b", -11625)], make_cfg()
        )
        assert candidate.rule == "R3"
        assert round(candidate.delta_pct, 1) == 20.8

    def test_r3_amount_decrease_within_band(self):
        [candidate] = propose_candidates(
            [make_txn("p", -10000)], [make_booked("b", -9830)], make_cfg()
        )
        assert candidate.rule == "R3"
        assert round(candidate.delta_pct, 1) == -1.7

    def test_r3_ignores_uplift_beyond_band(self):
        assert propose_candidates(
            [make_txn("p", -10000)], [make_booked("b", -20000)], make_cfg()
        ) == []

    def test_r3_ignores_opposite_sign(self):
        assert propose_candidates(
            [make_txn("p", -10000)], [make_booked("b", 10500)], make_cfg()
        ) == []

    def test_r4_probe_merges_into_the_tip_match(self):
        # Tip plus a $1 authorization probe, both superseded by one posted row.
        cluster = [make_txn("p1", -20498), make_txn("p2", -100)]
        booked = make_booked("b", -24000)
        [candidate] = propose_candidates(cluster, [booked], make_cfg())
        assert candidate.rule == "R3+R4"
        assert {r.id for r in candidate.pending} == {"p1", "p2"}
        assert candidate.booked.id == "b"

    def test_r4_probe_alone(self):
        [candidate] = propose_candidates(
            [make_txn("p", -100)], [make_booked("b", -24000)], make_cfg()
        )
        assert candidate.rule == "R4"

    def test_r5_reversal_pair(self):
        cluster = [make_txn("p1", -17500), make_txn("p2", 17500)]
        [candidate] = propose_candidates(cluster, [], make_cfg())
        assert candidate.rule == "R5"
        assert candidate.booked is None
        assert {r.id for r in candidate.pending} == {"p1", "p2"}

    def test_two_booked_candidates_are_held(self):
        # The monthly-subscription shape: same amount, more than one survivor.
        candidates = propose_candidates(
            [make_txn("p", -14900, descriptor="Tumblewell")],
            [
                make_booked("b1", -14900, descriptor="Tumblewell", day=5),
                make_booked("b2", -14900, descriptor="Tumblewell", day=12),
            ],
            make_cfg(),
        )
        [candidate] = candidates
        assert candidate.rule == "ambiguous"
        assert candidate.booked is None
        assert candidate.alternatives == 2

    def test_no_booked_candidate_yields_nothing(self):
        assert propose_candidates([make_txn("p", -23000)], [], make_cfg()) == []

    def test_each_pending_row_lands_in_at_most_one_candidate(self):
        cluster = [make_txn("p1", -10547), make_txn("p2", -147)]
        booked = [make_booked("b", -10694)]
        candidates = propose_candidates(cluster, booked, make_cfg())
        claimed = [r.id for c in candidates for r in c.pending]
        assert sorted(claimed) == ["p1", "p2"]


class TestFindBookedCandidates:
    def _session(self, rows):
        session = MagicMock()
        session.query.return_value.filter.return_value.all.return_value = rows
        return session

    def test_window_is_symmetric(self):
        # The posted row is dated a day *earlier* than the pending one, because
        # it carries the transaction date where the pending row carried the
        # authorization date.
        cluster = [make_txn("p", -11225, day=4, descriptor="WESTGATE BOURBON BAR")]
        earlier = make_booked("b", -13225, day=3, descriptor="WESTGATE BOURBON BAR")
        found = find_booked_candidates(self._session([earlier]), cluster, 7)
        assert [t.id for t in found] == ["b"]

    def test_rows_outside_the_window_are_dropped(self):
        cluster = [make_txn("p", -11225, day=20)]
        far = make_booked("b", -13225, day=3)
        assert find_booked_candidates(self._session([far]), cluster, 7) == []

    def test_pending_and_child_rows_are_never_candidates(self):
        cluster = [make_txn("p", -11225, day=8)]
        other_pending = make_txn("p2", -13225, day=9)
        child = make_booked("c", -13225, day=9, is_child=1)
        assert find_booked_candidates(self._session([other_pending, child]), cluster, 7) == []

    def test_unrelated_merchant_is_not_a_candidate(self):
        cluster = [make_txn("p", -11225, day=8, descriptor="CORNER CANTINA")]
        unrelated = make_booked("b", -13225, day=9, descriptor="NETFLIX")
        assert find_booked_candidates(self._session([unrelated]), cluster, 7) == []


# ---------------------------------------------------------------------------
# Stage 2/3 — adjudication and action
# ---------------------------------------------------------------------------


def tip_candidate(**kwargs):
    pending = make_txn("p", -9625, **kwargs)
    booked = make_booked("b", -11625)
    return Candidate("R3", (pending,), booked, 20.8), pending, booked


class TestResolveCandidate:
    def _run(self, candidate, responses, mode="suggest", threshold="high"):
        import actual_cat.prompts as prompts

        llm = MockLLM(responses)
        audit = MagicMock()
        resolve_candidate(candidate, llm, audit, make_cfg(mode, threshold), prompts)
        return llm, audit

    def test_suggest_tags_the_pending_row_only(self):
        candidate, pending, booked = tip_candidate()
        _, audit = self._run(candidate, [{"is_same_purchase": True, "confidence": "high"}])
        assert pending.notes == SUGGEST_TAG
        assert booked.notes is None
        pending.delete.assert_not_called()
        assert audit.log.call_args.kwargs["action"] == "tagged"

    def test_apply_deletes_the_pending_row_and_never_the_booked_one(self):
        candidate, pending, booked = tip_candidate()
        _, audit = self._run(
            candidate, [{"is_same_purchase": True, "confidence": "high"}], mode="apply"
        )
        pending.delete.assert_called_once()
        booked.delete.assert_not_called()
        assert audit.log.call_args.kwargs["action"] == "deleted"

    def test_apply_deletes_every_row_in_the_cluster(self):
        p1, p2 = make_txn("p1", -20498), make_txn("p2", -100)
        booked = make_booked("b", -24000)
        candidate = Candidate("R3+R4", (p1, p2), booked, 17.1)
        self._run(candidate, [{"is_same_purchase": True, "confidence": "high"}], mode="apply")
        p1.delete.assert_called_once()
        p2.delete.assert_called_once()

    def test_below_threshold_falls_back_to_suggest(self):
        candidate, pending, _ = tip_candidate()
        self._run(candidate, [{"is_same_purchase": True, "confidence": "medium"}], mode="apply")
        pending.delete.assert_not_called()
        assert pending.notes == SUGGEST_TAG

    def test_higher_confidence_than_threshold_still_applies(self):
        candidate, pending, _ = tip_candidate()
        self._run(
            candidate,
            [{"is_same_purchase": True, "confidence": "high"}],
            mode="apply",
            threshold="medium",
        )
        pending.delete.assert_called_once()

    def test_rejected_leaves_the_row_untouched(self):
        candidate, pending, _ = tip_candidate()
        _, audit = self._run(
            candidate, [{"is_same_purchase": False, "confidence": "high"}], mode="apply"
        )
        pending.delete.assert_not_called()
        assert pending.notes is None
        assert audit.log.call_args.kwargs["mode"] == "duplicate-rejected"

    def test_llm_failure_is_logged_and_skipped(self):
        candidate, pending, _ = tip_candidate()
        _, audit = self._run(candidate, [{"error": "timeout"}], mode="apply")
        pending.delete.assert_not_called()
        assert pending.notes is None
        audit.log_failure.assert_called_once()

    def test_r5_never_deletes_even_in_apply_mode(self):
        p1, p2 = make_txn("p1", -17500), make_txn("p2", 17500)
        candidate = Candidate("R5", (p1, p2))
        _, audit = self._run(
            candidate, [{"is_same_purchase": True, "confidence": "high"}], mode="apply"
        )
        p1.delete.assert_not_called()
        p2.delete.assert_not_called()
        assert p1.notes == SUGGEST_TAG and p2.notes == SUGGEST_TAG
        assert audit.log.call_args.kwargs["action"] == "tagged"

    def test_ambiguous_holds_without_calling_the_llm(self):
        pending = make_txn("p", -14900)
        candidate = Candidate("ambiguous", (pending,), None, None, 2)
        llm, audit = self._run(candidate, [], mode="apply")
        assert llm.calls == []
        pending.delete.assert_not_called()
        assert pending.notes == REVIEW_TAG
        assert audit.log.call_args.kwargs["action"] == "held"

    def test_refuses_split_parent(self):
        candidate, pending, _ = tip_candidate(is_parent=1)
        llm, audit = self._run(
            candidate, [{"is_same_purchase": True, "confidence": "high"}], mode="apply"
        )
        assert llm.calls == []
        pending.delete.assert_not_called()
        assert pending.notes == REVIEW_TAG
        assert audit.log.call_args.kwargs["action"] == "refused"

    def test_refuses_reconciled_row(self):
        candidate, pending, _ = tip_candidate(reconciled=1)
        llm, _ = self._run(
            candidate, [{"is_same_purchase": True, "confidence": "high"}], mode="apply"
        )
        assert llm.calls == []
        pending.delete.assert_not_called()

    def test_refuses_transfer_linked_row(self):
        # Deleting a paired leg would leave its partner pointing at a tombstone.
        candidate, pending, _ = tip_candidate(transferred_id="other-leg")
        llm, _ = self._run(
            candidate, [{"is_same_purchase": True, "confidence": "high"}], mode="apply"
        )
        assert llm.calls == []
        pending.delete.assert_not_called()

    def test_audit_extra_carries_both_rows(self):
        candidate, _, _ = tip_candidate()
        _, audit = self._run(candidate, [{"is_same_purchase": True, "confidence": "high"}])
        extra = audit.log.call_args.kwargs["extra"]
        assert extra["rule"] == "R3"
        assert extra["pending_ids"] == ["p"]
        assert extra["booked_id"] == "b"
        assert extra["booked_amount_cents"] == -11625
        assert extra["delta_pct"] == 20.8


class TestRenderDuplicatePrompt:
    def test_pair_prompt_includes_both_rows_and_the_delta(self):
        candidate, _, _ = tip_candidate()
        prompt = render_duplicate_prompt(candidate)
        assert "-96.25" in prompt and "-116.25" in prompt
        assert "+20.8%" in prompt
        assert "tip-adjustment band" in prompt

    def test_reversal_prompt_has_no_posted_row(self):
        candidate = Candidate("R5", (make_txn("p1", -17500), make_txn("p2", 17500)))
        prompt = render_duplicate_prompt(candidate)
        assert "Posted row" not in prompt
        assert "reversal" in prompt

    def test_cluster_prompt_shows_the_pending_total(self):
        candidate = Candidate(
            "R2", (make_txn("p1", -10547), make_txn("p2", -147)), make_booked("b", -10694), 0.0
        )
        prompt = render_duplicate_prompt(candidate)
        assert "Pending row 1" in prompt and "Pending row 2" in prompt
        assert "-106.94" in prompt


class TestProcessDuplicates:
    def _run(self, pending_rows, booked_rows, responses, mode="suggest"):
        import actual_cat.prompts as prompts

        llm = MockLLM(responses)
        audit = MagicMock()
        actual = MagicMock()

        with (
            patch("actual_cat.duplicates.find_pending", return_value=pending_rows),
            patch("actual_cat.duplicates.find_booked_candidates", return_value=booked_rows),
        ):
            process_duplicates(actual, llm, audit, make_cfg(mode), prompts)
        return llm, audit

    def test_end_to_end_suggest(self):
        pending = make_txn("p", -9625)
        booked = make_booked("b", -11625)
        llm, audit = self._run(
            [pending], [booked], [{"is_same_purchase": True, "confidence": "high"}]
        )
        assert len(llm.calls) == 1
        assert pending.notes == SUGGEST_TAG
        assert audit.log.call_args.kwargs["pipeline"] == "duplicates"

    def test_already_tagged_row_is_skipped(self):
        pending = make_txn("p", -9625, notes=f"{SUGGEST_TAG} SOME MEMO")
        llm, _ = self._run([pending], [make_booked("b", -11625)], [])
        assert llm.calls == []

    def test_held_row_is_not_re_adjudicated(self):
        pending = make_txn("p", -9625, notes=REVIEW_TAG)
        llm, _ = self._run([pending], [make_booked("b", -11625)], [])
        assert llm.calls == []

    def test_no_pending_rows_means_no_calls(self):
        llm, _ = self._run([], [], [])
        assert llm.calls == []
