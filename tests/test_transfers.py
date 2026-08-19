from unittest.mock import MagicMock, patch

from actual_cat.transfers import (
    pair_as_transfer,
    process_transfers,
    relink_recreated_transfers,
    repair_transfer_pairs,
)


def make_txn(
    id="txn-1",
    amount=-50000,
    acct="acct-checking",
    date="2026-06-01",
    transferred_id=None,
    tombstone=0,
    notes=None,
    imported_description=None,
    account_name="Checking",
    offbudget=False,
    category_id=None,
    is_parent=0,
    is_child=0,
    reconciled=0,
    financial_id=None,
    pending=False,
) -> MagicMock:
    txn = MagicMock()
    txn.id = id
    # Bank-sync fields the defer_pending filter reads.
    txn.cleared = 0 if pending else 1
    txn.raw_synced_data = '{"booked": false}' if pending else None
    txn.amount = amount
    txn.acct = acct
    txn.date = date
    txn.transferred_id = transferred_id
    txn.tombstone = tombstone
    txn.notes = notes
    txn.imported_description = imported_description
    txn.category_id = category_id
    txn.is_parent = is_parent
    txn.is_child = is_child
    txn.reconciled = reconciled
    txn.financial_id = financial_id
    txn.account = MagicMock()
    txn.account.name = account_name
    txn.account.offbudget = offbudget
    # Each account owns a transfer payee (account.payee); its id is what a
    # transfer leg in the *other* account points at.
    txn.account.payee = MagicMock()
    txn.account.payee.id = f"xferpayee-{account_name}"
    txn.payee = None
    return txn


class MockLLM:
    def __init__(self, responses: list[dict]):
        self.responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def complete_json(self, system: str, user: str) -> dict:
        self.calls.append((system, user))
        return self.responses.pop(0)


class TestPairAsTransfer:
    def test_sets_transferred_id_both_sides(self):
        a = make_txn(id="a")
        b = make_txn(id="b")
        pair_as_transfer(a, b)
        assert a.transferred_id == "b"
        assert b.transferred_id == "a"

    def test_sets_each_leg_to_other_accounts_transfer_payee(self):
        a = make_txn(id="a", account_name="Checking")
        b = make_txn(id="b", account_name="Savings")
        pair_as_transfer(a, b)
        # leg in Checking points at Savings's transfer payee, and vice versa
        assert a.payee_id == "xferpayee-Savings"
        assert b.payee_id == "xferpayee-Checking"

    def test_clears_category_on_both_legs(self):
        a = make_txn(id="a", category_id="cat-1")
        b = make_txn(id="b", category_id="cat-2")
        pair_as_transfer(a, b)
        assert a.category_id is None
        assert b.category_id is None


class TestFindTransferCandidates:
    def _run(self, candidates, defer_pending=False):
        from actual_cat.transfers import find_transfer_candidates

        session = MagicMock()
        chain = MagicMock()
        chain.filter.return_value = chain
        chain.all.return_value = candidates
        session.query.return_value = chain

        txn = make_txn(id="a", amount=-50000, acct="checking")
        txn.get_date.return_value = __import__("datetime").date(2026, 6, 1)
        return find_transfer_candidates(session, txn, 3, defer_pending)

    def test_pending_partner_matched_by_default(self):
        partner = make_txn(id="b", amount=50000, acct="savings", pending=True)
        assert self._run([partner]) == [partner]

    def test_pending_partner_excluded_when_deferred(self):
        # Pairing a pending leg leaves its partner pointing at a tombstone once
        # the duplicate is deleted — a failure already seen in the wild.
        partner = make_txn(id="b", amount=50000, acct="savings", pending=True)
        assert self._run([partner], defer_pending=True) == []

    def test_booked_partner_kept_when_deferred(self):
        partner = make_txn(id="b", amount=50000, acct="savings")
        assert self._run([partner], defer_pending=True) == [partner]


class TestProcessTransfers:
    def _run(self, txns, llm_responses, mode="suggest", threshold="high"):
        import actual_cat.prompts as prompts

        llm = MockLLM(llm_responses)
        audit = MagicMock()
        cfg = MagicMock()
        cfg.transfer_mode = mode
        cfg.transfer_threshold = threshold
        cfg.transfer_window_days = 3
        cfg.duplicates_defer_pending = False
        actual = MagicMock()
        actual.session = MagicMock()

        with (
            patch("actual_cat.categorization.get_transactions", return_value=txns),
            patch("actual_cat.transfers.find_transfer_candidates") as mock_candidates,
        ):
            # Default: no candidates unless overridden per test
            mock_candidates.return_value = []
            process_transfers(actual, llm, audit, cfg, prompts)

        return llm, audit, mock_candidates

    def test_no_candidates_no_llm_calls(self):
        txn = make_txn()
        llm, audit, _ = self._run([txn], [])
        assert llm.calls == []

    def test_suggest_mode_tags_both_sides(self):
        import actual_cat.prompts as prompts

        txn_a = make_txn(id="a", amount=-50000, acct="checking")
        txn_b = make_txn(id="b", amount=50000, acct="savings")

        llm = MockLLM([{"is_transfer": True, "confidence": "high", "reasoning": "x"}])
        audit = MagicMock()
        cfg = MagicMock()
        cfg.transfer_mode = "suggest"
        cfg.transfer_threshold = "high"
        cfg.transfer_window_days = 3
        cfg.duplicates_defer_pending = False
        actual = MagicMock()

        with (
            patch("actual_cat.categorization.get_transactions", return_value=[txn_a]),
            patch("actual_cat.transfers.find_transfer_candidates", return_value=[txn_b]),
        ):
            process_transfers(actual, llm, audit, cfg, prompts)

        assert "#ai-suggested-transfer" in txn_a.notes
        assert "#ai-suggested-transfer" in txn_b.notes
        assert txn_a.transferred_id != txn_b.id  # not paired in suggest mode

    def test_apply_mode_high_confidence_pairs(self):
        import actual_cat.prompts as prompts

        txn_a = make_txn(id="a", amount=-50000, acct="checking")
        txn_b = make_txn(id="b", amount=50000, acct="savings")

        llm = MockLLM([{"is_transfer": True, "confidence": "high", "reasoning": "x"}])
        audit = MagicMock()
        cfg = MagicMock()
        cfg.transfer_mode = "apply"
        cfg.transfer_threshold = "high"
        cfg.transfer_window_days = 3
        cfg.duplicates_defer_pending = False
        actual = MagicMock()

        with (
            patch("actual_cat.categorization.get_transactions", return_value=[txn_a]),
            patch("actual_cat.transfers.find_transfer_candidates", return_value=[txn_b]),
        ):
            process_transfers(actual, llm, audit, cfg, prompts)

        assert txn_a.transferred_id == "b"
        assert txn_b.transferred_id == "a"
        assert "#ai-assisted" in txn_a.notes
        assert "#ai-assisted" in txn_b.notes

    def test_suggest_mode_strips_stale_category_tag(self):
        import actual_cat.prompts as prompts

        txn_a = make_txn(id="a", amount=-50000, acct="checking")
        txn_b = make_txn(
            id="b", amount=50000, acct="savings",
            notes="#ai:uncertain ONLINE TRANSFER TO SAVINGS",
        )

        llm = MockLLM([{"is_transfer": True, "confidence": "high", "reasoning": "x"}])
        audit = MagicMock()
        cfg = MagicMock()
        cfg.transfer_mode = "suggest"
        cfg.transfer_threshold = "high"
        cfg.transfer_window_days = 3
        cfg.duplicates_defer_pending = False
        actual = MagicMock()

        with (
            patch("actual_cat.categorization.get_transactions", return_value=[txn_a]),
            patch("actual_cat.transfers.find_transfer_candidates", return_value=[txn_b]),
        ):
            process_transfers(actual, llm, audit, cfg, prompts)

        assert "#ai:uncertain" not in txn_b.notes
        assert "#ai-suggested-transfer" in txn_b.notes
        assert "ONLINE TRANSFER TO SAVINGS" in txn_b.notes

    def test_apply_mode_strips_stale_category_tag(self):
        import actual_cat.prompts as prompts

        txn_a = make_txn(id="a", amount=-50000, acct="checking")
        txn_b = make_txn(
            id="b", amount=50000, acct="savings",
            notes="#ai:uncertain ONLINE TRANSFER TO SAVINGS",
        )

        llm = MockLLM([{"is_transfer": True, "confidence": "high", "reasoning": "x"}])
        audit = MagicMock()
        cfg = MagicMock()
        cfg.transfer_mode = "apply"
        cfg.transfer_threshold = "high"
        cfg.transfer_window_days = 3
        cfg.duplicates_defer_pending = False
        actual = MagicMock()

        with (
            patch("actual_cat.categorization.get_transactions", return_value=[txn_a]),
            patch("actual_cat.transfers.find_transfer_candidates", return_value=[txn_b]),
        ):
            process_transfers(actual, llm, audit, cfg, prompts)

        assert txn_b.transferred_id == "a"
        assert "#ai:uncertain" not in txn_b.notes
        assert "#ai-assisted" in txn_b.notes
        assert "ONLINE TRANSFER TO SAVINGS" in txn_b.notes

    def test_rejected_transfer_no_tags(self):
        import actual_cat.prompts as prompts

        txn_a = make_txn(id="a", amount=-50000)
        txn_b = make_txn(id="b", amount=50000)

        llm = MockLLM([{"is_transfer": False, "confidence": "low", "reasoning": "coincidence"}])
        audit = MagicMock()
        cfg = MagicMock()
        cfg.transfer_mode = "suggest"
        cfg.transfer_threshold = "high"
        cfg.transfer_window_days = 3
        cfg.duplicates_defer_pending = False
        actual = MagicMock()

        with (
            patch("actual_cat.categorization.get_transactions", return_value=[txn_a]),
            patch("actual_cat.transfers.find_transfer_candidates", return_value=[txn_b]),
        ):
            process_transfers(actual, llm, audit, cfg, prompts)

        assert txn_a.notes is None
        assert txn_b.notes is None
        audit.log.assert_called_once()
        assert audit.log.call_args.kwargs["action"] == "none"

    def test_pair_deduplication(self):
        """A↔B should only be evaluated once, not once from each side."""
        import actual_cat.prompts as prompts

        txn_a = make_txn(id="a", amount=-50000, acct="checking")
        txn_b = make_txn(id="b", amount=50000, acct="savings")

        llm = MockLLM([{"is_transfer": True, "confidence": "high", "reasoning": "x"}])
        audit = MagicMock()
        cfg = MagicMock()
        cfg.transfer_mode = "suggest"
        cfg.transfer_threshold = "high"
        cfg.transfer_window_days = 3
        cfg.duplicates_defer_pending = False
        actual = MagicMock()

        with (
            patch("actual_cat.categorization.get_transactions", return_value=[txn_a, txn_b]),
            patch("actual_cat.transfers.find_transfer_candidates", side_effect=[[txn_b], [txn_a]]),
        ):
            process_transfers(actual, llm, audit, cfg, prompts)

        assert len(llm.calls) == 1  # only evaluated once

    def test_llm_error_logs_failure(self):
        import actual_cat.prompts as prompts

        txn_a = make_txn(id="a", amount=-50000)
        txn_b = make_txn(id="b", amount=50000)

        llm = MockLLM([{"error": "LLM call failure: timeout"}])
        audit = MagicMock()
        cfg = MagicMock()
        cfg.transfer_mode = "suggest"
        cfg.transfer_threshold = "high"
        cfg.transfer_window_days = 3
        cfg.duplicates_defer_pending = False
        actual = MagicMock()

        with (
            patch("actual_cat.categorization.get_transactions", return_value=[txn_a]),
            patch("actual_cat.transfers.find_transfer_candidates", return_value=[txn_b]),
        ):
            process_transfers(actual, llm, audit, cfg, prompts)

        audit.log_failure.assert_called_once()


class TestRepairTransferPairs:
    """repair_transfer_pairs() re-asserts the transfer invariant that bank-sync
    reconciliation silently breaks: payee_id/category_id are fair game to fix on
    any paired row, but #ai-assisted is only ever restored, never added new."""

    def _run(self, rows):
        session = MagicMock()
        chain = MagicMock()
        chain.filter.return_value = chain
        chain.all.return_value = rows
        session.query.return_value = chain

        actual = MagicMock()
        actual.session = session
        audit = MagicMock()

        repaired = repair_transfer_pairs(actual, audit)
        return repaired, audit

    def test_payee_reset_is_repaired_partner_left_alone(self):
        txn = make_txn(id="a", account_name="Checking")
        partner = make_txn(id="b", account_name="Savings")
        # bank sync overwrote txn's transfer payee with the imported merchant payee
        # and stripped the marker from notes; partner is untouched.
        txn.payee_id = "merchant-payee"
        txn.notes = "ONLINE TRANSFER TO SAVINGS"
        partner.payee_id = "xferpayee-Checking"
        partner.notes = "#ai-assisted"
        txn.transfer = partner

        repaired, audit = self._run([txn])

        assert repaired == 1
        assert txn.payee_id == "xferpayee-Savings"
        assert partner.payee_id == "xferpayee-Checking"  # unchanged
        assert partner.notes == "#ai-assisted"  # unchanged
        audit.log.assert_called_once()
        assert audit.log.call_args.kwargs["action"] == "repaired"

    def test_marker_not_restored_when_partner_lacks_it(self):
        txn = make_txn(id="a", account_name="Checking")
        partner = make_txn(id="b", account_name="Savings")
        txn.payee_id = "merchant-payee"
        txn.notes = "ONLINE TRANSFER TO SAVINGS"
        partner.payee_id = "xferpayee-Checking"
        partner.notes = None  # partner was never AI-tagged
        txn.transfer = partner

        repaired, _ = self._run([txn])

        assert repaired == 1
        assert txn.payee_id == "xferpayee-Savings"
        assert "#ai-assisted" not in (txn.notes or "")

    def test_user_made_pair_payee_repaired_but_gains_no_marker(self):
        txn = make_txn(id="a", account_name="Checking")
        partner = make_txn(id="b", account_name="Savings")
        # Neither leg was ever AI-tagged — a manual transfer the user created —
        # but bank sync still reset this leg's payee.
        txn.payee_id = "wrong-payee"
        txn.notes = None
        partner.payee_id = "xferpayee-Checking"
        partner.notes = None
        txn.transfer = partner

        repaired, _ = self._run([txn])

        assert repaired == 1
        assert txn.payee_id == "xferpayee-Savings"
        assert "#ai-assisted" not in (txn.notes or "")

    def test_healthy_pair_is_noop_and_emits_no_audit_record(self):
        txn = make_txn(id="a", account_name="Checking")
        partner = make_txn(id="b", account_name="Savings")
        txn.payee_id = "xferpayee-Savings"
        partner.payee_id = "xferpayee-Checking"
        txn.notes = "#ai-assisted"
        partner.notes = "#ai-assisted"
        txn.transfer = partner

        repaired, audit = self._run([txn])

        assert repaired == 0
        audit.log.assert_not_called()

    def test_orphaned_partner_reported_not_repaired(self):
        txn = make_txn(id="a", account_name="Checking")
        txn.payee_id = "merchant-payee"  # would look damaged if it had a partner
        txn.transferred_id = "ghost-id"
        # Transactions.transfer is conditioned on the remote row's tombstone,
        # so both "missing" and "tombstoned" partners surface as None here.
        txn.transfer = None

        repaired, audit = self._run([txn])

        assert repaired == 0
        assert txn.payee_id == "merchant-payee"  # untouched
        audit.log.assert_called_once()
        assert audit.log.call_args.kwargs["action"] == "orphaned"


class TestRelinkRecreatedTransfers:
    """A sync triggered from the Actual UI deletes and re-creates a leg instead of
    reconciling it. Both sides end up unpaired, so repair_transfer_pairs() cannot
    see the damage at all — the tombstoned predecessor is the only record of what
    the pair used to be, and relinking from it must never become a guess."""

    def _run(self, rows):
        session = MagicMock()
        chain = MagicMock()
        chain.all.return_value = rows
        session.query.return_value = chain

        actual = MagicMock()
        actual.session = session
        audit = MagicMock()

        relinked = relink_recreated_transfers(actual, audit)
        return relinked, audit

    def _scenario(self, **partner_kwargs):
        """The observed shape: savings leg re-created under a new id, same
        financial_id; its predecessor is tombstoned but still points at the
        checking leg, which Actual unpaired on its way out."""
        replacement = make_txn(
            id="savings-new", amount=25, acct="acct-savings",
            account_name="Savings", financial_id="TRN-abc",
        )
        predecessor = make_txn(
            id="savings-old", amount=25, acct="acct-savings", account_name="Savings",
            financial_id="TRN-abc", tombstone=1, transferred_id="checking-1",
        )
        defaults = dict(
            id="checking-1", amount=-25, acct="acct-checking",
            account_name="Checking", notes="#ai-assisted TRANSFER",
        )
        defaults.update(partner_kwargs)
        partner = make_txn(**defaults)
        return replacement, predecessor, partner

    def test_recreated_leg_is_relinked_without_an_llm_call(self):
        replacement, predecessor, partner = self._scenario()

        relinked, audit = self._run([replacement, predecessor, partner])

        assert relinked == 1
        assert replacement.transferred_id == "checking-1"
        assert partner.transferred_id == "savings-new"
        assert replacement.payee_id == "xferpayee-Checking"
        assert partner.payee_id == "xferpayee-Savings"
        assert replacement.category_id is None
        assert audit.log.call_args.kwargs["action"] == "relinked"
        assert audit.log.call_args.kwargs["extra"]["partner_id"] == "checking-1"

    def test_marker_is_restored_from_the_surviving_partner(self):
        replacement, predecessor, partner = self._scenario()

        self._run([replacement, predecessor, partner])

        # The predecessor had already lost its own marker to a reconcile; the
        # surviving partner is what proves the pair was ours.
        assert "#ai-assisted" in replacement.notes

    def test_user_made_pair_gains_no_marker(self):
        replacement, predecessor, partner = self._scenario(notes="RECURRING TRANSFER")

        relinked, _ = self._run([replacement, predecessor, partner])

        assert relinked == 1
        assert "#ai-assisted" not in (replacement.notes or "")

    def test_refuses_when_prior_partner_is_already_paired_elsewhere(self):
        replacement, predecessor, partner = self._scenario(transferred_id="someone-else")

        relinked, audit = self._run([replacement, predecessor, partner])

        assert relinked == 0
        assert replacement.transferred_id is None
        assert audit.log.call_args.kwargs["action"] == "relink-refused"

    def test_refuses_when_predecessors_disagree_on_the_partner(self):
        replacement, predecessor, partner = self._scenario()
        other = make_txn(
            id="savings-older", amount=25, acct="acct-savings", account_name="Savings",
            financial_id="TRN-abc", tombstone=1, transferred_id="checking-2",
        )

        relinked, audit = self._run([replacement, predecessor, other, partner])

        assert relinked == 0
        assert replacement.transferred_id is None
        assert "disagree" in audit.log.call_args.kwargs["extra"]["reason"]

    def test_refuses_when_amounts_are_no_longer_inverse(self):
        replacement, predecessor, partner = self._scenario(amount=-9999)

        relinked, audit = self._run([replacement, predecessor, partner])

        assert relinked == 0
        assert "inverse" in audit.log.call_args.kwargs["extra"]["reason"]

    def test_refuses_when_financial_id_is_not_unique_among_live_rows(self):
        replacement, predecessor, partner = self._scenario()
        twin = make_txn(
            id="savings-twin", amount=25, acct="acct-savings",
            account_name="Savings", financial_id="TRN-abc",
        )

        relinked, audit = self._run([replacement, predecessor, twin, partner])

        assert relinked == 0
        assert "more than one live row" in audit.log.call_args.kwargs["extra"]["reason"]

    def test_reconciled_replacement_is_left_alone_silently(self):
        replacement, predecessor, partner = self._scenario()
        replacement.reconciled = 1

        relinked, audit = self._run([replacement, predecessor, partner])

        assert relinked == 0
        audit.log.assert_not_called()

    def test_no_predecessor_is_a_silent_noop(self):
        replacement, _, partner = self._scenario()

        relinked, audit = self._run([replacement, partner])

        assert relinked == 0
        assert replacement.transferred_id is None
        audit.log.assert_not_called()

    def test_already_paired_rows_are_not_touched(self):
        """The live state on prod today: the replacement is already paired, and
        three tombstoned predecessors still point at the same partner."""
        replacement, predecessor, partner = self._scenario()
        replacement.transferred_id = "checking-1"
        partner.transferred_id = "savings-new"

        relinked, audit = self._run([replacement, predecessor, partner])

        assert relinked == 0
        audit.log.assert_not_called()
