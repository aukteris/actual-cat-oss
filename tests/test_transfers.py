from unittest.mock import MagicMock, patch

from actual_cat.transfers import pair_as_transfer, process_transfers


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
