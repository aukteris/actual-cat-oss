from unittest.mock import MagicMock, patch

from actual_cat.categorization import find_uncategorized, is_income_category, process_categorization


class MockLLM:
    def __init__(self, responses: list[dict]):
        self.responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def complete_json(self, system: str, user: str) -> dict:
        self.calls.append((system, user))
        return self.responses.pop(0)


def make_txn(
    id="txn-1",
    category_id=None,
    is_parent=0,
    transferred_id=None,
    tombstone=0,
    notes=None,
    imported_description="WHOLE FOODS",
    amount=-8500,
    date="2026-06-01",
    account_name="Checking",
    offbudget=False,
    payee_name="Whole Foods",
    pending=False,
) -> MagicMock:
    txn = MagicMock()
    txn.id = id
    # Bank-sync fields the defer_pending filter reads.
    txn.cleared = 0 if pending else 1
    txn.raw_synced_data = '{"booked": false}' if pending else None
    txn.category_id = category_id
    txn.is_parent = is_parent
    txn.transferred_id = transferred_id
    txn.tombstone = tombstone
    txn.notes = notes
    txn.imported_description = imported_description
    txn.amount = amount
    txn.date = date
    txn.account = MagicMock()
    txn.account.name = account_name
    txn.account.offbudget = offbudget
    txn.account.__bool__ = lambda self: True
    txn.payee = MagicMock()
    txn.payee.name = payee_name
    return txn


def make_cfg(mode="suggest", threshold="high", history_enabled=False) -> MagicMock:
    cfg = MagicMock()
    cfg.categorization_mode = mode
    cfg.categorization_threshold = threshold
    cfg.history_enabled = history_enabled
    cfg.history_payee_top_n = 3
    cfg.history_min_count = 1
    cfg.duplicates_defer_pending = False
    return cfg


class TestFindUncategorized:
    def test_includes_uncategorized(self):
        txn = make_txn()
        with patch("actual_cat.categorization.get_transactions", return_value=[txn]):
            result = find_uncategorized(MagicMock())
        assert txn in result

    def test_excludes_already_categorized(self):
        txn = make_txn(category_id="cat-uuid")
        with patch("actual_cat.categorization.get_transactions", return_value=[txn]):
            result = find_uncategorized(MagicMock())
        assert result == []

    def test_excludes_split_parent(self):
        txn = make_txn(is_parent=1)
        with patch("actual_cat.categorization.get_transactions", return_value=[txn]):
            result = find_uncategorized(MagicMock())
        assert result == []

    def test_excludes_already_paired_transfer(self):
        txn = make_txn(transferred_id="other-txn-id")
        with patch("actual_cat.categorization.get_transactions", return_value=[txn]):
            result = find_uncategorized(MagicMock())
        assert result == []

    def test_excludes_tombstoned(self):
        txn = make_txn(tombstone=1)
        with patch("actual_cat.categorization.get_transactions", return_value=[txn]):
            result = find_uncategorized(MagicMock())
        assert result == []

    def test_excludes_offbudget_account(self):
        txn = make_txn(offbudget=True)
        with patch("actual_cat.categorization.get_transactions", return_value=[txn]):
            result = find_uncategorized(MagicMock())
        assert result == []

    def test_includes_pending_by_default(self):
        txn = make_txn(pending=True)
        with patch("actual_cat.categorization.get_transactions", return_value=[txn]):
            result = find_uncategorized(MagicMock())
        assert result == [txn]

    def test_excludes_pending_when_deferred(self):
        # A pending row may yet turn out to be a duplicate of its own posted
        # counterpart; categorizing it spends a call on a row headed for the bin.
        txn = make_txn(pending=True)
        with patch("actual_cat.categorization.get_transactions", return_value=[txn]):
            result = find_uncategorized(MagicMock(), defer_pending=True)
        assert result == []

    def test_deferring_keeps_booked_rows(self):
        txn = make_txn()
        with patch("actual_cat.categorization.get_transactions", return_value=[txn]):
            result = find_uncategorized(MagicMock(), defer_pending=True)
        assert result == [txn]

    def test_excludes_ai_marker(self):
        txn = make_txn(notes="#ai:groceries-food")
        with patch("actual_cat.categorization.get_transactions", return_value=[txn]):
            result = find_uncategorized(MagicMock())
        assert result == []


class TestIsIncomeCategory:
    def test_true_for_category_in_income_group(self):
        cat = MagicMock()
        cat.id = "cat-1"
        group = MagicMock()
        group.is_income = 1
        group.categories = [cat]
        with patch("actual_cat.categorization.get_category_groups", return_value=[group]):
            assert is_income_category(MagicMock(), "cat-1") is True

    def test_false_for_category_in_expense_group(self):
        cat = MagicMock()
        cat.id = "cat-1"
        group = MagicMock()
        group.is_income = 0
        group.categories = [cat]
        with patch("actual_cat.categorization.get_category_groups", return_value=[group]):
            assert is_income_category(MagicMock(), "cat-1") is False

    def test_false_when_id_not_found(self):
        with patch("actual_cat.categorization.get_category_groups", return_value=[]):
            assert is_income_category(MagicMock(), "cat-1") is False


class TestProcessCategorization:
    def _run(self, txns, llm_responses, mode="suggest", threshold="high"):
        import actual_cat.prompts as prompts

        llm = MockLLM(llm_responses)
        audit = MagicMock()
        cfg = make_cfg(mode=mode, threshold=threshold)
        actual = MagicMock()
        actual.session = MagicMock()

        with (
            patch("actual_cat.categorization.get_transactions", return_value=txns),
            patch("actual_cat.categorization.get_category_groups", return_value=[]),
            patch("actual_cat.categorization.lookup_category_id", return_value="cat-uuid"),
        ):
            process_categorization(actual, llm, audit, cfg, "schema text", prompts)

        return llm, audit, txns

    def test_suggest_mode_tags_not_applies(self):
        txn = make_txn()
        self._run(
            [txn],
            [{"category": "Groceries / Food", "confidence": "high", "tags": [], "reasoning": "x"}],
            mode="suggest",
        )
        assert "#ai:groceries-food" in txn.notes
        assert txn.category_id is None  # not applied

    def test_apply_mode_high_confidence_applies(self):
        txn = make_txn()
        self._run(
            [txn],
            [{"category": "Groceries / Food", "confidence": "high", "tags": [], "reasoning": "x"}],
            mode="apply", threshold="high",
        )
        assert txn.category_id == "cat-uuid"
        assert "#ai-assisted" in txn.notes

    def test_apply_mode_low_confidence_suggests(self):
        txn = make_txn()
        self._run(
            [txn],
            [{"category": "Groceries / Food", "confidence": "low", "tags": [], "reasoning": "x"}],
            mode="apply", threshold="high",
        )
        assert txn.category_id is None
        assert "#ai:groceries-food" in txn.notes

    def test_uncertain_always_tagged_never_applied(self):
        txn = make_txn()
        self._run(
            [txn],
            [{"category": "Uncertain", "confidence": "low", "tags": [], "reasoning": "x"}],
            mode="apply",
        )
        assert txn.category_id is None
        assert "#ai:uncertain" in txn.notes

    def test_invalid_category_logs_failure(self):
        txn = make_txn()
        import actual_cat.prompts as prompts

        llm = MockLLM([
            {"category": "Made Up / Category", "confidence": "high", "tags": [], "reasoning": "x"}
        ])
        audit = MagicMock()
        cfg = make_cfg()
        actual = MagicMock()

        with (
            patch("actual_cat.categorization.get_transactions", return_value=[txn]),
            patch("actual_cat.categorization.get_category_groups", return_value=[]),
            patch("actual_cat.categorization.lookup_category_id", return_value=None),
        ):
            process_categorization(actual, llm, audit, cfg, "schema", prompts)

        audit.log_failure.assert_called_once()
        assert txn.category_id is None

    def _run_income_on_outflow(self, mode):
        txn = make_txn()  # amount=-8500, an outflow
        import actual_cat.prompts as prompts

        llm = MockLLM([
            {"category": "Income / Salary", "confidence": "high", "tags": [], "reasoning": "x"}
        ])
        audit = MagicMock()
        cfg = make_cfg(mode=mode, threshold="high")
        actual = MagicMock()

        income_cat = MagicMock()
        income_cat.id = "cat-uuid"
        income_group = MagicMock()
        income_group.is_income = 1
        income_group.categories = [income_cat]

        with (
            patch("actual_cat.categorization.get_transactions", return_value=[txn]),
            patch("actual_cat.categorization.get_category_groups", return_value=[income_group]),
            patch("actual_cat.categorization.lookup_category_id", return_value="cat-uuid"),
        ):
            process_categorization(actual, llm, audit, cfg, "schema", prompts)

        return txn, audit

    def test_income_category_on_outflow_rejected_in_apply_mode(self):
        txn, audit = self._run_income_on_outflow(mode="apply")
        audit.log_failure.assert_called_once()
        assert txn.category_id is None
        assert txn.notes is None  # no tag written either

    def test_income_category_on_outflow_rejected_in_suggest_mode(self):
        # The safeguard sits before the mode check, so it blocks regardless
        # of categorization_mode — mirrors the hallucinated-category branch.
        txn, audit = self._run_income_on_outflow(mode="suggest")
        audit.log_failure.assert_called_once()
        assert txn.category_id is None
        assert txn.notes is None

    def test_llm_error_logs_failure_and_continues(self):
        txn1 = make_txn(id="t1")
        txn2 = make_txn(id="t2")
        self._run(
            [txn1, txn2],
            [
                {"error": "LLM call failure: timeout"},
                {"category": "Groceries / Food", "confidence": "high",
                 "tags": [], "reasoning": "x"},
            ],
        )
        # t1 errored, t2 should still be processed
        assert "#ai:groceries-food" in txn2.notes

    def test_idempotency_skips_marked_transactions(self):
        txn = make_txn(notes="#ai:groceries-food")
        import actual_cat.prompts as prompts

        llm = MockLLM([])
        audit = MagicMock()
        cfg = make_cfg()
        actual = MagicMock()

        with patch("actual_cat.categorization.get_transactions", return_value=[txn]):
            process_categorization(actual, llm, audit, cfg, "schema", prompts)

        assert llm.calls == []  # LLM never called

    def test_extra_tags_appended(self):
        txn = make_txn()
        self._run(
            [txn],
            [{"category": "Groceries / Food", "confidence": "high",
              "tags": ["#tax:deductible:home"], "reasoning": "x"}],
        )
        assert "#tax:deductible:home" in txn.notes


class TestRenderUserPrompt:
    def test_no_hint_when_empty(self):
        from actual_cat.categorization import render_user_prompt
        prompt = render_user_prompt(make_txn(), "schema text")
        assert "Past categorizations for this payee" not in prompt

    def test_hint_appended_when_provided(self):
        from actual_cat.categorization import render_user_prompt
        hint = "\nPast categorizations for this payee:\n- Groceries / Food (8×)\n"
        prompt = render_user_prompt(make_txn(), "schema text", hint)
        assert "Past categorizations for this payee" in prompt
        assert "Groceries / Food (8×)" in prompt


class TestProcessCategorizationHistory:
    def test_payee_history_reaches_prompt(self):
        from collections import Counter

        import actual_cat.prompts as prompts

        txn = make_txn()
        txn.payee_id = "payee-1"
        llm = MockLLM(
            [{"category": "Groceries / Food", "confidence": "high", "tags": [], "reasoning": "x"}]
        )
        audit = MagicMock()
        cfg = make_cfg(history_enabled=True)
        actual = MagicMock()

        with (
            patch("actual_cat.categorization.get_transactions", return_value=[txn]),
            patch("actual_cat.categorization.lookup_category_id", return_value="cat-uuid"),
            patch(
                "actual_cat.categorization.history.build_category_path_map",
                return_value={},
            ),
            patch(
                "actual_cat.categorization.history.build_histories",
                return_value=({"payee-1": Counter({"Groceries / Food": 8})}, {}),
            ),
        ):
            process_categorization(actual, llm, audit, cfg, "schema", prompts)

        # The single LLM call's user message should carry the payee history hint.
        _, user_msg = llm.calls[0]
        assert "Past categorizations for this payee" in user_msg
        assert "Groceries / Food (8×)" in user_msg
