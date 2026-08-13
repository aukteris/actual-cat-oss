from unittest.mock import MagicMock

from actual_cat.categorization import render_user_prompt
from actual_cat.transfers import render_transfer_prompt


def make_txn(
    imported_description=None,
    payee_name=None,
    amount=-5000,
    date="2026-06-01",
    account_name="Checking",
    offbudget=False,
    notes=None,
) -> MagicMock:
    txn = MagicMock()
    txn.imported_description = imported_description
    txn.payee = MagicMock(name=payee_name) if payee_name else None
    txn.amount = amount
    txn.date = date
    txn.account = MagicMock()
    txn.account.name = account_name
    txn.account.offbudget = offbudget
    txn.notes = notes
    return txn


class TestRenderUserPrompt:
    def test_all_fields_present(self):
        # payee.name is the friendly name; notes is the raw bank descriptor
        txn = make_txn(imported_description="SQ *COFFEE SHOP", payee_name="Coffee Shop",
                       amount=-450, notes="SQ *COFFEE SHOP 123 MAIN ST ANYTOWN ST USA")
        text = render_user_prompt(txn, "Discretionary\n  - Dining Out")
        assert "Coffee Shop" in text             # friendly payee name
        assert "SQ *COFFEE SHOP" in text         # raw descriptor in notes
        assert "-450" in text
        assert "-4.50" in text
        assert "Discretionary" in text

    def test_missing_imported_payee(self):
        txn = make_txn(imported_description=None)
        text = render_user_prompt(txn, "schema")
        assert "(none)" in text

    def test_missing_cleaned_payee(self):
        txn = make_txn(payee_name=None)
        text = render_user_prompt(txn, "schema")
        assert "(none)" in text

    def test_missing_notes(self):
        txn = make_txn(notes=None)
        text = render_user_prompt(txn, "schema")
        assert "(none)" in text

    def test_offbudget_account_label(self):
        txn = make_txn(account_name="Mortgage", offbudget=True)
        text = render_user_prompt(txn, "schema")
        assert "off-budget" in text

    def test_onbudget_account_label(self):
        txn = make_txn(account_name="Checking", offbudget=False)
        text = render_user_prompt(txn, "schema")
        assert "on-budget" in text


class TestRenderTransferPrompt:
    def test_both_transactions_present(self):
        txn_a = make_txn(imported_description="TRANSFER OUT", amount=-50000,
                         account_name="Checking", date="2026-06-01")
        txn_b = make_txn(imported_description="TRANSFER IN", amount=50000,
                         account_name="Savings", date="2026-06-02")
        text = render_transfer_prompt(txn_a, txn_b)
        assert "Checking" in text
        assert "Savings" in text
        assert "TRANSFER OUT" in text
        assert "TRANSFER IN" in text
        assert "-500.00" in text
        assert "+500.00" in text

    def test_missing_imported_payee_falls_back_to_payee_name(self):
        txn_a = make_txn(imported_description=None, payee_name="My Bank")
        txn_b = make_txn(imported_description=None, payee_name=None)
        text = render_transfer_prompt(txn_a, txn_b)
        assert "My Bank" in text
        assert "(none)" in text


class TestDuplicateSystemPrompt:
    """The two things the prompt exists to teach, learned from live runs."""

    def test_names_both_expected_false_positives(self):
        from actual_cat.prompts import DUPLICATE_SYSTEM

        assert "subscription" in DUPLICATE_SYSTEM
        assert "second visit" in DUPLICATE_SYSTEM.lower()

    def test_explains_that_a_posted_row_may_be_dated_a_day_earlier(self):
        # Without this, the model rejects true pairs on date order alone —
        # observed on a live DevBudget run before the guidance was added.
        from actual_cat.prompts import DUPLICATE_SYSTEM

        assert "earlier" in DUPLICATE_SYSTEM
        assert "authorization date" in DUPLICATE_SYSTEM

    def test_states_that_the_apply_action_is_deletion(self):
        from actual_cat.prompts import DUPLICATE_SYSTEM

        assert "deleting the pending row" in DUPLICATE_SYSTEM
