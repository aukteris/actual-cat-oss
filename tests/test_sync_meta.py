"""Unit tests for raw_synced_data parsing and descriptor normalization."""

import json
from unittest.mock import MagicMock

from actual_cat.sync_meta import (
    bank_txn_id,
    descriptor_similarity,
    descriptor_text,
    descriptor_tokens,
    is_pending,
    parse_synced,
)


def make_txn(
    *,
    raw=None,
    cleared=1,
    financial_id=None,
    imported_description=None,
    payee_name=None,
) -> MagicMock:
    txn = MagicMock()
    txn.raw_synced_data = raw if raw is None or isinstance(raw, str) else json.dumps(raw)
    txn.cleared = cleared
    txn.financial_id = financial_id
    txn.imported_description = imported_description
    if payee_name is None:
        txn.payee = None
    else:
        txn.payee = MagicMock()
        txn.payee.name = payee_name
    return txn


class TestParseSynced:
    def test_absent_payload(self):
        assert parse_synced(make_txn()) is None

    def test_malformed_json(self):
        assert parse_synced(make_txn(raw="{not json")) is None

    def test_non_dict_payload(self):
        assert parse_synced(make_txn(raw="[1, 2]")) is None

    def test_valid_payload(self):
        payload = parse_synced(make_txn(raw={"booked": False, "amount": "-230.00"}))
        assert payload == {"booked": False, "amount": "-230.00"}


class TestIsPending:
    def test_never_pending_manual_row(self):
        # No payload at all: manual entry, starting balance, or split child.
        assert is_pending(make_txn(cleared=0)) is False

    def test_pending(self):
        assert is_pending(make_txn(raw={"booked": False}, cleared=0)) is True

    def test_posted_in_place_keeps_stale_booked_false(self):
        # booked is a snapshot from first import and never refreshed; cleared is
        # what says the bank finalized it. This is the trap the conjunction exists
        # for — 161 live rows in the reference budget look like this.
        assert is_pending(make_txn(raw={"booked": False}, cleared=1)) is False

    def test_booked_true(self):
        assert is_pending(make_txn(raw={"booked": True}, cleared=0)) is False

    def test_booked_key_absent(self):
        assert is_pending(make_txn(raw={"amount": "-1.00"}, cleared=0)) is False

    def test_malformed_payload_is_not_pending(self):
        assert is_pending(make_txn(raw="{not json", cleared=0)) is False

    def test_cleared_none_is_not_pending(self):
        assert is_pending(make_txn(raw={"booked": False}, cleared=None)) is False


class TestBankTxnId:
    def test_prefers_column(self):
        txn = make_txn(financial_id="FIN-1", raw={"transactionId": "TRN-2"})
        assert bank_txn_id(txn) == "FIN-1"

    def test_falls_back_to_payload(self):
        assert bank_txn_id(make_txn(raw={"transactionId": "TRN-2"})) == "TRN-2"

    def test_none_when_unknown(self):
        assert bank_txn_id(make_txn()) is None


class TestDescriptorText:
    def test_prefers_imported_description(self):
        txn = make_txn(imported_description="TST* CORNER CANTINA", payee_name="Corner Cantina")
        assert descriptor_text(txn) == "TST* CORNER CANTINA"

    def test_falls_back_to_payee_then_payload(self):
        assert descriptor_text(make_txn(payee_name="Corner Cantina")) == "Corner Cantina"
        assert descriptor_text(make_txn(raw={"payeeName": "Corner Cantina"})) == "Corner Cantina"

    def test_empty_when_nothing_available(self):
        assert descriptor_text(make_txn()) == ""


class TestDescriptorTokens:
    def test_drops_stopwords_and_bare_numbers(self):
        txn = make_txn(imported_description="TST* CORNER CANTINA 1234 ANYTOWN OR US")
        assert descriptor_tokens(txn) == frozenset({"corner", "cantina", "anytown"})


class TestDescriptorSimilarity:
    def _tokens(self, text):
        return descriptor_tokens(make_txn(imported_description=text))

    def test_posted_row_gains_city(self):
        assert descriptor_similarity(
            self._tokens("CORNER CANTINA"), self._tokens("CORNER CANTINA ANYTOWN")
        ) == 1.0

    def test_country_to_state_suffix_swap(self):
        assert descriptor_similarity(
            self._tokens("MERCHANT ANYTOWN US"), self._tokens("MERCHANT ANYTOWN OR")
        ) == 1.0

    def test_domain_suffix_swap(self):
        assert descriptor_similarity(
            self._tokens("Tumblewell"), self._tokens("Tumblewell.com")
        ) == 1.0

    def test_unrelated_merchants(self):
        assert descriptor_similarity(
            self._tokens("CORNER CANTINA"), self._tokens("NETFLIX")
        ) == 0.0

    def test_partial_overlap(self):
        assert descriptor_similarity(
            self._tokens("RAMEN HOUSE"), self._tokens("RAMEN GARDEN")
        ) == 0.5

    def test_empty_side_scores_zero(self):
        assert descriptor_similarity(frozenset(), self._tokens("CORNER CANTINA")) == 0.0
