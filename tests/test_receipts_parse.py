"""Unit tests for receipt parsing, validation, and date resolution."""


import pytest

from actual_cat.receipts.parse import (
    compute_confidence,
    drop_double_counted_savings,
    infer_country_code,
    resolve_receipt_date,
    validate_receipt,
)

# ---------------------------------------------------------------------------
# compute_confidence
# ---------------------------------------------------------------------------

class TestComputeConfidence:
    def test_exact_sum_high(self):
        items = [{"amount_cents": 100}, {"amount_cents": 50}]
        conf, diff = compute_confidence(items, 150)
        assert conf == "high"
        assert diff == 0

    def test_within_tolerance_high(self):
        items = [{"amount_cents": 100}, {"amount_cents": 49}]
        conf, diff = compute_confidence(items, 150)
        assert conf == "high"
        assert diff == 1

    def test_exceeds_tolerance_low(self):
        items = [{"amount_cents": 100}, {"amount_cents": 10}]
        conf, diff = compute_confidence(items, 150)
        assert conf == "low"
        assert diff == 40

    def test_null_amount_forces_low(self):
        items = [{"amount_cents": 100}, {"amount_cents": None}]
        conf, _ = compute_confidence(items, 100)
        assert conf == "low"

    def test_negative_amount_counted(self):
        items = [{"amount_cents": 200}, {"amount_cents": -50}]
        conf, diff = compute_confidence(items, 150)
        assert conf == "high"
        assert diff == 0


# ---------------------------------------------------------------------------
# validate_receipt
# ---------------------------------------------------------------------------

class TestValidateReceipt:
    def _good(self, **overrides):
        base = {
            "merchant": "Target",
            "date": "2026-06-01",
            "date_raw": "06/01/2026",
            "location_raw": "123 Main St, Seattle, WA",
            "total_cents": 4217,
            "line_items": [
                {"description": "Groceries", "amount_cents": 3000, "category": "Food / Groceries"},
                {"description": "Shampoo", "amount_cents": 1217,
                 "category": "Personal / Personal Care"},
            ],
        }
        base.update(overrides)
        return base

    def test_valid_high_confidence(self):
        result = validate_receipt(self._good())
        assert result["merchant"] == "Target"
        assert result["total_cents"] == 4217
        assert len(result["line_items"]) == 2
        assert result["confidence"] == "high"
        assert result["date_raw"] == "06/01/2026"
        assert result["location_raw"] == "123 Main St, Seattle, WA"

    def test_missing_date_raw_allowed(self):
        data = self._good()
        data["date_raw"] = None
        result = validate_receipt(data)
        assert result["date_raw"] is None
        assert result["confidence"] == "high"

    def test_missing_location_raw_allowed(self):
        data = self._good()
        data["location_raw"] = None
        result = validate_receipt(data)
        assert result["location_raw"] is None

    def test_null_amount_cents_allowed_low_confidence(self):
        data = self._good()
        data["line_items"][0]["amount_cents"] = None
        result = validate_receipt(data)
        assert result["line_items"][0]["amount_cents"] is None
        assert result["confidence"] == "low"

    def test_non_int_amount_coerced_to_none(self):
        data = self._good()
        data["line_items"][0]["amount_cents"] = "not-a-number"
        result = validate_receipt(data)
        assert result["line_items"][0]["amount_cents"] is None
        assert result["confidence"] == "low"

    def test_rounding_diff_nudged_and_high(self):
        data = self._good()
        data["line_items"][0]["amount_cents"] = 2999  # 1c short
        result = validate_receipt(data)
        total = sum(i["amount_cents"] for i in result["line_items"])
        assert total == 4217
        assert result["confidence"] == "high"

    def test_large_diff_low_confidence_untouched(self):
        data = self._good()
        data["line_items"][0]["amount_cents"] = 1000  # way off
        result = validate_receipt(data)
        assert result["confidence"] == "low"
        assert result["line_items"][0]["amount_cents"] == 1000

    def test_negative_amount_cents_allowed(self):
        data = self._good()
        data["line_items"].append(
            {"description": "Discount", "amount_cents": -100, "category": "General"}
        )
        data["total_cents"] = 4117
        result = validate_receipt(data)
        assert result["confidence"] == "high"
        neg = next(i for i in result["line_items"] if i["description"] == "Discount")
        assert neg["amount_cents"] == -100

    def test_error_key_raises(self):
        with pytest.raises(ValueError, match="timeout"):
            validate_receipt({"error": "LLM call failure: timeout"})

    def test_missing_merchant_raises(self):
        with pytest.raises(ValueError):
            validate_receipt(self._good(merchant=""))

    def test_zero_total_raises(self):
        with pytest.raises(ValueError):
            validate_receipt(self._good(total_cents=0))

    def test_empty_line_items_raises(self):
        with pytest.raises(ValueError):
            validate_receipt(self._good(line_items=[]))

    def test_category_defaults_to_uncertain_when_missing(self):
        data = self._good()
        del data["line_items"][0]["category"]
        result = validate_receipt(data)
        assert result["line_items"][0]["category"] == "Uncertain"

# ---------------------------------------------------------------------------
# infer_country_code
# ---------------------------------------------------------------------------

class TestInferCountryCode:
    def test_us_phone_number(self):
        location = "1900 SE Kirkland Way, Vancouver, WA 98683, +1-555-123-4567"
        assert infer_country_code(location) == "US"

    def test_international_phone_uk(self):
        location = "123 High Street, London, +44 20 7946 0958"
        assert infer_country_code(location) == "GB"

    def test_international_phone_ireland(self):
        location = "45 O'Connell Street, Dublin, +353 1 234 5678"
        assert infer_country_code(location) == "IE"

    def test_international_phone_japan(self):
        location = "1-2-3 Shibuya, Tokyo, +81 3-1234-5678"
        assert infer_country_code(location) == "JP"

    def test_international_phone_australia(self):
        location = "123 George Street, Sydney NSW, +61 2 9876 5432"
        assert infer_country_code(location) == "AU"

    def test_country_name_United_States(self):
        location = "Costco Wholesale, United States"
        assert infer_country_code(location) == "US"

    def test_country_name_United_Kingdom(self):
        location = "Tesco Extra, United Kingdom"
        assert infer_country_code(location) == "GB"

    def test_country_name_Ireland(self):
        location = "Dunnes Stores, Ireland"
        assert infer_country_code(location) == "IE"

    def test_country_name_Japan(self):
        location = "7-Eleven, Japan"
        assert infer_country_code(location) == "JP"

    def test_country_name_Australia(self):
        location = "Coles Supermarket, Australia"
        assert infer_country_code(location) == "AU"

    def test_country_name_Germany(self):
        location = "Aldi, Germany"
        assert infer_country_code(location) == "DE"

    def test_country_name_France(self):
        location = "Carrefour, France"
        assert infer_country_code(location) == "FR"

    def test_country_name_Canada(self):
        location = "Loblaws, Canada"
        assert infer_country_code(location) == "CA"

    def test_country_name_Mexico(self):
        location = "Walmart, Mexico"
        assert infer_country_code(location) == "MX"

    def test_empty_location(self):
        assert infer_country_code("") is None

    def test_none_location(self):
        assert infer_country_code(None) is None

# ---------------------------------------------------------------------------
# resolve_receipt_date
# ---------------------------------------------------------------------------

class TestResolveReceiptDate:
    _RECEIVED_TS = "2026-08-09T18:19:13"

    def test_us_format_mdy_with_us_location(self):
        """08/09/2026 with US location should resolve to August 9 (MDY)."""
        date_raw = "08/09/2026"
        location = "1900 SE Kirkland Way, Vancouver, WA 98683"
        iso_date, was_ambiguous = resolve_receipt_date(date_raw, location, self._RECEIVED_TS)
        assert iso_date == "2026-08-09"
        assert was_ambiguous is False

    def test_dmy_format_with_uk_location(self):
        """09/08/2026 with UK location should resolve to August 9 (DMY)."""
        date_raw = "09/08/2026"
        location = "123 High Street, London, United Kingdom"
        iso_date, was_ambiguous = resolve_receipt_date(date_raw, location, self._RECEIVED_TS)
        assert iso_date == "2026-08-09"
        assert was_ambiguous is False

    def test_ireland_location_dmy(self):
        """15/03/2026 with Ireland location should resolve to March 15 (DMY)."""
        date_raw = "15/03/2026"
        location = "45 O'Connell Street, Dublin, Ireland"
        iso_date, was_ambiguous = resolve_receipt_date(date_raw, location, "2026-03-15T10:00:00")
        assert iso_date == "2026-03-15"
        assert was_ambiguous is False

    def test_japan_location_ymd(self):
        """2026/08/09 with Japan location should resolve to August 9 (YMD)."""
        date_raw = "2026/08/09"
        location = "1-2-3 Shibuya, Tokyo, Japan"
        iso_date, was_ambiguous = resolve_receipt_date(date_raw, location, self._RECEIVED_TS)
        assert iso_date == "2026-08-09"
        assert was_ambiguous is False

    def test_australia_location_dmy(self):
        """25/12/2026 with Australia location should resolve to December 25 (DMY)."""
        date_raw = "25/12/2026"
        location = "123 George Street, Sydney NSW, Australia"
        iso_date, was_ambiguous = resolve_receipt_date(date_raw, location, "2026-12-25T10:00:00")
        assert iso_date == "2026-12-25"
        assert was_ambiguous is False

    def test_fallback_to_received_ts_when_no_location(self):
        """08/09/2026 with no location should use received_ts-nearest parse."""
        date_raw = "08/09/2026"
        iso_date, was_ambiguous = resolve_receipt_date(date_raw, None, self._RECEIVED_TS)
        # Without location, both MDY and DMY are possible; pick nearest to received_ts
        # received_ts is 2026-08-09, so 2026-08-09 (MDY) is closer than 2026-09-08 (DMY)
        assert iso_date == "2026-08-09"
        assert was_ambiguous is False

    def test_ambiguous_when_location_and_received_ts_disagree(self):
        """Location says DMY (Sept 8), but received_ts is Aug 9 -> ambiguous."""
        date_raw = "08/09/2026"
        location = "123 High Street, London, United Kingdom"  # DMY format
        # received_ts is Aug 9, but DMY would give Sept 8 (30 days away)
        iso_date, was_ambiguous = resolve_receipt_date(date_raw, location, self._RECEIVED_TS)
        # Cross-check triggers: DMY gives Sept 8, MDY gives Aug 9 (near received_ts)
        # They disagree, so we return None to trigger received_ts fallback
        assert iso_date is None
        assert was_ambiguous is True

    def test_none_date_raw_returns_none(self):
        iso_date, was_ambiguous = resolve_receipt_date(None, "US location", self._RECEIVED_TS)
        assert iso_date is None
        assert was_ambiguous is False

    def test_invalid_date_format_returns_none(self):
        iso_date, was_ambiguous = resolve_receipt_date(
            "not-a-date", "US location", self._RECEIVED_TS
        )
        assert iso_date is None
        assert was_ambiguous is False

    def test_date_far_from_received_ts_filtered_out(self):
        """Date 2 years away should be filtered out even if format matches."""
        date_raw = "08/09/2024"  # 2 years in the past
        location = "1900 SE Kirkland Way, Vancouver, WA 98683"
        iso_date, was_ambiguous = resolve_receipt_date(date_raw, location, self._RECEIVED_TS)
        # Should fall back to received_ts path since location-derived date is too far
        assert iso_date is None or was_ambiguous is True

    def test_canada_location_uses_mdy(self):
        """Canada uses MDY like US."""
        date_raw = "08/09/2026"
        location = "123 Yonge Street, Toronto, ON, Canada"
        iso_date, was_ambiguous = resolve_receipt_date(date_raw, location, self._RECEIVED_TS)
        assert iso_date == "2026-08-09"
        assert was_ambiguous is False

    def test_mexico_location_uses_dmy(self):
        """Mexico uses DMY."""
        date_raw = "09/08/2026"
        location = "Av. Reforma 123, Mexico City, Mexico"
        iso_date, was_ambiguous = resolve_receipt_date(date_raw, location, self._RECEIVED_TS)
        assert iso_date == "2026-08-09"
        assert was_ambiguous is False


# ---------------------------------------------------------------------------
# Integration: validate_receipt + resolve_receipt_date
# ---------------------------------------------------------------------------

class TestValidateAndResolve:
    _RECEIVED_TS = "2026-08-09T18:19:13"

    def test_full_pipeline_us_receipt(self):
        """US receipt with date_raw and location_raw flows through correctly."""
        raw = {
            "merchant": "COSTCO WHOLESALE",
            "date_raw": "08/09/2026",
            "location_raw": "1900 SE Kirkland Way, Vancouver, WA 98683",
            "total_cents": 63115,
            "line_items": [
                {"description": "ITEM 1", "amount_cents": 30000, "category": "Food"},
                {"description": "ITEM 2", "amount_cents": 33115, "category": "Food"},
            ],
        }
        validated = validate_receipt(raw)
        assert validated["date_raw"] == "08/09/2026"
        assert validated["location_raw"] == "1900 SE Kirkland Way, Vancouver, WA 98683"

        # Resolve the date
        iso_date, was_ambiguous = resolve_receipt_date(
            validated["date_raw"], validated["location_raw"], self._RECEIVED_TS
        )
        assert iso_date == "2026-08-09"
        assert was_ambiguous is False

    def test_full_pipeline_uk_receipt(self):
        """UK receipt with DMY date format."""
        raw = {
            "merchant": "TESCO",
            "date_raw": "09/08/2026",
            "location_raw": "123 High Street, London, United Kingdom",
            "total_cents": 5000,
            "line_items": [
                {"description": "Bread", "amount_cents": 2000, "category": "Food"},
                {"description": "Milk", "amount_cents": 3000, "category": "Food"},
            ],
        }
        validated = validate_receipt(raw)
        iso_date, was_ambiguous = resolve_receipt_date(
            validated["date_raw"], validated["location_raw"], "2026-08-09T10:00:00"
        )
        assert iso_date == "2026-08-09"
        assert was_ambiguous is False


class TestDropDoubleCountedSavings:
    """A two-column receipt prints an already-discounted price AND separate savings
    rows. Emitting both subtracts every discount twice — the real failure that made
    a $57.64 receipt read as $51.64 and a $218.94 one as $63.10."""

    def _items(self, pairs):
        return [{"description": d, "amount_cents": a, "category": "x"} for d, a in pairs]

    def test_drops_savings_rows_that_make_the_total_reconcile(self):
        items = self._items([
            ("GHRDL DK & SLT", 749), ("Member Savings", -50),
            ("GHIRARDELLI MILK C", 749), ("Member Savings", -150),
            ("PLSH FD 7IN BEAR", 869), ("BOOKS", 599),
            ("FORT GEORGE SHERMA", 1299), ("DEP BEER 6 PK NFS", 60),
            ("Member Savings", -200),
            ("PORTLAND CIDER LMT", 1399), ("DEP BEER 4 PK NFS", 40),
            ("Member Savings", -200),
        ])
        kept, dropped = drop_double_counted_savings(items, 5764)
        assert dropped is True
        assert len(kept) == 8
        assert sum(i["amount_cents"] for i in kept) == 5764
        assert not any("Savings" in i["description"] for i in kept)

    def test_leaves_genuine_discount_rows_alone(self):
        """Costco prints one price column, so its discount rows are real. Removing
        them moves the sum AWAY from the total, so they must survive."""
        items = self._items([
            ("BIG ITEM", 10000), ("INSTANT SAVINGS", -1500), ("SMALL ITEM", 500),
        ])
        kept, dropped = drop_double_counted_savings(items, 9000)
        assert dropped is False
        assert len(kept) == 3

    def test_no_change_when_items_already_reconcile(self):
        items = self._items([("A", 500), ("Member Savings", -100)])
        kept, dropped = drop_double_counted_savings(items, 400)
        assert dropped is False
        assert len(kept) == 2

    def test_no_change_when_dropping_would_overshoot(self):
        """Only an exact reconciliation justifies deleting data."""
        items = self._items([("A", 500), ("Member Savings", -100), ("B", 300)])
        kept, dropped = drop_double_counted_savings(items, 9999)
        assert dropped is False
        assert len(kept) == 3

    def test_unreadable_amount_disables_the_repair(self):
        items = [
            {"description": "A", "amount_cents": 500, "category": "x"},
            {"description": "Member Savings", "amount_cents": None, "category": "x"},
        ]
        kept, dropped = drop_double_counted_savings(items, 500)
        assert dropped is False

    def test_validate_receipt_applies_the_repair_and_scores_high(self):
        data = {
            "merchant": "SAFEWAY", "total_cents": 5764,
            "line_items": [
                {"description": "GHRDL DK & SLT", "amount_cents": 749},
                {"description": "Member Savings", "amount_cents": -50},
                {"description": "GHIRARDELLI MILK C", "amount_cents": 749},
                {"description": "Member Savings", "amount_cents": -150},
                {"description": "PLSH FD 7IN BEAR", "amount_cents": 869},
                {"description": "BOOKS", "amount_cents": 599},
                {"description": "FORT GEORGE SHERMA", "amount_cents": 1299},
                {"description": "DEP BEER 6 PK NFS", "amount_cents": 60},
                {"description": "Member Savings", "amount_cents": -200},
                {"description": "PORTLAND CIDER LMT", "amount_cents": 1399},
                {"description": "DEP BEER 4 PK NFS", "amount_cents": 40},
                {"description": "Member Savings", "amount_cents": -200},
            ],
        }
        result = validate_receipt(data)
        assert result["confidence"] == "high"
        assert result["dropped_double_counted_savings"] is True
        assert len(result["line_items"]) == 8
