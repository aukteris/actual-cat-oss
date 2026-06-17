"""Unit tests for the source-neutral receipt validator/confidence scorer."""

import pytest

from actual_cat.receipts.parse import compute_confidence, validate_receipt


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


class TestValidateReceipt:
    def _good(self, **overrides):
        base = {
            "merchant": "Target",
            "date": "2026-06-01",
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

    def test_rounding_diff_nudged_and_high(self):
        data = self._good()
        data["line_items"][0]["amount_cents"] = 2999  # 1¢ short
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

    def test_null_amount_cents_allowed_low_confidence(self):
        data = self._good()
        data["line_items"][0]["amount_cents"] = None
        result = validate_receipt(data)
        assert result["confidence"] == "low"
        assert result["line_items"][0]["amount_cents"] is None

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

    def test_date_may_be_none(self):
        result = validate_receipt(self._good(date=None))
        assert result["date"] is None

    def test_category_defaults_to_uncertain_when_missing(self):
        data = self._good()
        del data["line_items"][0]["category"]
        result = validate_receipt(data)
        assert result["line_items"][0]["category"] == "Uncertain"

    def test_non_int_amount_coerced_to_none(self):
        data = self._good()
        data["line_items"][0]["amount_cents"] = "not-a-number"
        result = validate_receipt(data)
        assert result["line_items"][0]["amount_cents"] is None
        assert result["confidence"] == "low"
