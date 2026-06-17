"""Unit tests for receipt OCR parsing and confidence scoring."""


import pytest

from actual_cat.receipts.ocr import _compute_confidence, _validate_response, ocr_receipt


class MockLLM:
    def __init__(self, response: dict):
        self._response = response
        self.calls: list[tuple] = []

    def complete_json_vision(self, system: str, user: str, image_b64: str, media_type: str) -> dict:
        self.calls.append((system, user, image_b64, media_type))
        return self._response


# ---------------------------------------------------------------------------
# _compute_confidence
# ---------------------------------------------------------------------------

class TestComputeConfidence:
    def test_exact_sum_high(self):
        items = [{"amount_cents": 100}, {"amount_cents": 50}]
        conf, diff = _compute_confidence(items, 150)
        assert conf == "high"
        assert diff == 0

    def test_within_tolerance_high(self):
        items = [{"amount_cents": 100}, {"amount_cents": 49}]
        conf, diff = _compute_confidence(items, 150)
        assert conf == "high"
        assert diff == 1

    def test_exceeds_tolerance_low(self):
        items = [{"amount_cents": 100}, {"amount_cents": 10}]
        conf, diff = _compute_confidence(items, 150)
        assert conf == "low"
        assert diff == 40

    def test_null_amount_forces_low(self):
        items = [{"amount_cents": 100}, {"amount_cents": None}]
        conf, _ = _compute_confidence(items, 100)
        assert conf == "low"

    def test_negative_amount_counted(self):
        items = [{"amount_cents": 200}, {"amount_cents": -50}]
        conf, diff = _compute_confidence(items, 150)
        assert conf == "high"
        assert diff == 0


# ---------------------------------------------------------------------------
# _validate_response
# ---------------------------------------------------------------------------

class TestValidateResponse:
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
        result = _validate_response(self._good())
        assert result["merchant"] == "Target"
        assert result["total_cents"] == 4217
        assert len(result["line_items"]) == 2
        assert result["confidence"] == "high"

    def test_rounding_diff_nudged_and_high(self):
        data = self._good()
        data["line_items"][0]["amount_cents"] = 2999  # 1¢ short
        result = _validate_response(data)
        total = sum(i["amount_cents"] for i in result["line_items"])
        assert total == 4217
        assert result["confidence"] == "high"

    def test_large_diff_low_confidence_untouched(self):
        data = self._good()
        data["line_items"][0]["amount_cents"] = 1000  # way off
        result = _validate_response(data)
        assert result["confidence"] == "low"
        # amounts left untouched — not fabricated
        assert result["line_items"][0]["amount_cents"] == 1000

    def test_null_amount_cents_allowed_low_confidence(self):
        data = self._good()
        data["line_items"][0]["amount_cents"] = None
        result = _validate_response(data)
        assert result["confidence"] == "low"
        assert result["line_items"][0]["amount_cents"] is None

    def test_negative_amount_cents_allowed(self):
        data = self._good()
        data["line_items"].append(
            {"description": "Discount", "amount_cents": -100, "category": "General"}
        )
        data["total_cents"] = 4117
        result = _validate_response(data)
        assert result["confidence"] == "high"
        neg = next(i for i in result["line_items"] if i["description"] == "Discount")
        assert neg["amount_cents"] == -100

    def test_error_key_raises(self):
        with pytest.raises(ValueError, match="timeout"):
            _validate_response({"error": "LLM call failure: timeout"})

    def test_missing_merchant_raises(self):
        with pytest.raises(ValueError):
            _validate_response(self._good(merchant=""))

    def test_zero_total_raises(self):
        with pytest.raises(ValueError):
            _validate_response(self._good(total_cents=0))

    def test_negative_total_raises(self):
        with pytest.raises(ValueError):
            _validate_response(self._good(total_cents=-100))

    def test_empty_line_items_raises(self):
        with pytest.raises(ValueError):
            _validate_response(self._good(line_items=[]))

    def test_missing_line_items_raises(self):
        data = self._good()
        del data["line_items"]
        with pytest.raises(ValueError):
            _validate_response(data)

    def test_date_may_be_none(self):
        result = _validate_response(self._good(date=None))
        assert result["date"] is None

    def test_category_defaults_to_uncertain_when_missing(self):
        data = self._good()
        del data["line_items"][0]["category"]
        result = _validate_response(data)
        assert result["line_items"][0]["category"] == "Uncertain"

    def test_non_int_amount_coerced_to_none(self):
        data = self._good()
        data["line_items"][0]["amount_cents"] = "not-a-number"
        result = _validate_response(data)
        assert result["line_items"][0]["amount_cents"] is None
        assert result["confidence"] == "low"


# ---------------------------------------------------------------------------
# ocr_receipt
# ---------------------------------------------------------------------------

class TestOcrReceipt:
    _SCHEMA = "Food\n  - Groceries"
    _SYSTEM = "You are a receipt parser."

    def _jpeg_bytes(self) -> bytes:
        # Minimal JPEG-magic bytes — Pillow will fail to open, so no rotation
        # retry is attempted. Tests remain fast and single-call.
        return b"\xff\xd8\xff" + b"\x00" * 10

    def _png_bytes(self) -> bytes:
        return b"\x89PNG" + b"\x00" * 10

    def test_happy_path_returns_structured_data_with_confidence(self):
        llm = MockLLM({
            "merchant": "Trader Joe's",
            "date": "2026-06-01",
            "total_cents": 2000,
            "line_items": [
                {"description": "Produce", "amount_cents": 2000, "category": "Food / Groceries"}
            ],
        })
        result = ocr_receipt(self._jpeg_bytes(), llm, self._SYSTEM, self._SCHEMA)
        assert result["merchant"] == "Trader Joe's"
        assert result["total_cents"] == 2000
        assert result["confidence"] == "high"

    def test_llm_error_returned_as_error_dict(self):
        llm = MockLLM({"error": "LLM call failure: timeout"})
        result = ocr_receipt(self._jpeg_bytes(), llm, self._SYSTEM, self._SCHEMA)
        assert "error" in result

    def test_jpeg_media_type_detected(self):
        llm = MockLLM({
            "merchant": "Shop", "date": "2026-06-01", "total_cents": 100,
            "line_items": [
                {"description": "x", "amount_cents": 100, "category": "Food / Groceries"}
            ],
        })
        ocr_receipt(self._jpeg_bytes(), llm, self._SYSTEM, self._SCHEMA)
        assert llm.calls[0][3] == "image/jpeg"

    def test_png_media_type_detected(self):
        llm = MockLLM({
            "merchant": "Shop", "date": "2026-06-01", "total_cents": 100,
            "line_items": [
                {"description": "x", "amount_cents": 100, "category": "Food / Groceries"}
            ],
        })
        ocr_receipt(self._png_bytes(), llm, self._SYSTEM, self._SCHEMA)
        # Minimal bytes can't be opened by Pillow, so raw bytes are passed through
        # and the media type reflects the original magic bytes sniff.
        assert llm.calls[0][3] == "image/png"

    def test_schema_text_included_in_user_message(self):
        schema = "Unique schema marker XYZ"
        llm = MockLLM({
            "merchant": "Shop", "date": "2026-06-01", "total_cents": 100,
            "line_items": [
                {"description": "x", "amount_cents": 100, "category": "Food / Groceries"}
            ],
        })
        ocr_receipt(self._jpeg_bytes(), llm, self._SYSTEM, schema)
        user_msg = llm.calls[0][1]
        assert "Unique schema marker XYZ" in user_msg

    def test_high_confidence_no_rotation_retry(self):
        llm = MockLLM({
            "merchant": "Shop", "date": "2026-06-01", "total_cents": 100,
            "line_items": [
                {"description": "x", "amount_cents": 100, "category": "Food / Groceries"}
            ],
        })
        ocr_receipt(self._jpeg_bytes(), llm, self._SYSTEM, self._SCHEMA)
        # High confidence → only 1 LLM call (no rotation)
        assert len(llm.calls) == 1
