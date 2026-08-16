"""Unit tests for receipt OCR parsing and confidence scoring."""


import io
import time

import pytest

from actual_cat.receipts.ocr import (
    _ROTATIONS,
    _compute_confidence,
    _validate_response,
    ocr_receipt,
)


class MockLLM:
    def __init__(self, response: dict, *, sleep_seconds: float = 0.0):
        self._response = response
        self._sleep_seconds = sleep_seconds
        self.calls: list[tuple] = []

    def complete_json_vision(
        self,
        system: str,
        user: str,
        image_b64: str,
        media_type: str,
        *,
        timeout: float | None = None,
    ) -> dict:
        self.calls.append((system, user, image_b64, media_type, timeout))
        if self._sleep_seconds:
            time.sleep(self._sleep_seconds)
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


# ---------------------------------------------------------------------------
# Rotation gate + time bounds
#
# These use a real (Pillow-openable) image, unlike the tests above — the
# rotation path is only reachable when Pillow can actually rotate the input.
# ---------------------------------------------------------------------------

class TestRotationGate:
    _SCHEMA = "Food\n  - Groceries"
    _SYSTEM = "You are a receipt parser."

    def _real_jpeg(self) -> bytes:
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (40, 60), "white").save(buf, format="JPEG")
        return buf.getvalue()

    def _response(self, items: list[dict], total_cents: int) -> dict:
        return {
            "merchant": "Shop",
            "date": "2026-06-01",
            "total_cents": total_cents,
            "line_items": items,
        }

    def test_legible_read_that_does_not_reconcile_skips_rotation(self):
        """The regression test for the incident that motivated the gate.

        A long receipt the model clearly read — many line items, almost all
        amounts recovered — that simply doesn't add up. Rotating cannot help,
        and each extra pass costs ~100s against a real vision model.
        """
        items = [
            {"description": f"item {i}", "amount_cents": 100, "category": "Food / Groceries"}
            for i in range(9)
        ]
        items.append({"description": "smudged", "amount_cents": None, "category": "Uncertain"})
        llm = MockLLM(self._response(items, 5000))  # sum 900 vs total 5000
        result = ocr_receipt(self._real_jpeg(), llm, self._SYSTEM, self._SCHEMA)
        assert result["confidence"] == "low"
        assert result["unreadable_count"] == 1
        assert len(llm.calls) == 1

    def test_too_few_line_items_rotates(self):
        items = [{"description": "x", "amount_cents": 100, "category": "Food / Groceries"}]
        llm = MockLLM(self._response(items, 5000))
        ocr_receipt(self._real_jpeg(), llm, self._SYSTEM, self._SCHEMA)
        assert len(llm.calls) == 1 + len(_ROTATIONS)

    def test_mostly_unreadable_amounts_rotates(self):
        items = [{"description": "x", "amount_cents": 100, "category": "Food / Groceries"}]
        items += [
            {"description": f"blur {i}", "amount_cents": None, "category": "Uncertain"}
            for i in range(3)
        ]
        llm = MockLLM(self._response(items, 5000))  # 3 of 4 unreadable
        ocr_receipt(self._real_jpeg(), llm, self._SYSTEM, self._SCHEMA)
        assert len(llm.calls) == 1 + len(_ROTATIONS)

    def test_structural_failure_rotates(self):
        llm = MockLLM({"merchant": "", "total_cents": 100, "line_items": []})
        result = ocr_receipt(self._real_jpeg(), llm, self._SYSTEM, self._SCHEMA)
        assert "error" in result
        assert len(llm.calls) == 1 + len(_ROTATIONS)

    def test_budget_stops_rotation_early(self):
        items = [{"description": "x", "amount_cents": 100, "category": "Food / Groceries"}]
        llm = MockLLM(self._response(items, 5000), sleep_seconds=0.1)
        ocr_receipt(
            self._real_jpeg(), llm, self._SYSTEM, self._SCHEMA, budget_seconds=0.15,
        )
        # Gate is open (1 item), but the budget runs out before all rotations.
        assert 1 <= len(llm.calls) < 1 + len(_ROTATIONS)

    def test_request_timeout_clamped_to_remaining_budget(self):
        items = [{"description": "x", "amount_cents": 100, "category": "Food / Groceries"}]
        llm = MockLLM(self._response(items, 100))  # high confidence, single pass
        ocr_receipt(
            self._real_jpeg(), llm, self._SYSTEM, self._SCHEMA,
            request_timeout_seconds=300, budget_seconds=30,
        )
        timeout = llm.calls[0][4]
        assert timeout is not None and timeout <= 30

    def test_bounds_disabled_when_none(self):
        items = [{"description": "x", "amount_cents": 100, "category": "Food / Groceries"}]
        llm = MockLLM(self._response(items, 100))
        ocr_receipt(
            self._real_jpeg(), llm, self._SYSTEM, self._SCHEMA,
            request_timeout_seconds=None, budget_seconds=None,
        )
        assert llm.calls[0][4] is None


# ---------------------------------------------------------------------------
# Auto-crop
#
# A receipt photographed on a table or floor can be a narrow strip in a big
# frame; the vision server clamps every image to a fixed token budget, so the
# background eats most of it. Detection must be conservative — a crop that clips
# the total is worse than the wasted tokens.
# ---------------------------------------------------------------------------

class TestAutocrop:
    def _scene(self, frame=(400, 600), rect=(150, 40, 250, 560), bg=30, fg=245):
        from PIL import Image, ImageDraw
        img = Image.new("RGB", frame, (bg, bg, bg))
        ImageDraw.Draw(img).rectangle(rect, fill=(fg, fg, fg))
        return img

    def test_finds_a_narrow_bright_strip(self):
        from actual_cat.receipts.ocr import _receipt_bbox
        box = _receipt_bbox(self._scene())
        assert box is not None
        x0, y0, x1, y1 = box
        # Within padding of the drawn rectangle (150, 40, 250, 560)
        assert abs(x0 - 150) < 25 and abs(x1 - 250) < 25
        assert abs(y0 - 40) < 25 and abs(y1 - 560) < 25

    def test_full_height_strip_is_not_collapsed_vertically(self):
        """A narrow strip never clears a full-width row threshold, so rows must be
        judged inside the column band — otherwise the box collapses vertically."""
        from actual_cat.receipts.ocr import _receipt_bbox
        box = _receipt_bbox(self._scene(frame=(800, 2000), rect=(360, 20, 440, 1980)))
        assert box is not None
        assert (box[3] - box[1]) > 1800  # nearly the full height, not a fragment

    def test_speck_does_not_blow_up_the_box(self):
        from PIL import ImageDraw

        from actual_cat.receipts.ocr import _receipt_bbox
        img = self._scene()
        ImageDraw.Draw(img).rectangle((5, 5, 9, 9), fill=(255, 255, 255))  # debris
        box = _receipt_bbox(img)
        assert box is not None
        assert box[0] > 100  # the speck at x=5 was not absorbed

    def test_mostly_bright_frame_falls_back(self):
        """Receipt on a white counter — nothing to gain, don't risk clipping."""
        from actual_cat.receipts.ocr import _receipt_bbox
        assert _receipt_bbox(self._scene(rect=(2, 2, 398, 598))) is None

    def test_uniform_frame_falls_back(self):
        from PIL import Image

        from actual_cat.receipts.ocr import _receipt_bbox
        assert _receipt_bbox(Image.new("RGB", (400, 600), (200, 200, 200))) is None

    def test_autocrop_wrapper_reports_fraction(self):
        from actual_cat.receipts.ocr import _autocrop_receipt
        cropped, fraction = _autocrop_receipt(self._scene())
        assert fraction is not None and 0.0 < fraction < 1.0
        assert cropped.size[0] < 400

    def test_autocrop_never_raises(self):
        from actual_cat.receipts.ocr import _autocrop_receipt
        img, fraction = _autocrop_receipt(object())  # not an image at all
        assert fraction is None

    def test_normalize_image_honours_the_flag(self):
        import io as _io

        from actual_cat.receipts.ocr import _normalize_image
        buf = _io.BytesIO()
        self._scene().save(buf, format="JPEG", quality=95)
        raw = buf.getvalue()
        _, on, frac_on = _normalize_image(raw, autocrop=True)
        _, off, frac_off = _normalize_image(raw, autocrop=False)
        assert frac_on is not None and frac_off is None
        assert on.size[0] < off.size[0]

    def test_unopenable_bytes_fall_back(self):
        from actual_cat.receipts.ocr import _normalize_image
        data, img, frac = _normalize_image(b"\xff\xd8\xff" + b"\x00" * 10)
        assert data == b"\xff\xd8\xff" + b"\x00" * 10
        assert img is None and frac is None


class TestCropFallbackPass:
    _SCHEMA = "Food\n  - Groceries"
    _SYSTEM = "You are a receipt parser."

    def _scene_bytes(self) -> bytes:
        import io as _io

        from PIL import Image, ImageDraw
        img = Image.new("RGB", (400, 600), (30, 30, 30))
        ImageDraw.Draw(img).rectangle((150, 40, 250, 560), fill=(245, 245, 245))
        buf = _io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        return buf.getvalue()

    def _good(self, n=6):
        return {"merchant": "Shop", "date": "2026-06-01", "total_cents": n * 100,
                "line_items": [{"description": f"i{i}", "amount_cents": 100,
                                "category": "Food / Groceries"} for i in range(n)]}

    def test_any_imperfect_cropped_read_is_checked_against_the_full_frame(self):
        """A misplaced crop returns a coherent read of the part it kept — nothing
        about it looks wrong. Verifying against the full frame is the only catch."""
        class Seq(MockLLM):
            def __init__(self, responses):
                super().__init__({})
                self._responses = list(responses)
            def complete_json_vision(self, s, u, b, m, *, timeout=None):
                self.calls.append((s, u, b, m, timeout))
                return self._responses.pop(0)

        # Coherent, all amounts readable, simply doesn't reconcile — exactly what
        # a crop that clipped half the receipt produces.
        clipped = {"merchant": "Shop", "total_cents": 5000,
                   "line_items": [{"description": f"i{i}", "amount_cents": 100}
                                  for i in range(6)]}
        llm = Seq([clipped, self._good()])
        result = ocr_receipt(self._scene_bytes(), llm, self._SYSTEM, self._SCHEMA)
        assert len(llm.calls) == 2          # cropped, then uncropped — no rotations
        assert result["confidence"] == "high"
        assert result["cropped"] is False   # the winning read was the full frame

    def test_good_cropped_read_does_not_retry(self):
        llm = MockLLM(self._good())
        result = ocr_receipt(self._scene_bytes(), llm, self._SYSTEM, self._SCHEMA)
        assert len(llm.calls) == 1
        assert result["cropped"] is True
        assert result["crop_fraction"] is not None

    def test_cropped_read_wins_when_it_reconciles_better(self):
        """The full-frame check must not override a good crop — on the receipt
        that motivated cropping, the cropped read is the correct one."""
        class Seq(MockLLM):
            def __init__(self, responses):
                super().__init__({})
                self._responses = list(responses)
            def complete_json_vision(self, s, u, b, m, *, timeout=None):
                self.calls.append((s, u, b, m, timeout))
                return self._responses.pop(0)

        # Both low confidence, but the cropped read reconciles far closer to its
        # own total ($1 out) than the full-frame read does ($10 out).
        near = {"merchant": "Shop", "total_cents": 700,
                "line_items": [{"description": f"i{i}", "amount_cents": 100} for i in range(6)]}
        far = {"merchant": "Shop", "total_cents": 1600,
               "line_items": [{"description": f"i{i}", "amount_cents": 100} for i in range(6)]}
        llm = Seq([near, far])
        result = ocr_receipt(self._scene_bytes(), llm, self._SYSTEM, self._SCHEMA)
        assert len(llm.calls) == 2
        assert result["cropped"] is True          # the crop was kept
        assert result["total_cents"] == 700

    def test_no_full_frame_check_when_cropping_is_off(self):
        clipped = {"merchant": "Shop", "total_cents": 5000,
                   "line_items": [{"description": f"i{i}", "amount_cents": 100}
                                  for i in range(6)]}
        llm = MockLLM(clipped)
        ocr_receipt(self._scene_bytes(), llm, self._SYSTEM, self._SCHEMA, autocrop=False)
        assert len(llm.calls) == 1  # nothing to second-guess
