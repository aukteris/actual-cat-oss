"""Unit tests for the receipt extractor dispatch registry."""

from types import SimpleNamespace

from actual_cat.receipts.extract import extract_receipt

_PROMPTS = SimpleNamespace(
    RECEIPT_OCR_SYSTEM="ocr-system",
    RECEIPT_TEXT_SYSTEM="text-system",
)

_GOOD_RESPONSE = {
    "merchant": "Shop",
    "date": "2026-06-01",
    "total_cents": 100,
    "line_items": [{"description": "x", "amount_cents": 100, "category": "Food / Groceries"}],
}


class MockLLM:
    def __init__(self, response: dict):
        self._response = response
        self.text_calls: list[tuple] = []
        self.vision_calls: list[tuple] = []

    def complete_json(self, system: str, user: str) -> dict:
        self.text_calls.append((system, user))
        return self._response

    def complete_json_vision(self, system, user, image_b64, media_type) -> dict:
        self.vision_calls.append((system, user, image_b64, media_type))
        return self._response


def test_text_kind_dispatches_to_text_parser():
    llm = MockLLM(_GOOD_RESPONSE)
    meta = {"input_kind": "text", "text": "TRADER JOE'S\nProduce 1.00\nTotal 1.00"}
    result = extract_receipt(meta, llm, _PROMPTS, "schema")
    assert result["merchant"] == "Shop"
    assert len(llm.text_calls) == 1
    assert len(llm.vision_calls) == 0
    assert llm.text_calls[0][0] == "text-system"


def test_image_kind_dispatches_to_ocr(tmp_path):
    image_path = tmp_path / "r.jpg"
    image_path.write_bytes(b"\xff\xd8\xff" + b"\x00" * 10)  # minimal JPEG magic
    llm = MockLLM(_GOOD_RESPONSE)
    meta = {"input_kind": "image", "image_path": str(image_path)}
    result = extract_receipt(meta, llm, _PROMPTS, "schema")
    assert result["merchant"] == "Shop"
    assert len(llm.vision_calls) == 1
    assert len(llm.text_calls) == 0
    assert llm.vision_calls[0][0] == "ocr-system"


def test_missing_input_kind_defaults_to_image(tmp_path):
    image_path = tmp_path / "r.jpg"
    image_path.write_bytes(b"\xff\xd8\xff" + b"\x00" * 10)
    llm = MockLLM(_GOOD_RESPONSE)
    meta = {"image_path": str(image_path)}  # legacy meta, no input_kind
    result = extract_receipt(meta, llm, _PROMPTS, "schema")
    assert result["merchant"] == "Shop"
    assert len(llm.vision_calls) == 1


def test_unknown_kind_returns_error():
    llm = MockLLM(_GOOD_RESPONSE)
    meta = {"input_kind": "pdf", "payload": "..."}
    result = extract_receipt(meta, llm, _PROMPTS, "schema")
    assert "error" in result
    assert "pdf" in result["error"]


def test_text_kind_missing_text_returns_error():
    llm = MockLLM(_GOOD_RESPONSE)
    result = extract_receipt({"input_kind": "text", "text": "   "}, llm, _PROMPTS, "schema")
    assert "error" in result


def test_image_kind_missing_path_returns_error():
    llm = MockLLM(_GOOD_RESPONSE)
    result = extract_receipt({"input_kind": "image"}, llm, _PROMPTS, "schema")
    assert "error" in result
