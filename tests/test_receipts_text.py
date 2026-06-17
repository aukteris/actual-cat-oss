"""Unit tests for plain-text receipt parsing."""

from actual_cat.receipts.text import parse_receipt_text


class MockLLM:
    def __init__(self, response: dict):
        self._response = response
        self.calls: list[tuple] = []

    def complete_json(self, system: str, user: str) -> dict:
        self.calls.append((system, user))
        return self._response


_SCHEMA = "Food\n  - Groceries"
_SYSTEM = "You are a receipt text parser."
_TEXT = "TRADER JOE'S\nProduce  20.00\nTotal  20.00"


def test_happy_path_returns_structured_data_with_confidence():
    llm = MockLLM({
        "merchant": "Trader Joe's",
        "date": "2026-06-01",
        "total_cents": 2000,
        "line_items": [
            {"description": "Produce", "amount_cents": 2000, "category": "Food / Groceries"}
        ],
    })
    result = parse_receipt_text(_TEXT, llm, _SYSTEM, _SCHEMA)
    assert result["merchant"] == "Trader Joe's"
    assert result["total_cents"] == 2000
    assert result["confidence"] == "high"


def test_uses_text_llm_path_not_vision():
    llm = MockLLM({
        "merchant": "Shop", "date": "2026-06-01", "total_cents": 100,
        "line_items": [{"description": "x", "amount_cents": 100, "category": "Food / Groceries"}],
    })
    parse_receipt_text(_TEXT, llm, _SYSTEM, _SCHEMA)
    assert len(llm.calls) == 1
    assert llm.calls[0][0] == _SYSTEM


def test_llm_error_returned_as_error_dict():
    llm = MockLLM({"error": "LLM call failure: timeout"})
    result = parse_receipt_text(_TEXT, llm, _SYSTEM, _SCHEMA)
    assert "error" in result


def test_schema_and_text_included_in_user_message():
    llm = MockLLM({
        "merchant": "Shop", "date": "2026-06-01", "total_cents": 100,
        "line_items": [{"description": "x", "amount_cents": 100, "category": "Food / Groceries"}],
    })
    parse_receipt_text(_TEXT, llm, _SYSTEM, "Unique schema marker XYZ")
    user_msg = llm.calls[0][1]
    assert "Unique schema marker XYZ" in user_msg
    assert "TRADER JOE'S" in user_msg


def test_low_confidence_when_sum_mismatches():
    llm = MockLLM({
        "merchant": "Shop", "date": "2026-06-01", "total_cents": 5000,
        "line_items": [{"description": "x", "amount_cents": 100, "category": "Food / Groceries"}],
    })
    result = parse_receipt_text(_TEXT, llm, _SYSTEM, _SCHEMA)
    assert result["confidence"] == "low"
