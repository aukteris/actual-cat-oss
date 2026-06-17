"""Unit tests for receipt pass-2 line-item categorization."""

from collections import Counter
from types import SimpleNamespace

from actual_cat.receipts.categorize import categorize_line_items

_PROMPTS = SimpleNamespace(RECEIPT_CATEGORIZE_SYSTEM="categorize-system")


class MockLLM:
    def __init__(self, response: dict):
        self._response = response
        self.calls: list[tuple[str, str]] = []

    def complete_json(self, system: str, user: str) -> dict:
        self.calls.append((system, user))
        return self._response


def _cfg(enabled=True):
    return SimpleNamespace(
        history_enabled=enabled, history_item_top_n=3, history_min_count=1
    )


def _receipt(*descriptions):
    return {
        "merchant": "Test Store",
        "total_cents": 100 * len(descriptions),
        "line_items": [
            {"description": d, "amount_cents": 100, "category": "Uncertain"}
            for d in descriptions
        ],
    }


def test_categories_written_back_in_order():
    receipt = _receipt("Bananas", "Shampoo")
    llm = MockLLM({"categories": ["Food / Groceries", "Personal / Care"]})
    result = categorize_line_items(receipt, llm, _PROMPTS, "schema", {}, _cfg())
    assert result["line_items"][0]["category"] == "Food / Groceries"
    assert result["line_items"][1]["category"] == "Personal / Care"


def test_count_mismatch_leaves_uncertain():
    receipt = _receipt("Bananas", "Shampoo")
    llm = MockLLM({"categories": ["Food / Groceries"]})  # too few
    result = categorize_line_items(receipt, llm, _PROMPTS, "schema", {}, _cfg())
    assert all(i["category"] == "Uncertain" for i in result["line_items"])


def test_error_response_leaves_uncertain():
    receipt = _receipt("Bananas")
    llm = MockLLM({"error": "LLM call failure: timeout"})
    result = categorize_line_items(receipt, llm, _PROMPTS, "schema", {}, _cfg())
    assert result["line_items"][0]["category"] == "Uncertain"


def test_empty_line_items_no_llm_call():
    receipt = {"merchant": "X", "total_cents": 0, "line_items": []}
    llm = MockLLM({"categories": []})
    categorize_line_items(receipt, llm, _PROMPTS, "schema", {}, _cfg())
    assert llm.calls == []


def test_history_hint_included_in_prompt():
    receipt = _receipt("Bananas")
    item_history = {"bananas": Counter({"Food / Groceries": 5})}
    llm = MockLLM({"categories": ["Food / Groceries"]})
    categorize_line_items(receipt, llm, _PROMPTS, "schema", item_history, _cfg())
    user_msg = llm.calls[0][1]
    assert "previously categorized as: Food / Groceries (5×)" in user_msg


def test_history_disabled_omits_hints():
    receipt = _receipt("Bananas")
    item_history = {"bananas": Counter({"Food / Groceries": 5})}
    llm = MockLLM({"categories": ["Food / Groceries"]})
    categorize_line_items(receipt, llm, _PROMPTS, "schema", item_history, _cfg(enabled=False))
    user_msg = llm.calls[0][1]
    assert "previously categorized as" not in user_msg
