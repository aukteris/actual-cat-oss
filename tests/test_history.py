"""Unit tests for the historical-categorization lookups."""

from collections import Counter
from types import SimpleNamespace
from unittest.mock import patch

from actual_cat import history


def _cat(id, name, tombstone=0):
    return SimpleNamespace(id=id, name=name, tombstone=tombstone)


def _group(name, categories, tombstone=0):
    return SimpleNamespace(name=name, categories=categories, tombstone=tombstone)


def _txn(*, category_id, is_child=0, payee_id=None, notes=None, tombstone=0):
    return SimpleNamespace(
        category_id=category_id,
        is_child=is_child,
        payee_id=payee_id,
        notes=notes,
        tombstone=tombstone,
    )


class TestNormalize:
    def test_lowercases_and_strips_punctuation(self):
        assert history.normalize("KS  Org. Peanut-Butter!") == "ks org peanut butter"

    def test_collapses_whitespace(self):
        assert history.normalize("  a   b  ") == "a b"


class TestBuildCategoryPathMap:
    def test_builds_group_slash_category(self):
        groups = [_group("Food", [_cat("c1", "Groceries"), _cat("c2", "Dining")])]
        with patch("actual_cat.history.get_category_groups", return_value=groups):
            path_map = history.build_category_path_map(object())
        assert path_map == {"c1": "Food / Groceries", "c2": "Food / Dining"}

    def test_skips_tombstoned(self):
        groups = [
            _group("Food", [_cat("c1", "Groceries"), _cat("c2", "Dead", tombstone=1)]),
            _group("Gone", [_cat("c3", "X")], tombstone=1),
        ]
        with patch("actual_cat.history.get_category_groups", return_value=groups):
            path_map = history.build_category_path_map(object())
        assert path_map == {"c1": "Food / Groceries"}


class TestBuildHistories:
    _PATHS = {"c1": "Food / Groceries", "c2": "Personal / Care"}

    def _build(self, txns):
        with patch("actual_cat.history.get_transactions", return_value=txns):
            return history.build_histories(object(), self._PATHS)

    def test_payee_history_tallies_standalone(self):
        txns = [
            _txn(category_id="c1", payee_id="p1"),
            _txn(category_id="c1", payee_id="p1"),
            _txn(category_id="c2", payee_id="p1"),
        ]
        payee, item = self._build(txns)
        assert payee["p1"] == Counter({"Food / Groceries": 2, "Personal / Care": 1})
        assert item == {}

    def test_item_history_tallies_children_by_description(self):
        txns = [
            _txn(category_id="c1", is_child=1, notes="Bananas"),
            _txn(category_id="c1", is_child=1, notes="bananas"),  # normalizes to same key
            _txn(category_id="c2", is_child=1, notes="Shampoo"),
        ]
        payee, item = self._build(txns)
        assert payee == {}
        assert item["bananas"] == Counter({"Food / Groceries": 2})
        assert item["shampoo"] == Counter({"Personal / Care": 1})

    def test_skips_uncategorized_and_unknown_and_tombstoned(self):
        txns = [
            _txn(category_id=None, payee_id="p1"),            # uncategorized
            _txn(category_id="gone", payee_id="p1"),          # not in path_map
            _txn(category_id="c1", payee_id="p1", tombstone=1),  # tombstoned
            _txn(category_id="c1", is_child=1, notes=""),     # blank description
        ]
        payee, item = self._build(txns)
        assert payee == {}
        assert item == {}


class TestRenderPayeeHint:
    _HIST = {"p1": Counter({"Food / Groceries": 8, "Personal / Care": 1})}

    def test_renders_top_n(self):
        hint = history.render_payee_hint(self._HIST, "p1", top_n=3, min_count=1)
        assert "Past categorizations for this payee" in hint
        assert "Food / Groceries (8×)" in hint
        assert "Personal / Care (1×)" in hint

    def test_respects_top_n_limit(self):
        hint = history.render_payee_hint(self._HIST, "p1", top_n=1, min_count=1)
        assert "Food / Groceries (8×)" in hint
        assert "Personal / Care" not in hint

    def test_min_count_filters(self):
        hint = history.render_payee_hint(self._HIST, "p1", top_n=3, min_count=2)
        assert "Food / Groceries (8×)" in hint
        assert "Personal / Care" not in hint

    def test_empty_when_no_payee_or_history(self):
        assert history.render_payee_hint(self._HIST, None, top_n=3, min_count=1) == ""
        assert history.render_payee_hint(self._HIST, "unknown", top_n=3, min_count=1) == ""
        assert history.render_payee_hint({}, "p1", top_n=3, min_count=1) == ""


class TestItemHints:
    _HIST = {
        "bananas": Counter({"Food / Groceries": 5}),
        "gift card": Counter({"Gifts / Cards": 2, "Food / Groceries": 1}),
    }

    def test_aligned_to_descriptions(self):
        hints = history.item_hints(
            self._HIST, ["Bananas", "Unknown Item", "Gift Card"], top_n=3, min_count=1
        )
        assert hints[0] == "Food / Groceries (5×)"
        assert hints[1] == ""  # no history
        assert "Gifts / Cards (2×)" in hints[2]

    def test_min_count_filters_to_empty(self):
        hints = history.item_hints(self._HIST, ["Bananas"], top_n=3, min_count=10)
        assert hints == [""]
