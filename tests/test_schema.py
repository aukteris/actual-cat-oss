from unittest.mock import MagicMock, patch

from actual_cat.schema import build_schema_text


def make_cat(name: str, tombstone: int = 0) -> MagicMock:
    c = MagicMock()
    c.name = name
    c.tombstone = tombstone
    return c


def make_group(name: str, is_income: int = 0, tombstone: int = 0, cats=()) -> MagicMock:
    g = MagicMock()
    g.name = name
    g.is_income = is_income
    g.tombstone = tombstone
    g.categories = list(cats)
    return g


def test_basic_schema_text():
    groups = [
        make_group("Discretionary", cats=[make_cat("Dining Out"), make_cat("Groceries")]),
        make_group("Fixed Expenses", cats=[make_cat("Rent")]),
    ]
    with patch("actual_cat.schema.get_category_groups", return_value=groups):
        text = build_schema_text(MagicMock())

    assert "Discretionary" in text
    assert "  - Dining Out" in text
    assert "  - Groceries" in text
    assert "Fixed Expenses" in text
    assert "  - Rent" in text


def test_income_groups_included():
    groups = [
        make_group("Income", is_income=1, cats=[make_cat("Salary")]),
        make_group("Spending", cats=[make_cat("Food")]),
    ]
    with patch("actual_cat.schema.get_category_groups", return_value=groups):
        text = build_schema_text(MagicMock())

    assert "Income" in text
    assert "  - Salary" in text
    assert "Spending" in text


def test_tombstoned_group_excluded():
    groups = [
        make_group("Old Group", tombstone=1, cats=[make_cat("Old Cat")]),
        make_group("Active", cats=[make_cat("Active Cat")]),
    ]
    with patch("actual_cat.schema.get_category_groups", return_value=groups):
        text = build_schema_text(MagicMock())

    assert "Old Group" not in text
    assert "Active" in text


def test_tombstoned_category_excluded():
    groups = [
        make_group("Group", cats=[make_cat("Live Cat"), make_cat("Dead Cat", tombstone=1)]),
    ]
    with patch("actual_cat.schema.get_category_groups", return_value=groups):
        text = build_schema_text(MagicMock())

    assert "Live Cat" in text
    assert "Dead Cat" not in text


def test_groups_sorted_alphabetically():
    groups = [make_group("Zzz"), make_group("Aaa")]
    with patch("actual_cat.schema.get_category_groups", return_value=groups):
        text = build_schema_text(MagicMock())

    assert text.index("Aaa") < text.index("Zzz")


def test_empty_schema():
    with patch("actual_cat.schema.get_category_groups", return_value=[]):
        text = build_schema_text(MagicMock())
    assert text == ""
