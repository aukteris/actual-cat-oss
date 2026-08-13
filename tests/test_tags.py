from actual_cat.tags import (
    append_tag,
    has_ai_marker,
    has_duplicate_marker,
    slugify_category,
    strip_category_tag,
)


class TestHasAiMarker:
    def test_none_notes(self):
        assert not has_ai_marker(None)

    def test_empty_notes(self):
        assert not has_ai_marker("")

    def test_no_marker(self):
        assert not has_ai_marker("some normal note")

    def test_ai_assisted(self):
        assert has_ai_marker("#ai-assisted")

    def test_ai_suggested_with_category(self):
        assert has_ai_marker("note #ai:groceries-food trailing")

    def test_ai_suggested_transfer(self):
        assert has_ai_marker("#ai-suggested-transfer")

    def test_marker_embedded_in_text(self):
        assert has_ai_marker("paid online #ai-assisted rest of note")

    def test_non_marker_hash(self):
        assert not has_ai_marker("#tax:deductible:home")


class TestDuplicateMarkers:
    def test_suggested_duplicate_hides_row_from_categorization(self):
        assert has_ai_marker("#ai-suggested-duplicate GREEN GROCER")

    def test_review_tag_hides_row_from_categorization(self):
        assert has_ai_marker("#ai-duplicate-review GREEN GROCER")

    def test_has_duplicate_marker_matches_both(self):
        assert has_duplicate_marker("#ai-suggested-duplicate")
        assert has_duplicate_marker("memo #ai-duplicate-review")

    def test_has_duplicate_marker_ignores_other_pipelines(self):
        assert not has_duplicate_marker("#ai-suggested-transfer")
        assert not has_duplicate_marker("#ai:groceries-food")
        assert not has_duplicate_marker(None)


class TestSlugifyCategory:
    def test_group_and_category(self):
        assert slugify_category("Usual Expenses / Food") == "usual-expenses-food"

    def test_single_word_group(self):
        assert slugify_category("Groceries / Food") == "groceries-food"

    def test_uncertain(self):
        assert slugify_category("Uncertain") == "uncertain"

    def test_no_spaces_in_output(self):
        result = slugify_category("Fixed Expenses / Rent and Utilities")
        assert " " not in result

    def test_no_slash_in_output(self):
        result = slugify_category("Home Infrastructure / Internet")
        assert "/" not in result
        assert result == "home-infrastructure-internet"


class TestAppendTag:
    def test_none_notes(self):
        assert append_tag(None, "#ai-assisted") == "#ai-assisted"

    def test_empty_notes(self):
        assert append_tag("", "#ai-assisted") == "#ai-assisted"

    def test_prepends_to_existing(self):
        result = append_tag("some note", "#ai-assisted")
        assert result == "#ai-assisted some note"

    def test_idempotent(self):
        result = append_tag("#ai-assisted", "#ai-assisted")
        assert result == "#ai-assisted"

    def test_idempotent_already_prepended(self):
        result = append_tag("#ai-assisted some note", "#ai-assisted")
        assert result == "#ai-assisted some note"

    def test_tag_as_substring_not_confused(self):
        result = append_tag("note", "#ai-assisted")
        assert result.startswith("#ai-assisted")


class TestStripCategoryTag:
    def test_removes_marker_keeps_prose(self):
        assert strip_category_tag("#ai:uncertain ONLINE TRANSFER") == "ONLINE TRANSFER"

    def test_keeps_freeform_tags(self):
        result = strip_category_tag("#ai:usual-expenses-food #groceries WHOLEFDS")
        assert result == "#groceries WHOLEFDS"

    def test_preserves_transfer_marker(self):
        result = strip_category_tag("#ai-suggested-transfer #ai:uncertain x")
        assert result == "#ai-suggested-transfer x"

    def test_none_returns_empty(self):
        assert strip_category_tag(None) == ""

    def test_empty_returns_empty(self):
        assert strip_category_tag("") == ""
