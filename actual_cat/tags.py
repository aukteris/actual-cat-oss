import re

AI_MARKER_RE = re.compile(
    r"#ai(-suggested-transfer|-assisted|-suggested-split|-receipt-split"
    r"|-suggested-duplicate|-duplicate-review|:\S+)"
)
CATEGORY_TAG_RE = re.compile(r"#ai:\S+")


def slugify_category(category_path: str) -> str:
    """Convert 'Group / Category' to 'group/category' for use in a tag.

    Actual Budget tags are space-delimited; spaces inside a tag break parsing.
    Slash is kept as the group/category separator to mirror the schema structure.
    """
    return category_path.lower().replace(" / ", "-").replace(" ", "-")


SPLIT_MARKER_RE = re.compile(r"#ai(-suggested-split|-receipt-split)")
APPLIED_SPLIT_MARKER_RE = re.compile(r"#ai-receipt-split")
DUPLICATE_MARKER_RE = re.compile(r"#ai(-suggested-duplicate|-duplicate-review)")


def has_ai_marker(notes: str | None) -> bool:
    if not notes:
        return False
    return bool(AI_MARKER_RE.search(notes))


def strip_category_tag(notes: str | None) -> str:
    """Remove any #ai:<category> marker, preserving everything else.

    Used when a transaction is promoted to a transfer after having been
    categorized on an earlier run (one leg posted before the other). Only the
    colon-form marker we own is removed; #ai-suggested-transfer / #ai-assisted
    and free-form #tags are left intact.
    """
    if not notes:
        return ""
    return CATEGORY_TAG_RE.sub("", notes).replace("  ", " ").strip()


def has_split_marker(notes: str | None) -> bool:
    if not notes:
        return False
    return bool(SPLIT_MARKER_RE.search(notes))


def has_applied_split_marker(notes: str | None) -> bool:
    """True only for finalized receipt splits (#ai-receipt-split).

    Unlike has_split_marker, transactions bearing only #ai-suggested-split
    return False here — they remain eligible for a retry submission.
    """
    if not notes:
        return False
    return bool(APPLIED_SPLIT_MARKER_RE.search(notes))


def has_duplicate_marker(notes: str | None) -> bool:
    """True for a row the duplicate pipeline has already ruled on.

    Covers both #ai-suggested-duplicate (believed to be a duplicate) and
    #ai-duplicate-review (held or refused). Either way the row is in front of
    the user already, so re-adjudicating it would just spend a call per run.
    """
    if not notes:
        return False
    return bool(DUPLICATE_MARKER_RE.search(notes))


def remove_tag(notes: str | None, tag: str) -> str:
    """Remove a specific tag string from notes."""
    if not notes:
        return ""
    return notes.replace(tag, "").replace("  ", " ").strip()


def append_tag(notes: str | None, tag: str) -> str:
    """Append a tag to notes, preserving original content.

    Tags are separated from prose content (and from each other) by a single
    space. The tag is not appended if already present (idempotent).
    """
    existing = notes or ""
    existing = existing.strip()
    if tag in existing:
        return existing
    if existing:
        return f"{tag} {existing}"
    return tag
