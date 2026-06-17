from typing import Any

from actual.queries import get_category_groups


def build_schema_text(session: Any) -> str:
    """Render the current category tree as a text block for the system prompt.

    Built fresh each run so prompt stays in sync with schema edits in the UI.
    Spike confirmed: is_income and tombstone are int (0/1); income groups excluded.
    """

    groups = get_category_groups(session)
    lines: list[str] = []
    for group in sorted(groups, key=lambda g: g.name):
        if group.is_income or group.tombstone:
            continue
        lines.append(group.name)
        for cat in sorted(group.categories, key=lambda c: c.name):
            if cat.tombstone:
                continue
            lines.append(f"  - {cat.name}")
    return "\n".join(lines)
