from typing import Any

from actual.queries import get_category_groups


def build_schema_text(session: Any) -> str:
    """Render the current category tree as a text block for the system prompt.

    Built fresh each run so prompt stays in sync with schema edits in the UI.
    Spike confirmed: is_income and tombstone are int (0/1).

    Income groups are included (not just expense groups) so genuine income
    (payroll, deposits) has a real category to land in — see
    categorization.is_income_category for the paired safeguard that keeps
    an income category from being applied to an outflow transaction.
    """

    groups = get_category_groups(session)
    lines: list[str] = []
    for group in sorted(groups, key=lambda g: g.name):
        if group.tombstone:
            continue
        lines.append(group.name)
        for cat in sorted(group.categories, key=lambda c: c.name):
            if cat.tombstone:
                continue
            lines.append(f"  - {cat.name}")
    return "\n".join(lines)
