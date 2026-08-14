"""
Replay the pending-duplicate matcher over the budget's own history. Read-only.

The budget is its own labeled data set. Rows that were imported pending and
later deleted by hand are ground truth for what the matcher *should* propose;
rows that were never pending are ground truth for what it should not. This
script measures both, so the rule constants can be tuned before the pipeline is
ever enabled — it costs nothing and never calls the LLM.

Only Stage 1 (clustering + rules R1-R5) is exercised. Stage 2 adjudication is
what removes the lookalikes the rules cannot separate, so a nonzero spurious
rate here is expected rather than a defect: it is the number the prompt has to
beat.

Run:
  ACTUAL_PASSWORD=... venv/bin/python scripts/backtest_duplicates.py
  ACTUAL_PASSWORD=... venv/bin/python scripts/backtest_duplicates.py --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tomllib
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv  # noqa: E402

from actual_cat.duplicates import (  # noqa: E402
    Candidate,
    cluster_pending,
    matches_cluster,
    propose_candidates,
)
from actual_cat.sync_meta import is_pending, parse_synced  # noqa: E402


def was_pending(txn: Any) -> bool:
    """Imported as pending, whatever happened to it afterwards.

    Deliberately not sync_meta.is_pending(): a row deleted while pending keeps
    cleared = 0, but a row that cleared before being deleted does not, and both
    belong to the labeled history.

    Use this only to build the positive set. It must never gate which rows can
    be a *posted counterpart* — `booked` is a snapshot that is never refreshed,
    so a posted row that arrived pending still reports false here forever, and
    excluding those discards the very survivors the matcher needs. That gate is
    is_pending(), the cleared-aware conjunction the pipeline itself uses.
    """
    payload = parse_synced(txn)
    return payload is not None and payload.get("booked") is False


def band_config(raw: dict[str, Any]) -> SimpleNamespace:
    """The band constants propose_candidates() reads, from [duplicates] or defaults."""
    return SimpleNamespace(
        duplicates_window_days=raw.get("window_days", 7),
        duplicates_max_uplift_pct=raw.get("max_uplift_pct", 40),
        duplicates_max_reduction_pct=raw.get("max_reduction_pct", 5),
        duplicates_auth_hold_max_cents=raw.get("auth_hold_max_cents", 200),
    )


def _within_window(cluster: list[Any], row: Any, window_days: int) -> bool:
    row_date = row.get_date()
    return any(abs((row_date - member.get_date()).days) <= window_days for member in cluster)


def candidates_for(
    rows: list[Any], survivors: list[Any], cfg: SimpleNamespace
) -> list[Candidate]:
    """Run Stage 1 over `rows`, searching `survivors` for the posted counterpart.

    Mirrors find_booked_candidates()'s filters without the session, which is the
    reason Stage 1 is pure over row lists.
    """
    window = cfg.duplicates_window_days
    out: list[Candidate] = []
    for cluster in cluster_pending(rows):
        cluster_ids = {r.id for r in cluster}
        booked = [
            r for r in survivors
            if r.acct == cluster[0].acct
            and r.id not in cluster_ids
            and not r.is_child
            and not is_pending(r)
            and _within_window(cluster, r, window)
            and matches_cluster(cluster, r)
        ]
        out.extend(propose_candidates(cluster, booked, cfg))
    return out


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Backtest the pending-duplicate matcher")
    parser.add_argument("--config", default="config.toml",
                        help="source of the [actual] connection and [duplicates] bands")
    parser.add_argument("--negatives", type=int, default=200,
                        help="how many never-pending booked rows to test against")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()

    with open(args.config, "rb") as f:
        raw = tomllib.load(f)

    ca_bundle = raw.get("paths", {}).get("ca_bundle")
    if ca_bundle:
        os.environ["REQUESTS_CA_BUNDLE"] = ca_bundle
        os.environ["SSL_CERT_FILE"] = ca_bundle

    password = os.environ.get("ACTUAL_PASSWORD")
    if not password:
        sys.exit("ACTUAL_PASSWORD not set")

    from actual import Actual
    from actual.database import Transactions

    kwargs: dict[str, Any] = dict(
        base_url=raw["actual"]["base_url"], password=password, file=raw["actual"]["file"]
    )
    if os.environ.get("ACTUAL_ENCRYPTION_PASSWORD"):
        kwargs["encryption_password"] = os.environ["ACTUAL_ENCRYPTION_PASSWORD"]

    cfg = band_config(raw.get("duplicates", {}))

    # Read-only throughout: nothing is written and actual.commit() is never called.
    with Actual(**kwargs) as actual:
        rows = [r for r in actual.session.query(Transactions).all() if r.date is not None]

        survivors = [r for r in rows if not r.tombstone]
        positives = [r for r in rows if r.tombstone and was_pending(r)]
        negatives = [r for r in survivors if not was_pending(r)][-args.negatives:]

        recovered_ids: set[str] = set()
        held_ids: set[str] = set()
        by_rule: Counter[str] = Counter()
        for candidate in candidates_for(positives, survivors, cfg):
            by_rule[candidate.rule] += 1
            if candidate.rule == "ambiguous":
                held_ids.update(r.id for r in candidate.pending)
            else:
                recovered_ids.update(r.id for r in candidate.pending)

        # Pretend each never-pending row was pending and see whether the rules
        # would have proposed removing it.
        spurious = [c for c in candidates_for(negatives, survivors, cfg) if c.rule != "ambiguous"]

        report = {
            "positives": len(positives),
            "recovered": len(recovered_ids),
            "recall": round(len(recovered_ids) / len(positives), 3) if positives else None,
            # Held rows are surfaced for review rather than acted on, so they are
            # neither a hit nor a miss — counting them as misses understates the
            # matcher and hides the fact that it did the right thing.
            "held": len(held_ids - recovered_ids),
            "by_rule": dict(by_rule),
            "negatives": len(negatives),
            "spurious": len(spurious),
            "spurious_rate": round(len(spurious) / len(negatives), 3) if negatives else None,
            "window_days": cfg.duplicates_window_days,
            "max_uplift_pct": cfg.duplicates_max_uplift_pct,
            "max_reduction_pct": cfg.duplicates_max_reduction_pct,
            "auth_hold_max_cents": cfg.duplicates_auth_hold_max_cents,
        }

        if args.json:
            print(json.dumps(report, indent=2))
            return

        print(f"\nPending-duplicate backtest — {raw['actual']['file']}")
        print(f"  window +/-{cfg.duplicates_window_days}d, "
              f"band [-{cfg.duplicates_max_reduction_pct}%, +{cfg.duplicates_max_uplift_pct}%], "
              f"probe <= {cfg.duplicates_auth_hold_max_cents}c\n")

        print(f"  Recall     {len(recovered_ids)}/{len(positives)} hand-deleted pending rows "
              f"proposed")
        for rule, count in sorted(by_rule.items()):
            print(f"               {rule:<12} {count}")

        held_only = held_ids - recovered_ids
        if held_only:
            print(f"\n  Held       {len(held_only)} more surfaced for review rather than acted "
                  f"on (several booked candidates)")

        if negatives:
            print(f"\n  Spurious   {len(spurious)}/{len(negatives)} never-pending rows would "
                  f"have been proposed ({len(spurious) / len(negatives):.1%})")
            print("             Stage 2 adjudication exists to remove these; this is the")
            print("             rate the prompt has to beat, not a defect.")

        missed = [r for r in positives if r.id not in recovered_ids and r.id not in held_ids]
        if missed:
            print(f"\n  Missed ({len(missed)}) — neither proposed nor held:")
            for row in missed[:20]:
                print(f"    {row.get_date()}  {row.amount / 100:>10.2f}  "
                      f"{(row.imported_description or '')[:40]}")
        print()


if __name__ == "__main__":
    main()
