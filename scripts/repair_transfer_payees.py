"""
One-off repair for transfer legs damaged by bank-sync reconciliation.

Actual.run_bank_sync() reconciles every re-reported transaction through
reconcile_transaction(update_existing=True), which is not configurable off.
On an already-paired leg that overwrites payee_id back to the imported
merchant payee and strips the #ai-assisted marker from notes, while leaving
transferred_id set. The result is a malformed half-transfer: linked in the
data model, but Actual no longer renders it as a transfer, and no pipeline
selector (all filter on transferred_id IS NULL) ever looks at the row again.

This shares its classification and repair rules with
actual_cat.transfers.repair_transfer_pairs() — the ongoing per-run guard that
does the same thing automatically from now on — via classify_transfer_leg()
and apply_transfer_repair(), so the one-off sweep and the pipeline safeguard
cannot drift apart.

payee_id/category_id are corrected unconditionally on any paired row, AI-made
or user-made. #ai-assisted is restored only when the partner leg still
carries it, so a manual transfer (never tagged on either side) is repaired
without being mislabeled as agent output.

Orphaned rows (partner missing or tombstoned) are reported, never touched —
that needs a human decision, not an automatic rewrite.

Read-only by default; nothing is written and actual.commit() is never called
unless --apply is passed.

Run:
  venv/bin/python scripts/repair_transfer_payees.py            # dry run, prints the diff
  venv/bin/python scripts/repair_transfer_payees.py --json     # same, machine-readable
  venv/bin/python scripts/repair_transfer_payees.py --apply    # writes and commits
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tomllib
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv  # noqa: E402

from actual_cat.transfers import apply_transfer_repair, classify_transfer_leg  # noqa: E402

LABELS = ("healthy", "payee_reset", "marker_lost", "categorized", "orphaned")


def row_report(txn: Any, partner: Any | None, fields: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": txn.id,
        "account": txn.account.name if txn.account else None,
        "date": str(txn.get_date()),
        "amount": round(txn.amount / 100, 2) if txn.amount is not None else None,
        "current_payee_id": txn.payee_id,
        "target_payee_id": fields.get("payee_id", txn.payee_id),
        "partner_id": partner.id if partner is not None else txn.transferred_id,
    }


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Repair transfer legs damaged by bank-sync reconciliation "
        "(payee_id / category_id / #ai-assisted)"
    )
    parser.add_argument("--config", default="config.toml",
                        help="source of the [actual] connection")
    parser.add_argument("--apply", action="store_true",
                        help="write repairs and commit (default: dry run, nothing written)")
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

    with Actual(**kwargs) as actual:
        rows = (
            actual.session.query(Transactions)
            .filter(Transactions.transferred_id.isnot(None), Transactions.tombstone == 0)
            .all()
        )

        by_label: dict[str, list[dict[str, Any]]] = {label: [] for label in LABELS}

        for txn in rows:
            partner = txn.transfer
            label, fields = classify_transfer_leg(txn, partner)
            by_label[label].append(row_report(txn, partner, fields))

            if args.apply and fields:
                apply_transfer_repair(txn, fields)

        if args.apply:
            actual.commit()

        affected = by_label["payee_reset"] + by_label["marker_lost"] + by_label["categorized"]

        report = {
            "file": raw["actual"]["file"],
            "total_paired_rows": len(rows),
            "applied": args.apply,
            "counts": {label: len(by_label[label]) for label in LABELS},
            "rows": affected,
            "orphaned_rows": by_label["orphaned"],
        }

        if args.json:
            print(json.dumps(report, indent=2))
            return 0

        print(f"\nTransfer-payee repair — {raw['actual']['file']}")
        print(f"  {len(rows)} paired rows scanned\n")

        if affected:
            verb = "Repaired" if args.apply else "Would repair"
            print(f"  {verb} ({len(affected)}):")
            print(
                f"    {'id':<38} {'account':<24} {'date':<12} {'amount':>10}  "
                f"{'current payee':<38} {'target payee':<38} partner id"
            )
            for row in affected:
                amount = f"{row['amount']:.2f}" if row["amount"] is not None else ""
                print(
                    f"    {row['id']:<38} {(row['account'] or '')[:24]:<24} {row['date']:<12} "
                    f"{amount:>10}  {(row['current_payee_id'] or '')[:38]:<38} "
                    f"{(row['target_payee_id'] or '')[:38]:<38} {row['partner_id']}"
                )
        else:
            print("  Nothing to repair.")

        if by_label["orphaned"]:
            print(
                f"\n  Orphaned ({len(by_label['orphaned'])}) — partner missing or "
                "tombstoned, needs a human decision, not touched:"
            )
            for row in by_label["orphaned"]:
                print(
                    f"    {row['id']:<38} {(row['account'] or '')[:24]:<24} {row['date']:<12} "
                    f"partner={row['partner_id']}"
                )

        if not args.apply and affected:
            print("\n  Dry run — nothing written. Re-run with --apply to write and commit.")

        print()
        return 0


if __name__ == "__main__":
    sys.exit(main())
