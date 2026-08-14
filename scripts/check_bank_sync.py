"""
Read-only pre-flight for the scheduled bank-sync pipeline.

Lists every account with a bank-sync source configured, whether the provider
reports it as `configured`, its most recent transaction date, and what start
date a sync would derive from that. Run this before enabling `[bank_sync]`,
and again any time imports stop — it's the fastest way to tell "the account
lost its provider link" from "nothing new to import".

Never calls run_bank_sync() and never commits.

Run:
    ACTUAL_PASSWORD=... venv/bin/python scripts/check_bank_sync.py
"""

import os
import sys
import tomllib
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    from dotenv import load_dotenv

    load_dotenv()

    with open("config.toml", "rb") as f:
        raw = tomllib.load(f)

    ca_bundle = raw.get("paths", {}).get("ca_bundle")
    if ca_bundle:
        os.environ["REQUESTS_CA_BUNDLE"] = ca_bundle
        os.environ["SSL_CERT_FILE"] = ca_bundle

    password = os.environ.get("ACTUAL_PASSWORD")
    if not password:
        print("FAIL — ACTUAL_PASSWORD not set", file=sys.stderr)
        return 1

    from actual import Actual
    from actual.queries import get_accounts, get_transactions

    kwargs: dict[str, Any] = dict(
        base_url=raw["actual"]["base_url"], password=password, file=raw["actual"]["file"]
    )
    if os.environ.get("ACTUAL_ENCRYPTION_PASSWORD"):
        kwargs["encryption_password"] = os.environ["ACTUAL_ENCRYPTION_PASSWORD"]

    # Read-only throughout: nothing is written and actual.commit() is never called.
    with Actual(**kwargs) as actual:
        accounts = get_accounts(actual.session)
        sync_accounts = [a for a in accounts if a.account_sync_source and a.account_id]

        if not sync_accounts:
            print("No accounts with a bank-sync source configured.")
            return 0

        status_cache: dict[str, bool] = {}
        any_unconfigured = False

        print(f"{'Account':<30} {'Source':<12} {'Configured':<11} {'Last txn':<12} Sync start date")
        for acct in sync_accounts:
            method = acct.account_sync_source.lower()
            if method not in status_cache:
                status_cache[method] = actual.bank_sync_status(method).data.configured
            configured = status_cache[method]
            if not configured:
                any_unconfigured = True

            txns = get_transactions(actual.session, account=acct)
            if txns:
                last_date = max(t.get_date() for t in txns)
                start_date = last_date
            else:
                last_date = None
                start_date = "90 days ago (first sync)"

            print(
                f"{acct.name:<30} {method:<12} {str(configured):<11} "
                f"{str(last_date) if last_date else '(none)':<12} {start_date}"
            )

    if any_unconfigured:
        print(
            "\nFAIL — at least one account's provider reports not configured "
            "(expired/revoked credentials).",
            file=sys.stderr,
        )
        return 1

    print("\nPASS — every sync-enabled account is configured.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
