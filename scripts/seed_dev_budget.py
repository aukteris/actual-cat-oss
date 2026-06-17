"""
Seed the DevBudget with crafted transactions for Phase B/C integration testing.

Transaction structure matches real bank card imports:
  - payee = institution's friendly merchant name (e.g. "Whole Foods")
  - notes = raw bank descriptor with processor prefix + address
    (e.g. "WHOLEFDS #1234 100 EXAMPLE ST ANYTOWN 00000 ST USA")

Covers:
  - Clear chain merchants (Whole Foods, Target, Starbucks)
  - Payment-processor prefixes (SQ *, PAR*, STRIPE*, PP*, AMZN MKTP)
  - Genuine ambiguity (bare ACH reference, no useful descriptor)
  - Refund on a credit card (positive amount)
  - Obvious transfer pair (Checking -> Savings, same day)
  - Credit-card payment pair (Checking -> CC, PMT descriptor)
  - Coincidental amount match between two unrelated merchants

Run:
  ACTUAL_PASSWORD=... python scripts/seed_dev_budget.py
"""

import argparse
import os
import sys
from datetime import date, timedelta
from pathlib import Path

from actual import Actual
from actual.queries import create_transaction, get_accounts, get_category_groups
from dotenv import load_dotenv


def _stage_receipt_fixture(store_path: str) -> None:
    """Stage the Costco fixture image as a fresh inbox receipt."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from actual_cat.receipts.store import save_received

    fixture = Path(__file__).parent / "fixtures" / "costco_receipt.jpg"
    if not fixture.exists():
        print(f"  WARNING: receipt fixture not found at {fixture} — skipping inbox stage")
        return

    receipt_id = save_received(store_path, fixture, source="seed")
    print(f"  Staged receipt fixture → inbox (id={receipt_id})")

load_dotenv()

parser = argparse.ArgumentParser(description="Seed DevBudget with test transactions")
parser.add_argument("--receipts", action="store_true",
                    help="Also seed a Costco receipt-test transaction and stage its image")
parser.add_argument("--store-path", default="receipts",
                    help="Receipt store path (default: receipts)")
args = parser.parse_args()

BASE_URL = os.environ.get("ACTUAL_BASE_URL", "https://actual.example.com")
BUDGET_FILE = "DevBudget"
CA_BUNDLE = "/etc/ssl/certs/ca-certificates.crt"

password = os.environ.get("ACTUAL_PASSWORD")
if not password:
    sys.exit("ACTUAL_PASSWORD not set")

os.environ["REQUESTS_CA_BUNDLE"] = CA_BUNDLE
os.environ["SSL_CERT_FILE"] = CA_BUNDLE

TODAY = date.today()
# Optional suffix so a fresh batch can be seeded for re-testing without colliding
# with the stable imported_id dedupe (e.g. RUN_TAG=$(date +%s)).
RUN_TAG = os.environ.get("RUN_TAG", "")

with Actual(base_url=BASE_URL, password=password, file=BUDGET_FILE) as actual:
    accounts = {a.name: a for a in get_accounts(actual.session)}
    print(f"Accounts found: {list(accounts.keys())}")

    required = {"Checking", "Savings", "Credit Card"}
    missing = required - set(accounts.keys())
    if missing:
        print(f"WARNING: missing accounts {missing} — create them in the Actual UI first")
        print("Continuing with available accounts...")

    checking = accounts.get("Checking")
    savings = accounts.get("Savings")
    cc = accounts.get("Credit Card")

    seeded = []

    def add(account, amount_dollars, payee, notes=None, days_ago=0, txn_date=None):
        if account is None:
            print(f"  SKIP (no account): {payee}")
            return None
        txn = create_transaction(
            actual.session,
            date=txn_date or (TODAY - timedelta(days=days_ago)),
            account=account,
            payee=payee,
            notes=notes,
            amount=amount_dollars,
            imported_id=f"seed-{payee}-{days_ago}{RUN_TAG}",
        )
        seeded.append(txn)
        print(f"  Added: {payee:35s} {amount_dollars:+8.2f}  ({account.name})")
        return txn

    print("\nSeeding categorization candidates...")

    # Clear chain merchants — friendly name + raw descriptor with address
    add(checking, -85.42, "Whole Foods Market",
        notes="WHOLEFDS #1234 100 EXAMPLE ST ANYTOWN 00000 ST USA",
        days_ago=1)
    add(checking, -124.17, "Target",
        notes="TARGET 00012345 200 EXAMPLE AVE ANYTOWN 00000 ST USA",
        days_ago=2)
    add(cc, -18.50, "Starbucks",
        notes="STARBUCKS #08734 300 EXAMPLE RD ANYTOWN 00000 ST USA",
        days_ago=1)

    # Payment-processor prefixes — friendly name hides the processor
    add(checking, -45.00, "Local Bike Shop",
        notes="SQ *LOCAL BIKE SHOP 400 EXAMPLE BLVD ANYTOWN 00000 ST USA",
        days_ago=3)
    add(cc, -12.99, "Substack",
        notes="STRIPE*SUBSTACK.COM 185 BERRY ST STE 550 SAN FRANCISCO 94107 CA USA",
        days_ago=4)
    add(cc, -9.99, "YouTube Premium",
        notes="PP*YOUTUBEPREEMIUM 901 CHERRY AVE SAN BRUNO 94066 CA USA",
        days_ago=5)
    add(checking, -67.43, "Amazon",
        notes="AMZN MKTP US*AB12CD34 AMZN.COM/BILL WA 98109 WA USA",
        days_ago=6)

    # Genuine ambiguity — no useful signal in either field
    add(checking, -200.00, "ACH Debit",
        notes="ACH DEBIT 4829201 ORIG CO NAME:PAYROLL MISC",
        days_ago=7)

    # Refund on credit card (positive amount)
    add(cc, 32.50, "Target",
        notes="TARGET REFUND 00012345 200 EXAMPLE AVE ANYTOWN 00000 ST USA",
        days_ago=2)

    print("\nSeeding transfer candidates...")

    # Obvious transfer pair — Checking -> Savings, same day
    add(checking, -500.00, "Transfer to Savings",
        notes="ONLINE TRANSFER TO SAVINGS ACCOUNT XXXXXX1234",
        days_ago=8)
    add(savings, 500.00, "Transfer from Checking",
        notes="ONLINE TRANSFER FROM CHECKING ACCOUNT XXXXXX5678",
        days_ago=8)

    # Credit-card payment — Checking -> CC, PMT in descriptor
    add(checking, -1200.00, "Apple Card Payment",
        notes="APPLE CARD AUTOPAY PMT APPLE BANK FOR SAVINGS NEW YORK NY USA",
        days_ago=9)
    add(cc, 1200.00, "Apple Card Payment",
        notes="AUTOPAY PMT THANK YOU",
        days_ago=9)

    # Coincidental amount match — two unrelated merchants, same dollar total
    add(checking, -75.00, "REI Co-op",
        notes="REI #37 500 EXAMPLE ST ANYTOWN 00000 ST USA",
        days_ago=10)
    add(cc, -75.00, "Netflix",
        notes="NETFLIX.COM LOS GATOS 95032 CA USA",
        days_ago=11)

    if args.receipts:
        print("\nSeeding receipt-test transaction (Costco, $317.37, yesterday)...")

        # Look up the Groceries category so the transaction arrives pre-categorized.
        # This exercises the receipt-override path: the worker must clear the existing
        # category when applying the split.
        groceries_id = None
        for group in get_category_groups(actual.session):
            if group.tombstone:
                continue
            for cat in group.categories:
                if cat.name == "Groceries" and not cat.tombstone:
                    groceries_id = cat.id
                    break

        costco_txn = add(
            checking, -317.37, "Costco",
            notes="COSTCO WHSE #000 123 EXAMPLE ST ANYTOWN 00000 ST USA",
            days_ago=1,
        )
        if costco_txn and groceries_id:
            costco_txn.category_id = groceries_id
            print(f"  Pre-categorized as Groceries (id={groceries_id}) to test override path")
        elif costco_txn:
            print("  WARNING: 'Groceries' category not found — transaction left uncategorized")

        _stage_receipt_fixture(args.store_path)

    actual.commit()
    print(f"\nDone. {len(seeded)} transactions seeded into DevBudget.")
    print("Run: python -m actual_cat")
