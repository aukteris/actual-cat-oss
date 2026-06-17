"""
Delete ALL transactions in the DevBudget (dev reset helper).

Tombstones every transaction via actualpy's .delete() and syncs, so the server
and any connected clients see them removed. DevBudget only — never point this
at a real budget.

Run:
  ACTUAL_PASSWORD=... python scripts/clear_dev_budget.py
"""

import os
import sys

from actual import Actual
from actual.database import Transactions
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.environ.get("ACTUAL_BASE_URL", "https://actual.example.com")
BUDGET_FILE = "DevBudget"
CA_BUNDLE = "/etc/ssl/certs/ca-certificates.crt"

password = os.environ.get("ACTUAL_PASSWORD")
if not password:
    sys.exit("ACTUAL_PASSWORD not set")

if BUDGET_FILE != "DevBudget":
    sys.exit("Refusing to clear a non-Dev budget")

os.environ["REQUESTS_CA_BUNDLE"] = CA_BUNDLE
os.environ["SSL_CERT_FILE"] = CA_BUNDLE

with Actual(base_url=BASE_URL, password=password, file=BUDGET_FILE) as actual:
    rows = (
        actual.session.query(Transactions)
        .filter(Transactions.tombstone == 0)
        .all()
    )
    print(f"Found {len(rows)} live transactions in {BUDGET_FILE}")

    for t in rows:
        t.delete()

    actual.commit()
    print(f"Deleted {len(rows)} transactions. DevBudget is now empty.")
