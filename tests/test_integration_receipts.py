"""Integration tests for the receipt splitting pipeline.

These tests run against a real dev budget (set ACTUAL_BASE_URL for the server).
They are skipped unless ACTUAL_PASSWORD is set in the environment.

Run:
    venv/bin/pytest -m integration -v

Each test:
  - Seeds a transaction directly into the live DevBudget
  - Injects a pre-built pending receipt (bypasses OCR — tests match/split only)
  - Calls process_receipt_splits() directly
  - Asserts the outcome
  - Cleans up (tombstones the transaction, removes receipt files)
"""

import json
import os
import uuid
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Skip if not configured for integration testing
# ---------------------------------------------------------------------------

if not os.environ.get("ACTUAL_PASSWORD"):
    pytest.skip("ACTUAL_PASSWORD not set — skipping integration tests", allow_module_level=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cfg():
    from actual_cat.config import load_config
    return load_config()


@pytest.fixture(scope="module")
def llm(cfg):
    from actual_cat.llm import LLMClient
    return LLMClient(cfg.llm_text, cfg.llm_vision)


@pytest.fixture(scope="module")
def audit(cfg, tmp_path_factory):
    from actual_cat.audit import AuditLogger
    log_dir = tmp_path_factory.mktemp("audit")
    return AuditLogger(str(log_dir / "test-integration.jsonl"))


@pytest.fixture
def store_path(tmp_path):
    """Isolated receipt store for each test."""
    return str(tmp_path / "receipts")


@pytest.fixture
def actual_session(cfg):
    """Open an Actual session and yield it; commit + close after the test."""
    import os as _os

    from actual import Actual

    if cfg.ca_bundle:
        _os.environ["REQUESTS_CA_BUNDLE"] = cfg.ca_bundle
        _os.environ["SSL_CERT_FILE"] = cfg.ca_bundle

    kwargs = dict(base_url=cfg.base_url, password=cfg.password, file=cfg.file)
    if cfg.encryption_password:
        kwargs["encryption_password"] = cfg.encryption_password

    with Actual(**kwargs) as actual:
        yield actual
        actual.commit()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_checking(session):
    from actual.queries import get_accounts
    accounts = {a.name: a for a in get_accounts(session)}
    acct = accounts.get("Checking")
    if acct is None:
        pytest.skip("'Checking' account not found in DevBudget")
    return acct


def _seed_txn(session, account, amount_cents: int, txn_date: date,
               descriptor: str, category_id=None):
    from decimal import Decimal

    from actual.queries import create_transaction
    tag = uuid.uuid4().hex[:8]
    txn = create_transaction(
        session,
        date=txn_date,
        account=account,
        payee="Integration Test Payee",
        notes=descriptor,
        # create_transaction's amount is decimal *dollars* (it runs
        # decimal_to_cents internally); convert from the cents these tests use.
        amount=Decimal(amount_cents) / 100,
        imported_id=f"inttest-{tag}",
    )
    if category_id:
        txn.category_id = category_id
    return txn


def _inject_pending_receipt(store_path: str, total_cents: int, txn_date: date,
                             merchant: str = "Test Merchant",
                             category: str = "Food / Groceries") -> str:
    """Write a pre-built pending receipt JSON directly to store pending/."""
    import json
    import uuid
    from pathlib import Path

    receipt_id = uuid.uuid4().hex
    root = Path(store_path)
    (root / "pending").mkdir(parents=True, exist_ok=True)

    meta = {
        "id": receipt_id,
        "status": "pending",
        "source": "test",
        "received_ts": datetime.now(timezone.utc).isoformat(),
        "image_path": "/dev/null",
        "ocr": {
            "merchant": merchant,
            "date": txn_date.isoformat(),
            "total_cents": total_cents,
            "line_items": [
                {"description": "Test item", "amount_cents": total_cents, "category": category}
            ],
        },
    }
    (root / "pending" / f"{receipt_id}.json").write_text(json.dumps(meta, indent=2))
    return receipt_id


def _lookup_category_id(session, path: str):
    from actual_cat.categorization import lookup_category_id
    return lookup_category_id(session, path)


def _run_pipeline(actual, llm, audit, cfg, store_path, mode="suggest"):
    from unittest.mock import MagicMock

    import actual_cat.prompts as prompts
    from actual_cat.receipts.match import process_receipt_splits
    from actual_cat.schema import build_schema_text

    # Override store_path and mode via a lightweight cfg wrapper
    patched_cfg = MagicMock(wraps=cfg)
    patched_cfg.receipts_store_path = store_path
    patched_cfg.receipts_mode = mode
    patched_cfg.receipts_match_window_days = cfg.receipts_match_window_days
    patched_cfg.receipts_expiry_days = cfg.receipts_expiry_days
    patched_cfg.receipts_threshold = cfg.receipts_threshold

    schema_text = build_schema_text(actual.session)
    process_receipt_splits(actual, llm, audit, patched_cfg, schema_text, prompts)
    actual.commit()


def _done_records(store_path: str) -> list[dict]:
    root = Path(store_path) / "done"
    if not root.exists():
        return []
    return [json.loads(p.read_text()) for p in root.glob("*.json")]


def _pending_records(store_path: str) -> list[dict]:
    root = Path(store_path) / "pending"
    if not root.exists():
        return []
    return [json.loads(p.read_text()) for p in root.glob("*.json")]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_suggest_mode_tags_matching_transaction(actual_session, llm, audit, cfg, store_path):
    """Suggest mode: matched transaction gets #ai-suggested-split, no split rows created."""
    acct = _get_checking(actual_session.session)
    amount = -18347  # $183.47 — unlikely to collide with real transactions
    txn_date = date(2026, 6, 5)

    txn = _seed_txn(actual_session.session, acct, amount, txn_date, "INTTEST SUGGEST 18347")
    actual_session.commit()

    _inject_pending_receipt(store_path, abs(amount), txn_date, "Test Store Suggest")
    _run_pipeline(actual_session, llm, audit, cfg, store_path, mode="suggest")

    assert "#ai-suggested-split" in (txn.notes or ""), f"notes: {txn.notes!r}"
    assert txn.is_parent == 0, "suggest mode must not create split rows"

    done = _done_records(store_path)
    assert len(done) == 1
    assert done[0]["status"] == "suggested"
    assert done[0]["matched_txn_id"] == txn.id

    # Cleanup
    txn.tombstone = 1
    actual_session.commit()


def test_apply_mode_creates_split_rows(actual_session, llm, audit, cfg, store_path):
    """Apply mode: parent is_parent=1, category_id cleared, children sum to parent amount."""
    from actual.queries import get_transactions

    acct = _get_checking(actual_session.session)
    amount = -22519  # $225.19
    txn_date = date(2026, 6, 4)

    txn = _seed_txn(actual_session.session, acct, amount, txn_date, "INTTEST APPLY 22519")
    actual_session.commit()

    _inject_pending_receipt(store_path, abs(amount), txn_date, "Test Store Apply")
    _run_pipeline(actual_session, llm, audit, cfg, store_path, mode="apply")

    assert txn.is_parent == 1, "parent must have is_parent=1"
    assert txn.category_id is None, "parent category_id must be cleared"
    assert "#ai-receipt-split" in (txn.notes or ""), f"notes: {txn.notes!r}"

    # Children: sum must equal parent amount
    children = [
        t for t in get_transactions(actual_session.session)
        if getattr(t, "parent_id", None) == txn.id and not t.tombstone
    ]
    assert children, "no child split rows found"
    child_sum = sum(c.amount for c in children)
    assert child_sum == amount, f"children sum {child_sum} != parent {amount}"

    done = _done_records(store_path)
    assert len(done) == 1
    assert done[0]["status"] == "applied"

    # Cleanup
    txn.tombstone = 1
    for c in children:
        c.tombstone = 1
    actual_session.commit()


def test_no_match_leaves_receipt_pending(actual_session, llm, audit, cfg, store_path):
    """When no transaction matches the receipt amount, it stays pending."""
    txn_date = date(2026, 6, 3)
    # Use an amount that almost certainly has no matching transaction
    _inject_pending_receipt(store_path, 99991, txn_date, "Ghost Merchant")
    _run_pipeline(actual_session, llm, audit, cfg, store_path, mode="suggest")

    assert _pending_records(store_path), "receipt should remain pending"
    assert not _done_records(store_path), "receipt must not be moved to done"


def test_ambiguous_match_holds_receipt(actual_session, llm, audit, cfg, store_path):
    """When two transactions have the same amount, receipt stays pending (not guessed)."""
    acct = _get_checking(actual_session.session)
    amount = -10101  # $101.01
    txn_date = date(2026, 6, 2)

    txn_a = _seed_txn(actual_session.session, acct, amount, txn_date, "INTTEST AMBIG A")
    txn_b = _seed_txn(actual_session.session, acct, amount, txn_date, "INTTEST AMBIG B")
    actual_session.commit()

    _inject_pending_receipt(store_path, abs(amount), txn_date, "Ambiguous Merchant")
    _run_pipeline(actual_session, llm, audit, cfg, store_path, mode="suggest")

    assert _pending_records(store_path), "ambiguous receipt should remain pending"
    assert "#ai-suggested-split" not in (txn_a.notes or "")
    assert "#ai-suggested-split" not in (txn_b.notes or "")

    # Cleanup
    txn_a.tombstone = 1
    txn_b.tombstone = 1
    actual_session.commit()


def test_override_already_categorized_transaction(actual_session, llm, audit, cfg, store_path):
    """Receipt wins over an existing single category when applied."""
    acct = _get_checking(actual_session.session)
    amount = -33311  # $333.11
    txn_date = date(2026, 6, 1)

    # Pre-assign a category to simulate a transaction categorized by an earlier run
    groceries_id = _lookup_category_id(actual_session.session, "Food / Groceries")
    if not groceries_id:
        pytest.skip("'Food / Groceries' category not found in DevBudget schema")

    txn = _seed_txn(actual_session.session, acct, amount, txn_date,
                    "INTTEST OVERRIDE 33311", category_id=groceries_id)
    actual_session.commit()

    assert txn.category_id == groceries_id  # confirm pre-categorized

    _inject_pending_receipt(store_path, abs(amount), txn_date, "Override Test Merchant")
    _run_pipeline(actual_session, llm, audit, cfg, store_path, mode="apply")

    assert txn.category_id is None, "prior category must be cleared by the receipt split"
    assert txn.is_parent == 1

    done = _done_records(store_path)
    assert done[0]["prior_category"] == groceries_id

    # Cleanup
    from actual.queries import get_transactions
    children = [
        t for t in get_transactions(actual_session.session)
        if getattr(t, "parent_id", None) == txn.id and not t.tombstone
    ]
    txn.tombstone = 1
    for c in children:
        c.tombstone = 1
    actual_session.commit()
