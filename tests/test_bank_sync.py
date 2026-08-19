"""Unit tests for the scheduled bank-sync pipeline.

actualpy's own Actual.run_bank_sync() has no per-account failure isolation
(one account raising ActualBankSyncError aborts every account after it), so
bank_sync.py calls it once per account itself. These tests exercise that
loop, the account selection filters, and the state-gate integration —
against a fully mocked `actual` (no real Actual/session/network involved).
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

from actual.exceptions import ActualBankSyncError

from actual_cat.bank_sync import process_bank_sync
from actual_cat.state import SyncState

UTC = timezone.utc
NOW = datetime(2026, 8, 14, 12, 0, tzinfo=UTC)


def make_account(id="acct-1", name="Checking", sync_source="simplefin", account_id="ext-1"):
    acct = MagicMock()
    acct.id = id
    acct.name = name
    acct.account_sync_source = sync_source
    acct.account_id = account_id
    return acct


def make_cfg(**overrides):
    cfg = MagicMock()
    cfg.bank_sync_enabled = True
    cfg.bank_sync_interval_minutes = 360
    cfg.bank_sync_grace_minutes = 5
    cfg.bank_sync_max_runs_per_day = 0
    cfg.bank_sync_accounts = []
    cfg.bank_sync_exclude_accounts = []
    cfg.bank_sync_lookback_days = 0
    cfg.bank_sync_allow_first_sync = False
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def make_actual(run_bank_sync_side_effect=None, configured=True):
    actual = MagicMock()
    actual.session = MagicMock()
    actual.bank_sync_status.return_value.data.configured = configured
    if run_bank_sync_side_effect is not None:
        actual.run_bank_sync.side_effect = run_bank_sync_side_effect
    else:
        actual.run_bank_sync.return_value = []
    return actual


def _state(tmp_path: Path) -> SyncState:
    return SyncState(str(tmp_path / "state.json"))


def _patched(accounts, transactions_by_account=None):
    """Patch module-level account/transaction lookups the same way
    test_duplicates.py patches find_pending/find_booked_candidates."""
    txns = transactions_by_account or {}

    def get_transactions(session, account=None):
        return txns.get(account.id, [MagicMock()])  # non-empty by default

    return (
        patch("actual_cat.bank_sync.get_accounts", return_value=accounts),
        patch("actual_cat.bank_sync.get_transactions", side_effect=get_transactions),
    )


def test_disabled_pipeline_is_skipped(tmp_path: Path):
    actual = make_actual()
    audit = MagicMock()
    cfg = make_cfg(bank_sync_enabled=False)
    state = _state(tmp_path)

    with patch("actual_cat.bank_sync.get_accounts") as get_accounts:
        result = process_bank_sync(actual, audit, cfg, state)

    assert result == 0
    get_accounts.assert_not_called()
    reasons = [c.args[0]["reason"] for c in audit._write.call_args_list if "reason" in c.args[0]]
    assert "disabled" in reasons


def test_not_due_is_skipped(tmp_path: Path):
    actual = make_actual()
    audit = MagicMock()
    cfg = make_cfg()
    state = _state(tmp_path)
    state.set_last_run("bank_sync", NOW - timedelta(minutes=30))

    p1, p2 = _patched([make_account()])
    with p1, p2 as get_transactions_mock:
        result = process_bank_sync(actual, audit, cfg, state, now=NOW)

    assert result == 0
    get_transactions_mock.assert_not_called()
    reasons = [c.args[0]["reason"] for c in audit._write.call_args_list if "reason" in c.args[0]]
    assert "interval" in reasons


def test_one_account_failure_does_not_block_others(tmp_path: Path):
    a1, a2 = make_account("acct-1", "Checking"), make_account("acct-2", "Savings")
    error = ActualBankSyncError("ITEM_LOGIN_REQUIRED", "rejected", "expired credentials")
    actual = make_actual(run_bank_sync_side_effect=[error, [MagicMock(), MagicMock()]])
    audit = MagicMock()
    cfg = make_cfg()
    state = _state(tmp_path)

    p1, p2 = _patched([a1, a2])
    with p1, p2:
        result = process_bank_sync(actual, audit, cfg, state)

    assert result == 2
    assert actual.run_bank_sync.call_count == 2
    events = [c.args[0]["event"] for c in audit._write.call_args_list]
    assert "bank_sync_account_failed" in events
    assert "bank_sync_ok" in events
    ok_event = next(
        c.args[0] for c in audit._write.call_args_list if c.args[0]["event"] == "bank_sync_ok"
    )
    assert ok_event["accounts_synced"] == 1
    assert ok_event["imported_count"] == 2


def test_per_account_ok_event_is_written_for_each_synced_account(tmp_path: Path):
    """One bank_sync_account_ok per account that synced — the aggregatable form
    of the summary event's per_account map (a failed account gets no ok event)."""
    a1, a2, a3 = (
        make_account("acct-1", "Checking"),
        make_account("acct-2", "Savings"),
        make_account("acct-3", "Credit Card"),
    )
    error = ActualBankSyncError("ITEM_LOGIN_REQUIRED", "rejected", "expired credentials")
    actual = make_actual(run_bank_sync_side_effect=[[MagicMock()], error, []])
    audit = MagicMock()
    cfg = make_cfg()
    state = _state(tmp_path)

    p1, p2 = _patched([a1, a2, a3])
    with p1, p2:
        process_bank_sync(actual, audit, cfg, state)

    per_account_events = [
        c.args[0]
        for c in audit._write.call_args_list
        if c.args[0]["event"] == "bank_sync_account_ok"
    ]
    assert [(e["account"], e["imported_count"]) for e in per_account_events] == [
        ("Checking", 1),
        ("Credit Card", 0),
    ]
    assert all(e["pipeline"] == "bank_sync" for e in per_account_events)


def test_allowlist_selects_only_named_accounts(tmp_path: Path):
    a1, a2 = make_account("acct-1", "Checking"), make_account("acct-2", "Savings")
    actual = make_actual()
    audit = MagicMock()
    cfg = make_cfg(bank_sync_accounts=["Checking"])
    state = _state(tmp_path)

    p1, p2 = _patched([a1, a2])
    with p1, p2:
        process_bank_sync(actual, audit, cfg, state)

    synced_names = [c.kwargs["account"].name for c in actual.run_bank_sync.call_args_list]
    assert synced_names == ["Checking"]


def test_denylist_excludes_named_accounts(tmp_path: Path):
    a1, a2 = make_account("acct-1", "Checking"), make_account("acct-2", "Savings")
    actual = make_actual()
    audit = MagicMock()
    cfg = make_cfg(bank_sync_exclude_accounts=["Savings"])
    state = _state(tmp_path)

    p1, p2 = _patched([a1, a2])
    with p1, p2:
        process_bank_sync(actual, audit, cfg, state)

    synced_names = [c.kwargs["account"].name for c in actual.run_bank_sync.call_args_list]
    assert synced_names == ["Checking"]


def test_accounts_without_sync_source_are_never_called(tmp_path: Path):
    unlinked = make_account("acct-1", "Manual Cash", sync_source=None, account_id=None)
    actual = make_actual()
    audit = MagicMock()
    cfg = make_cfg()
    state = _state(tmp_path)

    p1, p2 = _patched([unlinked])
    with p1, p2:
        process_bank_sync(actual, audit, cfg, state)

    actual.run_bank_sync.assert_not_called()


def test_first_sync_guard_skips_zero_transaction_account_by_default(tmp_path: Path):
    acct = make_account()
    actual = make_actual()
    audit = MagicMock()
    cfg = make_cfg(bank_sync_allow_first_sync=False)
    state = _state(tmp_path)

    p1, p2 = _patched([acct], transactions_by_account={acct.id: []})
    with p1, p2:
        process_bank_sync(actual, audit, cfg, state)

    actual.run_bank_sync.assert_not_called()


def test_first_sync_guard_allows_when_configured(tmp_path: Path):
    acct = make_account()
    actual = make_actual()
    audit = MagicMock()
    cfg = make_cfg(bank_sync_allow_first_sync=True)
    state = _state(tmp_path)

    p1, p2 = _patched([acct], transactions_by_account={acct.id: []})
    with p1, p2:
        process_bank_sync(actual, audit, cfg, state)

    actual.run_bank_sync.assert_called_once()


def test_daily_cap_blocks_a_call_the_interval_would_have_allowed(tmp_path: Path):
    acct = make_account()
    actual = make_actual()
    audit = MagicMock()
    cfg = make_cfg(bank_sync_max_runs_per_day=1)
    state = _state(tmp_path)
    # Already synced once today, at the cap.
    state.record_account_run("bank_sync", acct.id, NOW - timedelta(hours=1), "ok")

    p1, p2 = _patched([acct])
    with p1, p2:
        result = process_bank_sync(actual, audit, cfg, state, now=NOW)

    assert result == 0
    actual.run_bank_sync.assert_not_called()
    reasons = [c.args[0]["reason"] for c in audit._write.call_args_list if "reason" in c.args[0]]
    assert "daily_cap" in reasons


def test_state_persisted_even_when_account_failed(tmp_path: Path):
    acct = make_account()
    error = ActualBankSyncError("ITEM_LOGIN_REQUIRED", "rejected", "expired credentials")
    actual = make_actual(run_bank_sync_side_effect=[error])
    audit = MagicMock()
    cfg = make_cfg()
    path = tmp_path / "state.json"
    state = SyncState(str(path))

    p1, p2 = _patched([acct])
    with p1, p2:
        process_bank_sync(actual, audit, cfg, state)

    assert path.exists()
    reloaded = SyncState(str(path))
    assert reloaded.last_run("bank_sync") is not None
    today = datetime.now(UTC).date().isoformat()
    assert reloaded.runs_today("bank_sync", acct.id, today) == 1


def test_unconfigured_provider_is_skipped(tmp_path: Path):
    acct = make_account()
    actual = make_actual(configured=False)
    audit = MagicMock()
    cfg = make_cfg()
    state = _state(tmp_path)

    p1, p2 = _patched([acct])
    with p1, p2:
        result = process_bank_sync(actual, audit, cfg, state)

    assert result == 0
    actual.run_bank_sync.assert_not_called()
