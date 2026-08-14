"""Unit tests for the bank-sync run-state gate: due(), under_daily_cap(), SyncState."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

from actual_cat.state import SyncState, due, under_daily_cap

UTC = timezone.utc


# ---------------------------------------------------------------------------
# due() — grace window
# ---------------------------------------------------------------------------


def test_due_never_ran():
    assert due(None, interval_minutes=360, grace_minutes=5) is True


def test_due_well_before_interval_is_not_due():
    now = datetime(2026, 8, 14, 12, 0, tzinfo=UTC)
    last_run = (now - timedelta(hours=1)).isoformat()
    assert due(last_run, interval_minutes=360, grace_minutes=5, now=now) is False


def test_due_exactly_on_interval():
    now = datetime(2026, 8, 14, 12, 0, tzinfo=UTC)
    last_run = (now - timedelta(minutes=360)).isoformat()
    assert due(last_run, interval_minutes=360, grace_minutes=5, now=now) is True


def test_due_within_grace_of_interval():
    """The 58-minute case from the design doc: a jittered timer lands a run a
    couple minutes early relative to the strict interval, and the grace window
    is what keeps that from being treated as not-yet-due."""
    now = datetime(2026, 8, 14, 12, 0, tzinfo=UTC)
    last_run = (now - timedelta(minutes=358)).isoformat()  # 2 min short of 360
    assert due(last_run, interval_minutes=360, grace_minutes=5, now=now) is True


def test_due_beyond_grace_is_not_due():
    now = datetime(2026, 8, 14, 12, 0, tzinfo=UTC)
    last_run = (now - timedelta(minutes=350)).isoformat()  # 10 min short of 360
    assert due(last_run, interval_minutes=360, grace_minutes=5, now=now) is False


def test_due_corrupt_timestamp_degrades_to_never_ran():
    now = datetime(2026, 8, 14, 12, 0, tzinfo=UTC)
    assert due("not-a-timestamp", interval_minutes=360, grace_minutes=5, now=now) is True


# ---------------------------------------------------------------------------
# under_daily_cap()
# ---------------------------------------------------------------------------


def test_under_daily_cap_uncapped_when_zero():
    assert under_daily_cap(runs_today=1000, max_runs_per_day=0) is True


def test_under_daily_cap_blocks_at_the_limit():
    assert under_daily_cap(runs_today=4, max_runs_per_day=4) is False


def test_under_daily_cap_allows_below_the_limit():
    assert under_daily_cap(runs_today=3, max_runs_per_day=4) is True


# ---------------------------------------------------------------------------
# SyncState — load/save, missing/corrupt file, atomic write
# ---------------------------------------------------------------------------


def test_missing_file_degrades_to_never_ran(tmp_path: Path):
    state = SyncState(str(tmp_path / "does-not-exist.json"))
    assert state.last_run("bank_sync") is None


def test_corrupt_file_degrades_to_never_ran(tmp_path: Path):
    path = tmp_path / "state.json"
    path.write_text("{not valid json")
    state = SyncState(str(path))
    assert state.last_run("bank_sync") is None


def test_save_then_reload_round_trips(tmp_path: Path):
    path = tmp_path / "state.json"
    now = datetime(2026, 8, 14, 6, 0, 11, tzinfo=UTC)

    state = SyncState(str(path))
    state.set_last_run("bank_sync", now)
    state.record_account_run("bank_sync", "acct-1", now, "ok")
    state.save()

    reloaded = SyncState(str(path))
    assert reloaded.last_run("bank_sync") == now.isoformat()
    assert reloaded.runs_today("bank_sync", "acct-1", now.date().isoformat()) == 1


def test_atomic_write_leaves_no_partial_file(tmp_path: Path):
    path = tmp_path / "state.json"
    state = SyncState(str(path))
    state.set_last_run("bank_sync", datetime.now(UTC))
    state.save()

    assert path.exists()
    assert not path.with_suffix(path.suffix + ".tmp").exists()


def test_runs_today_resets_on_utc_date_rollover(tmp_path: Path):
    path = tmp_path / "state.json"
    state = SyncState(str(path))
    day1 = datetime(2026, 8, 14, 23, 0, tzinfo=UTC)
    day2 = datetime(2026, 8, 15, 1, 0, tzinfo=UTC)

    state.record_account_run("bank_sync", "acct-1", day1, "ok")
    state.record_account_run("bank_sync", "acct-1", day1, "ok")
    assert state.runs_today("bank_sync", "acct-1", day1.date().isoformat()) == 2

    # A new UTC date should not inherit yesterday's count.
    assert state.runs_today("bank_sync", "acct-1", day2.date().isoformat()) == 0
    state.record_account_run("bank_sync", "acct-1", day2, "ok")
    assert state.runs_today("bank_sync", "acct-1", day2.date().isoformat()) == 1
