"""Durable run-state for pipelines that need a schedule, not just idempotency.

Kept separate from bank_sync.py because the gate (due-with-grace, daily cap)
is generic: any pipeline that wants its own cadence — receipts every 15
minutes, categorization hourly — reuses this against the same file instead of
a second implementation of the interval math.

A single JSON file, written atomically (tmp file + os.replace). Missing or
corrupt state degrades to "never ran" rather than raising — a first run or a
torn write should not crash the worker.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


class SyncState:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self._data = self._load()

    def _load(self) -> dict[str, Any]:
        try:
            with open(self.path) as f:
                data = json.load(f)
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with open(tmp, "w") as f:
            json.dump(self._data, f, indent=2)
        os.replace(tmp, self.path)

    def _account(self, pipeline: str, account_id: str) -> dict[str, Any]:
        accounts = self._data.setdefault(pipeline, {}).setdefault("accounts", {})
        entry: dict[str, Any] = accounts.setdefault(account_id, {})
        return entry

    def last_run(self, pipeline: str) -> str | None:
        run: str | None = self._data.get(pipeline, {}).get("last_run")
        return run

    def set_last_run(self, pipeline: str, when: datetime) -> None:
        self._data.setdefault(pipeline, {})["last_run"] = when.isoformat()

    def runs_today(self, pipeline: str, account_id: str, today: str) -> int:
        entry = self._account(pipeline, account_id)
        if entry.get("runs_today_date") != today:
            return 0
        count: int = entry.get("runs_today", 0)
        return count

    def record_account_run(
        self, pipeline: str, account_id: str, when: datetime, status: str
    ) -> None:
        today = when.date().isoformat()
        entry = self._account(pipeline, account_id)
        prior = entry.get("runs_today", 0) if entry.get("runs_today_date") == today else 0
        entry["last_run"] = when.isoformat()
        entry["last_status"] = status
        entry["runs_today_date"] = today
        entry["runs_today"] = prior + 1


def due(
    last_run: str | None,
    interval_minutes: int,
    grace_minutes: int,
    now: datetime | None = None,
) -> bool:
    """True when enough wall-clock time has passed, absorbing timer jitter.

    A strict `now - last_run >= interval` comparison drifts under a jittered
    timer (e.g. systemd's RandomizedDelaySec): a run landing a few minutes
    early evaluates as not-yet-due and slips to the next tick, then keeps
    slipping. The grace window makes "close enough" count as due.
    """
    if last_run is None:
        return True
    now = now or datetime.now(timezone.utc)
    try:
        last = datetime.fromisoformat(last_run)
    except ValueError:
        return True
    interval = timedelta(minutes=interval_minutes)
    grace = timedelta(minutes=grace_minutes)
    return now >= last + interval - grace


def under_daily_cap(runs_today: int, max_runs_per_day: int) -> bool:
    """max_runs_per_day == 0 means uncapped."""
    if max_runs_per_day == 0:
        return True
    return runs_today < max_runs_per_day
