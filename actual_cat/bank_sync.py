"""Scheduled bank sync — imports fresh transactions before every other pipeline runs.

Owning the sync inside the worker collapses two uncoordinated schedules (actual-
server's own pull cadence and the worker's hourly tick) into one ordering: import,
then rules, then every LLM pipeline, on rows guaranteed fresh for this run.

actualpy's own Actual.run_bank_sync(account=None, ...) iterates every account in
one call with no per-account failure isolation: one account raising
ActualBankSyncError aborts every account after it in the list (actual/__init__.py,
run_bank_sync / _run_bank_sync_account). This module works around that by calling
run_bank_sync(account=...) once per account, each in its own try/except, which also
gives per-account state — what per-account provider quotas want anyway.

Off by default (`[bank_sync] enabled = false`); this module is only ever reached
from __main__.py when the pipeline is enabled, but process_bank_sync() re-checks
`enabled` itself so it stays a self-contained, independently testable unit.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from actual.exceptions import ActualBankSyncError
from actual.queries import get_accounts, get_transactions

from .state import due, under_daily_cap

if TYPE_CHECKING:
    from .audit import AuditLogger
    from .config import Config
    from .state import SyncState

PIPELINE = "bank_sync"


def _select_accounts(actual: Any, cfg: "Config") -> list[Any]:
    """Sync-enabled accounts, filtered by the allow/deny list.

    An account needs both account_sync_source and account_id to be a sync
    candidate at all — matching the check run_bank_sync() itself makes.
    """
    accounts = get_accounts(actual.session)
    candidates = [a for a in accounts if a.account_sync_source and a.account_id]
    if cfg.bank_sync_accounts:
        allow = set(cfg.bank_sync_accounts)
        candidates = [a for a in candidates if a.name in allow]
    if cfg.bank_sync_exclude_accounts:
        deny = set(cfg.bank_sync_exclude_accounts)
        candidates = [a for a in candidates if a.name not in deny]
    return candidates


def _is_first_sync(actual: Any, acct: Any) -> bool:
    return len(get_transactions(actual.session, account=acct)) == 0


def process_bank_sync(
    actual: Any, audit: "AuditLogger", cfg: "Config", state: "SyncState"
) -> int:
    """Sync every eligible account, isolating failures per account.

    Returns the number of transactions imported. Persists state after the
    loop even on partial failure, so an account that errored isn't retried
    on the next tick just because a later account in the list succeeded.
    """
    now = datetime.now(timezone.utc)

    if not cfg.bank_sync_enabled:
        audit._write({"event": "bank_sync_skipped", "pipeline": PIPELINE, "reason": "disabled"})
        return 0

    last_run = state.last_run(PIPELINE)
    if not due(last_run, cfg.bank_sync_interval_minutes, cfg.bank_sync_grace_minutes, now):
        audit._write({"event": "bank_sync_skipped", "pipeline": PIPELINE, "reason": "interval"})
        return 0

    accounts = _select_accounts(actual, cfg)
    today = now.date().isoformat()
    due_accounts = [
        a
        for a in accounts
        if under_daily_cap(state.runs_today(PIPELINE, a.id, today), cfg.bank_sync_max_runs_per_day)
    ]
    if accounts and not due_accounts:
        audit._write({"event": "bank_sync_skipped", "pipeline": PIPELINE, "reason": "daily_cap"})
        return 0

    start_date = (
        now.date() - timedelta(days=cfg.bank_sync_lookback_days)
        if cfg.bank_sync_lookback_days > 0
        else None
    )

    status_cache: dict[str, bool] = {}
    imported_count = 0
    accounts_synced = 0
    per_account: dict[str, int] = {}

    for acct in due_accounts:
        method = acct.account_sync_source.lower()
        if method not in status_cache:
            status_cache[method] = actual.bank_sync_status(method).data.configured
        if not status_cache[method]:
            continue

        if not cfg.bank_sync_allow_first_sync and _is_first_sync(actual, acct):
            continue

        try:
            imported = actual.run_bank_sync(account=acct, start_date=start_date)
        except ActualBankSyncError as e:
            audit._write({
                "event": "bank_sync_account_failed",
                "pipeline": PIPELINE,
                "account": acct.name,
                "error_type": e.error_type,
                "status": e.status,
                "reason": e.reason,
            })
            state.record_account_run(PIPELINE, acct.id, now, "error")
            continue

        count = len(imported)
        imported_count += count
        accounts_synced += 1
        per_account[acct.name] = count
        state.record_account_run(PIPELINE, acct.id, now, "ok")

    state.set_last_run(PIPELINE, now)
    state.save()

    audit._write({
        "event": "bank_sync_ok",
        "pipeline": PIPELINE,
        "accounts_synced": accounts_synced,
        "imported_count": imported_count,
        "per_account": per_account,
    })
    return imported_count
