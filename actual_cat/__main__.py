import os
import sys
from typing import Any

from actual import Actual
from actual.queries import get_ruleset

from . import history, prompts
from .audit import AuditLogger
from .bank_sync import process_bank_sync
from .categorization import find_uncategorized, process_categorization
from .config import load_config
from .duplicates import process_duplicates
from .llm import LLMClient
from .receipts.ingest import poll_email
from .receipts.match import process_receipt_splits
from .receipts.process import process_inbox
from .schema import build_schema_text
from .state import SyncState
from .transfers import process_transfers


def main() -> None:
    cfg = load_config()

    if cfg.ca_bundle:
        os.environ["REQUESTS_CA_BUNDLE"] = cfg.ca_bundle
        os.environ["SSL_CERT_FILE"] = cfg.ca_bundle

    llm = LLMClient(
        cfg.llm_text,
        cfg.llm_vision,
        requests_per_minute=cfg.llm_requests_per_minute,
        max_retries=cfg.llm_max_retries,
    )
    audit = AuditLogger(cfg.audit_log_path)
    state = SyncState(cfg.state_path)

    kwargs: dict[str, Any] = dict(
        base_url=cfg.base_url,
        password=cfg.password,
        file=cfg.file,
    )
    if cfg.encryption_password:
        kwargs["encryption_password"] = cfg.encryption_password

    try:
        with Actual(**kwargs) as actual:
            # 0. Scheduled bank sync — before the rule engine, so freshly imported
            #    rows are processed in this same run instead of waiting a full tick.
            #    process_bank_sync() re-checks cfg.bank_sync_enabled itself; the
            #    if/else here just keeps the skip-audit shape consistent with the
            #    other optional pipelines below.
            try:
                if cfg.bank_sync_enabled:
                    process_bank_sync(actual, audit, cfg, state)
                    actual.commit()
                else:
                    audit.log_skipped_pipeline("bank_sync")
            except Exception as e:
                print(f"WARNING: bank sync failed ({e})", file=sys.stderr)
                audit._write({
                    "event": "bank_sync_failed", "pipeline": "bank_sync", "error": str(e)
                })

            # 1. Run Actual's built-in rule engine first.
            # Guard against malformed rules (e.g. empty category ID) failing
            # pydantic validation — the server runs rules on sync anyway.
            try:
                ruleset = get_ruleset(actual.session)
                for txn in find_uncategorized(actual.session, cfg.duplicates_defer_pending):
                    ruleset.run(txn)
                actual.commit()
            except Exception as e:
                print(f"WARNING: rule engine skipped ({e})", file=sys.stderr)

            # 1.5. Pending-duplicate resolution. Runs before every LLM pipeline so
            #      nothing is spent on — or written to — a row about to be deleted.
            if cfg.duplicates_enabled:
                process_duplicates(actual, llm, audit, cfg, prompts)
                actual.commit()
            else:
                audit.log_skipped_pipeline("duplicates")

            # 2. Transfer detection (before categorization so transfers aren't miscategorized)
            process_transfers(actual, llm, audit, cfg, prompts)
            actual.commit()

            # 3. IMAP email poll → inbox (before OCR so new emails are OCR'd this run)
            if cfg.receipts_enabled and cfg.email_enabled:
                try:
                    poll_email(cfg, audit)
                except Exception as e:
                    print(f"WARNING: email poll failed ({e})", file=sys.stderr)

            # 4. Receipt OCR — process any newly received images in inbox.
            #    Pass 1 (extract_receipt) transcribes line items; pass 2
            #    (categorize_line_items) assigns each item a category using the
            #    item-description history before the result is persisted.
            schema_text = build_schema_text(actual.session)
            if cfg.receipts_enabled and cfg.history_enabled:
                _path_map = history.build_category_path_map(actual.session)
                _, item_history = history.build_histories(actual.session, _path_map)
            else:
                item_history = {}
            if cfg.receipts_enabled:
                process_inbox(llm, audit, cfg, prompts, schema_text, item_history)

            # 5. Receipt split matching — match pending receipts to bank transactions
            if cfg.receipts_enabled:
                process_receipt_splits(actual, llm, audit, cfg, schema_text, prompts)
                actual.commit()
            else:
                audit.log_skipped_pipeline("receipt")

            # 6. Categorize remaining uncategorized transactions
            process_categorization(actual, llm, audit, cfg, schema_text, prompts)
            actual.commit()

            audit.log_run_complete()

    except Exception as e:
        print(f"FATAL: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
