import os
import sys
from typing import Any

from actual import Actual
from actual.queries import get_ruleset

from . import history, prompts
from .audit import AuditLogger
from .categorization import find_uncategorized, process_categorization
from .config import load_config
from .llm import LLMClient
from .receipts import store as receipt_store
from .receipts.categorize import categorize_line_items
from .receipts.extract import extract_receipt
from .receipts.ingest import poll_email
from .receipts.match import process_receipt_splits
from .receipts.parse import resolve_receipt_date
from .schema import build_schema_text
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

    kwargs: dict[str, Any] = dict(
        base_url=cfg.base_url,
        password=cfg.password,
        file=cfg.file,
    )
    if cfg.encryption_password:
        kwargs["encryption_password"] = cfg.encryption_password

    try:
        with Actual(**kwargs) as actual:
            # 1. Run Actual's built-in rule engine first.
            # Guard against malformed rules (e.g. empty category ID) failing
            # pydantic validation — the server runs rules on sync anyway.
            try:
                ruleset = get_ruleset(actual.session)
                for txn in find_uncategorized(actual.session):
                    ruleset.run(txn)
                actual.commit()
            except Exception as e:
                print(f"WARNING: rule engine skipped ({e})", file=sys.stderr)

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
                for inbox_meta in receipt_store.list_inbox(cfg.receipts_store_path):
                    receipt_id = inbox_meta["id"]
                    try:
                        result = extract_receipt(inbox_meta, llm, prompts, schema_text)
                        if "error" in result:
                            receipt_store.save_failed(
                                cfg.receipts_store_path, receipt_id, result["error"]
                            )
                            audit._write({"event": "receipt_ocr_failed", "pipeline": "receipt",
                                          "receipt_id": receipt_id, "error": result["error"]})
                        else:
                            # Resolve the raw date using location-derived format + received_ts cross-check
                            iso_date, was_ambiguous = resolve_receipt_date(
                                result.get("date_raw"),
                                result.get("location_raw"),
                                inbox_meta["received_ts"],
                            )
                            result["date"] = iso_date
                            if was_ambiguous:
                                audit._write({
                                    "event": "receipt_date_ambiguous",
                                    "pipeline": "receipt",
                                    "receipt_id": receipt_id,
                                    "resolved_date": iso_date,
                                    "location_raw": result.get("location_raw"),
                                })
                            result = categorize_line_items(
                                result, llm, prompts, schema_text, item_history, cfg
                            )
                            receipt_store.save_pending(cfg.receipts_store_path, receipt_id, result)
                            audit._write({"event": "receipt_ocr_ok", "pipeline": "receipt",
                                          "receipt_id": receipt_id,
                                          "merchant": result.get("merchant"),
                                          "total_cents": result.get("total_cents")})
                    except Exception as e:
                        receipt_store.save_failed(cfg.receipts_store_path, receipt_id, str(e))
                        audit._write({"event": "receipt_ocr_failed", "pipeline": "receipt",
                                      "receipt_id": receipt_id, "error": str(e)})

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
