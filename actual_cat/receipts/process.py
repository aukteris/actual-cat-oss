"""Inbox OCR pass — turn newly received receipts into pending ones.

Extracted from `__main__` so the attempt cap and give-up path are testable
without a live budget. One receipt failing here must never stop the ones behind
it, nor the pipeline steps that run after receipts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from . import store as receipt_store
from .categorize import categorize_line_items
from .extract import extract_receipt
from .parse import resolve_receipt_date

if TYPE_CHECKING:
    from ..audit import AuditLogger
    from ..config import Config
    from ..llm import LLMClient


def process_inbox(
    llm: "LLMClient",
    audit: "AuditLogger",
    cfg: "Config",
    prompts: Any,
    schema_text: str,
    item_history: dict[str, Any],
) -> None:
    """OCR every receipt sitting in the inbox, moving each to pending or failed.

    Pass 1 (extract_receipt) transcribes line items; pass 2 (categorize_line_items)
    assigns each item a category using the item-description history before the
    result is persisted.
    """
    for inbox_meta in receipt_store.list_inbox(cfg.receipts_store_path):
        receipt_id = inbox_meta["id"]
        try:
            # Count the attempt BEFORE trying it. If OCR runs long enough that the
            # service manager SIGTERMs the run, nothing after this line gets
            # written — a counter kept on the success path would never advance, and
            # one slow receipt would re-wedge every subsequent run forever.
            attempts = receipt_store.record_ocr_attempt(cfg.receipts_store_path, receipt_id)
            max_attempts = cfg.receipts_max_ocr_attempts
            if max_attempts and attempts > max_attempts:
                error = f"OCR gave up after {max_attempts} attempts"
                receipt_store.save_failed(cfg.receipts_store_path, receipt_id, error)
                audit._write({"event": "receipt_ocr_failed", "pipeline": "receipt",
                              "receipt_id": receipt_id, "error": error,
                              "attempts": max_attempts})
                continue

            result = extract_receipt(
                inbox_meta, llm, prompts, schema_text,
                request_timeout_seconds=cfg.receipts_ocr_request_timeout_seconds,
                budget_seconds=cfg.receipts_ocr_budget_seconds,
            )
            if "error" in result:
                receipt_store.save_failed(cfg.receipts_store_path, receipt_id, result["error"])
                audit._write({"event": "receipt_ocr_failed", "pipeline": "receipt",
                              "receipt_id": receipt_id, "error": result["error"]})
            else:
                # Resolve the raw date using location-derived format
                # + received_ts cross-check
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
