"""Unit tests for receipt store — status transitions and file layout."""

import json
from pathlib import Path

import pytest

from actual_cat.receipts.store import (
    list_inbox,
    list_pending,
    record_ocr_attempt,
    save_done,
    save_failed,
    save_pending,
    save_received,
)


@pytest.fixture
def store(tmp_path):
    return str(tmp_path / "receipts")


class TestSaveReceived:
    def test_creates_inbox_json(self, store):
        receipt_id = save_received(store, Path("/tmp/img.jpg"), "ios")
        meta = json.loads((Path(store) / "inbox" / f"{receipt_id}.json").read_text())
        assert meta["id"] == receipt_id
        assert meta["status"] == "received"
        assert meta["source"] == "ios"
        assert meta["image_path"] == "/tmp/img.jpg"

    def test_each_call_generates_unique_id(self, store):
        id1 = save_received(store, Path("/tmp/a.jpg"), "ios")
        id2 = save_received(store, Path("/tmp/b.jpg"), "ios")
        assert id1 != id2


class TestSavePending:
    def test_moves_from_inbox_to_pending(self, store):
        receipt_id = save_received(store, Path("/tmp/img.jpg"), "ios")
        ocr = {"merchant": "Target", "date": "2026-06-01", "total_cents": 1000, "line_items": []}
        save_pending(store, receipt_id, ocr)
        assert not (Path(store) / "inbox" / f"{receipt_id}.json").exists()
        meta = json.loads((Path(store) / "pending" / f"{receipt_id}.json").read_text())
        assert meta["status"] == "pending"
        assert meta["ocr"]["merchant"] == "Target"


class TestSaveDone:
    def test_moves_from_pending_to_done(self, store):
        receipt_id = save_received(store, Path("/tmp/img.jpg"), "ios")
        save_pending(store, receipt_id, {"merchant": "X", "total_cents": 100, "line_items": []})
        save_done(store, receipt_id, "applied", matched_txn_id="txn-1")
        assert not (Path(store) / "pending" / f"{receipt_id}.json").exists()
        meta = json.loads((Path(store) / "done" / f"{receipt_id}.json").read_text())
        assert meta["status"] == "applied"
        assert meta["matched_txn_id"] == "txn-1"

    def test_moves_from_inbox_to_done_when_no_pending(self, store):
        receipt_id = save_received(store, Path("/tmp/img.jpg"), "ios")
        save_done(store, receipt_id, "failed")
        assert not (Path(store) / "inbox" / f"{receipt_id}.json").exists()
        meta = json.loads((Path(store) / "done" / f"{receipt_id}.json").read_text())
        assert meta["status"] == "failed"

    def test_stores_splits_and_prior_category(self, store):
        receipt_id = save_received(store, Path("/tmp/img.jpg"), "ios")
        save_pending(store, receipt_id, {"merchant": "X", "total_cents": 100, "line_items": []})
        splits = [{"description": "Groceries", "amount_cents": 100, "category": "Food / Groceries"}]
        save_done(store, receipt_id, "suggested", splits=splits, prior_category="cat-abc")
        meta = json.loads((Path(store) / "done" / f"{receipt_id}.json").read_text())
        assert meta["splits"] == splits
        assert meta["prior_category"] == "cat-abc"


class TestListInbox:
    def test_returns_all_inbox_records(self, store):
        save_received(store, Path("/tmp/a.jpg"), "ios")
        save_received(store, Path("/tmp/b.jpg"), "email")
        records = list_inbox(store)
        assert len(records) == 2

    def test_empty_when_no_receipts(self, store):
        assert list_inbox(store) == []


class TestListPending:
    def test_returns_pending_after_ocr(self, store):
        rid = save_received(store, Path("/tmp/a.jpg"), "ios")
        save_pending(store, rid, {"merchant": "X", "total_cents": 500, "line_items": []})
        records = list_pending(store)
        assert len(records) == 1
        assert records[0]["status"] == "pending"


class TestRecordOcrAttempt:
    def test_first_attempt_returns_one(self, store):
        receipt_id = save_received(store, Path("/tmp/img.jpg"), "ios")
        assert record_ocr_attempt(store, receipt_id) == 1

    def test_increments_and_persists_across_calls(self, store):
        receipt_id = save_received(store, Path("/tmp/img.jpg"), "ios")
        record_ocr_attempt(store, receipt_id)
        record_ocr_attempt(store, receipt_id)
        assert record_ocr_attempt(store, receipt_id) == 3
        meta = json.loads((Path(store) / "inbox" / f"{receipt_id}.json").read_text())
        assert meta["ocr_attempts"] == 3
        assert "last_ocr_attempt_ts" in meta

    def test_legacy_meta_without_counter_starts_at_zero(self, store):
        receipt_id = save_received(store, Path("/tmp/img.jpg"), "ios")
        inbox_json = Path(store) / "inbox" / f"{receipt_id}.json"
        meta = json.loads(inbox_json.read_text())
        meta.pop("ocr_attempts", None)  # explicit about the pre-counter meta shape
        inbox_json.write_text(json.dumps(meta))
        assert record_ocr_attempt(store, receipt_id) == 1

    def test_counter_survives_into_the_done_record(self, store):
        """The count is written before OCR runs, so it must still be there when a
        later run gives up and files the receipt as failed."""
        receipt_id = save_received(store, Path("/tmp/img.jpg"), "ios")
        record_ocr_attempt(store, receipt_id)
        record_ocr_attempt(store, receipt_id)
        save_failed(store, receipt_id, "gave up")
        meta = json.loads((Path(store) / "done" / f"{receipt_id}.json").read_text())
        assert meta["ocr_attempts"] == 2
        assert meta["status"] == "failed"
        assert meta["error"] == "gave up"
