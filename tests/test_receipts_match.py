"""Unit tests for receipt match pipeline."""

from unittest.mock import MagicMock, patch

from actual_cat.receipts.match import find_receipt_match, process_receipt_splits


def make_txn(id="txn-1", amount=-1000, acct="acct-checking", date_int=20260601,
             transferred_id=None, tombstone=0, notes=None, is_parent=0,
             category_id=None, account_name="Checking", offbudget=False,
             pending=False):
    txn = MagicMock()
    txn.id = id
    # Bank-sync fields the defer_pending filter reads.
    txn.cleared = 0 if pending else 1
    txn.raw_synced_data = '{"booked": false}' if pending else None
    txn.amount = amount
    txn.acct = acct
    txn.date = date_int
    txn.transferred_id = transferred_id
    txn.tombstone = tombstone
    txn.notes = notes
    txn.is_parent = is_parent
    txn.category_id = category_id
    txn.account = MagicMock()
    txn.account.name = account_name
    txn.account.offbudget = offbudget
    txn.payee = None
    txn.imported_description = None
    return txn


def make_receipt_meta(total_cents=1000, date="2026-06-01", receipt_id="rcpt-1",
                      received_ts="2026-06-01T12:00:00+00:00"):
    return {
        "id": receipt_id,
        "status": "pending",
        "received_ts": received_ts,
        "ocr": {
            "merchant": "Target",
            "date": date,
            "total_cents": total_cents,
            "line_items": [
                {"description": "Groceries", "amount_cents": total_cents,
                 "category": "Food / Groceries"}
            ],
        },
    }


class TestFindReceiptMatch:
    def _run(self, candidates, receipt_meta=None, defer_pending=False):
        if receipt_meta is None:
            receipt_meta = make_receipt_meta()
        session = MagicMock()

        query_chain = MagicMock()
        query_chain.filter.return_value = query_chain
        query_chain.all.return_value = candidates
        session.query.return_value = query_chain

        return find_receipt_match(
            session, receipt_meta, window_days=3, defer_pending=defer_pending
        )

    def test_pending_txn_matched_by_default(self):
        txn = make_txn(amount=-1000, pending=True)
        assert self._run([txn]) == [txn]

    def test_pending_txn_excluded_when_deferred(self):
        # The split would be computed against the pre-tip authorization total.
        txn = make_txn(amount=-1000, pending=True)
        assert self._run([txn], defer_pending=True) == []

    def test_zero_total_returns_empty(self):
        meta = make_receipt_meta(total_cents=0)
        result = self._run([], receipt_meta=meta)
        assert result == []

    def test_missing_date_returns_empty(self):
        meta = make_receipt_meta()
        meta["ocr"]["date"] = None
        result = self._run([], receipt_meta=meta)
        assert result == []

    def test_invalid_date_returns_empty(self):
        meta = make_receipt_meta()
        meta["ocr"]["date"] = "not-a-date"
        result = self._run([], receipt_meta=meta)
        assert result == []

    def test_offbudget_txn_excluded(self):
        txn = make_txn(amount=-1000, offbudget=True)
        result = self._run([txn])
        assert result == []

    def test_suggested_split_txn_eligible_for_retry(self):
        # Transactions with only #ai-suggested-split remain eligible —
        # a better photo can finalize the split (hybrid retry).
        txn = make_txn(amount=-1000, notes="#ai-suggested-split some memo")
        result = self._run([txn])
        assert result == [txn]

    def test_receipt_split_applied_excluded(self):
        txn = make_txn(amount=-1000, notes="#ai-receipt-split some memo")
        result = self._run([txn])
        assert result == []

    def test_matching_txn_returned(self):
        txn = make_txn(amount=-1000)
        result = self._run([txn])
        assert result == [txn]

    def test_multiple_candidates_all_returned(self):
        txns = [make_txn(id=f"t{i}", amount=-1000) for i in range(3)]
        result = self._run(txns)
        assert len(result) == 3


class TestProcessReceiptSplits:
    def _make_cfg(self, mode="suggest", threshold="high", store_path="/tmp/receipts",
                  window_days=3, expiry_days=30):
        cfg = MagicMock()
        cfg.receipts_mode = mode
        cfg.receipts_threshold = threshold
        cfg.receipts_store_path = store_path
        cfg.receipts_match_window_days = window_days
        cfg.receipts_expiry_days = expiry_days
        cfg.duplicates_defer_pending = False
        return cfg

    def test_suggest_mode_tags_matched_txn(self, tmp_path):
        import actual_cat.prompts as prompts
        from actual_cat.receipts import store as receipt_store

        store_path = str(tmp_path / "receipts")
        rid = receipt_store.save_received(store_path, tmp_path / "img.jpg", "ios")
        ocr = {
            "merchant": "Target", "date": "2026-06-07", "total_cents": 1000,
            "line_items": [
                {"description": "x", "amount_cents": 1000, "category": "Food / Groceries"}
            ],
        }
        receipt_store.save_pending(store_path, rid, ocr)

        txn = make_txn(id="txn-1", amount=-1000)
        actual = MagicMock()
        audit = MagicMock()
        audit._write = MagicMock()
        cfg = self._make_cfg(store_path=store_path)

        with patch("actual_cat.receipts.match.find_receipt_match", return_value=[txn]):
            process_receipt_splits(actual, MagicMock(), audit, cfg, "schema", prompts)

        assert "#ai-suggested-split" in txn.notes
        # check done record exists
        import json
        from pathlib import Path
        done_files = list((Path(store_path) / "done").glob("*.json"))
        assert len(done_files) == 1
        meta = json.loads(done_files[0].read_text())
        assert meta["status"] == "suggested"
        assert meta["matched_txn_id"] == "txn-1"

    def test_no_match_leaves_pending(self, tmp_path):
        import actual_cat.prompts as prompts
        from actual_cat.receipts import store as receipt_store

        store_path = str(tmp_path / "receipts")
        rid = receipt_store.save_received(store_path, tmp_path / "img.jpg", "ios")
        ocr = {
            "merchant": "Target", "date": "2026-06-07", "total_cents": 1000,
            "line_items": [
                {"description": "x", "amount_cents": 1000, "category": "Food / Groceries"}
            ],
        }
        receipt_store.save_pending(store_path, rid, ocr)

        actual = MagicMock()
        audit = MagicMock()
        audit._write = MagicMock()
        cfg = self._make_cfg(store_path=store_path)

        with patch("actual_cat.receipts.match.find_receipt_match", return_value=[]):
            process_receipt_splits(actual, MagicMock(), audit, cfg, "schema", prompts)

        from pathlib import Path
        assert len(list((Path(store_path) / "done").glob("*.json"))) == 0
        assert len(list((Path(store_path) / "pending").glob("*.json"))) == 1

    def test_ambiguous_match_leaves_pending(self, tmp_path):
        import actual_cat.prompts as prompts
        from actual_cat.receipts import store as receipt_store

        store_path = str(tmp_path / "receipts")
        rid = receipt_store.save_received(store_path, tmp_path / "img.jpg", "ios")
        ocr = {
            "merchant": "Target", "date": "2026-06-07", "total_cents": 1000,
            "line_items": [
                {"description": "x", "amount_cents": 1000, "category": "Food / Groceries"}
            ],
        }
        receipt_store.save_pending(store_path, rid, ocr)

        txns = [make_txn(id=f"t{i}", amount=-1000) for i in range(2)]
        actual = MagicMock()
        audit = MagicMock()
        audit._write = MagicMock()
        cfg = self._make_cfg(store_path=store_path)

        with patch("actual_cat.receipts.match.find_receipt_match", return_value=txns):
            process_receipt_splits(actual, MagicMock(), audit, cfg, "schema", prompts)

        from pathlib import Path
        assert len(list((Path(store_path) / "done").glob("*.json"))) == 0

    def test_expired_receipt_moved_to_done(self, tmp_path):
        import actual_cat.prompts as prompts
        from actual_cat.receipts import store as receipt_store

        store_path = str(tmp_path / "receipts")
        rid = receipt_store.save_received(store_path, tmp_path / "img.jpg", "ios")
        ocr = {
            "merchant": "Target", "date": "2026-01-01", "total_cents": 1000,
            "line_items": [
                {"description": "x", "amount_cents": 1000, "category": "Food / Groceries"}
            ],
        }
        receipt_store.save_pending(store_path, rid, ocr)
        # backdate the received_ts to trigger expiry
        import json
        from pathlib import Path
        pending_file = Path(store_path) / "pending" / f"{rid}.json"
        meta = json.loads(pending_file.read_text())
        meta["received_ts"] = "2025-01-01T00:00:00+00:00"
        pending_file.write_text(json.dumps(meta))

        actual = MagicMock()
        audit = MagicMock()
        audit._write = MagicMock()
        cfg = self._make_cfg(store_path=store_path, expiry_days=30)

        with patch("actual_cat.receipts.match.find_receipt_match", return_value=[]):
            process_receipt_splits(actual, MagicMock(), audit, cfg, "schema", prompts)

        from pathlib import Path
        done_files = list((Path(store_path) / "done").glob("*.json"))
        assert len(done_files) == 1
        meta = json.loads(done_files[0].read_text())
        assert meta["status"] == "expired"

    def test_apply_mode_low_confidence_falls_back_to_suggest(self, tmp_path):
        import actual_cat.prompts as prompts
        from actual_cat.receipts import store as receipt_store

        store_path = str(tmp_path / "receipts")
        rid = receipt_store.save_received(store_path, tmp_path / "img.jpg", "ios")
        ocr = {
            "merchant": "Target", "date": "2026-06-07", "total_cents": 1000,
            "confidence": "low",  # OCR unreliable — should not apply
            "line_items": [
                {"description": "x", "amount_cents": 1000, "category": "Food / Groceries"}
            ],
        }
        receipt_store.save_pending(store_path, rid, ocr)

        txn = make_txn(id="txn-1", amount=-1000)
        actual = MagicMock()
        audit = MagicMock()
        audit._write = MagicMock()
        cfg = self._make_cfg(mode="apply", store_path=store_path)

        with patch("actual_cat.receipts.match.find_receipt_match", return_value=[txn]):
            process_receipt_splits(actual, MagicMock(), audit, cfg, "schema", prompts)

        # Should tag as suggested, not apply split rows
        assert "#ai-suggested-split" in txn.notes
        assert txn.is_parent != 1  # split not applied

    def test_apply_mode_high_confidence_applies(self, tmp_path):
        import actual_cat.prompts as prompts
        from actual_cat.receipts import store as receipt_store

        store_path = str(tmp_path / "receipts")
        rid = receipt_store.save_received(store_path, tmp_path / "img.jpg", "ios")
        ocr = {
            "merchant": "Target", "date": "2026-06-07", "total_cents": 1000,
            "confidence": "high",
            "line_items": [
                {"description": "x", "amount_cents": 1000, "category": "Food / Groceries"}
            ],
        }
        receipt_store.save_pending(store_path, rid, ocr)

        txn = make_txn(id="txn-1", amount=-1000)
        actual = MagicMock()
        audit = MagicMock()
        audit._write = MagicMock()
        cfg = self._make_cfg(mode="apply", store_path=store_path)

        with patch("actual_cat.receipts.match.find_receipt_match", return_value=[txn]):
            with patch("actual_cat.receipts.match._apply_splits") as mock_apply:
                process_receipt_splits(actual, MagicMock(), audit, cfg, "schema", prompts)
                mock_apply.assert_called_once()

    def test_suggest_mode_strips_prior_category_tag(self, tmp_path):
        import actual_cat.prompts as prompts
        from actual_cat.receipts import store as receipt_store

        store_path = str(tmp_path / "receipts")
        rid = receipt_store.save_received(store_path, tmp_path / "img.jpg", "ios")
        ocr = {
            "merchant": "Target", "date": "2026-06-07", "total_cents": 1000,
            "line_items": [
                {"description": "x", "amount_cents": 1000, "category": "Food / Groceries"}
            ],
        }
        receipt_store.save_pending(store_path, rid, ocr)

        txn = make_txn(id="txn-1", amount=-1000, notes="#ai:food/groceries some memo")
        actual = MagicMock()
        audit = MagicMock()
        audit._write = MagicMock()
        cfg = self._make_cfg(store_path=store_path)

        with patch("actual_cat.receipts.match.find_receipt_match", return_value=[txn]):
            process_receipt_splits(actual, MagicMock(), audit, cfg, "schema", prompts)

        assert "#ai:food/groceries" not in txn.notes
        assert "#ai-suggested-split" in txn.notes
        assert "some memo" in txn.notes
