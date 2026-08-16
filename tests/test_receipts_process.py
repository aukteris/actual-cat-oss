"""Unit tests for the inbox OCR pass — attempt cap and give-up path.

The motivating incident: a receipt whose OCR ran long enough that systemd
SIGTERMed the run. Because receipts is step 4 of 6, that killed split matching
and transaction categorization too, and the receipt stayed in the inbox to do it
again on the next run — indefinitely.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from actual_cat.receipts.process import process_inbox
from actual_cat.receipts.store import list_inbox, save_received

_PROMPTS = SimpleNamespace(
    RECEIPT_OCR_SYSTEM="ocr-system",
    RECEIPT_TEXT_SYSTEM="text-system",
    RECEIPT_ITEM_CATEGORIZE_SYSTEM="cat-system",
)


class FakeAudit:
    def __init__(self):
        self.events: list[dict] = []

    def _write(self, event: dict) -> None:
        self.events.append(event)

    def events_of(self, name: str) -> list[dict]:
        return [e for e in self.events if e["event"] == name]


class FakeLLM:
    """Vision call always fails, standing in for an unreadable/slow receipt."""

    def __init__(self, response: dict | None = None):
        self._response = response or {"error": "LLM call failure: timed out"}
        self.vision_calls = 0

    def complete_json_vision(self, system, user, image_b64, media_type, *, timeout=None) -> dict:
        self.vision_calls += 1
        return self._response

    def complete_json(self, system, user) -> dict:
        return {"line_items": []}


@pytest.fixture
def store(tmp_path):
    return str(tmp_path / "receipts")


def _cfg(store_path: str, **overrides):
    base = dict(
        receipts_store_path=store_path,
        receipts_max_ocr_attempts=3,
        receipts_ocr_request_timeout_seconds=180.0,
        receipts_ocr_budget_seconds=420.0,
        history_enabled=False,
        history_item_top_n=3,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _seed(store_path: str, tmp_path: Path) -> str:
    image = tmp_path / "r.jpg"
    image.write_bytes(b"\xff\xd8\xff" + b"\x00" * 10)
    return save_received(store_path, image, "ios")


def _run(store_path, llm, audit, **cfg_overrides):
    process_inbox(llm, audit, _cfg(store_path, **cfg_overrides), _PROMPTS, "schema", {})


def test_receipt_that_keeps_killing_the_run_is_eventually_given_up_on(store, tmp_path):
    """End-to-end shape of the incident.

    Each run is SIGTERMed mid-OCR, so the receipt is never filed and comes back
    on the next run. Only the attempt counter — written before OCR starts —
    survives, and after the cap the receipt is failed so the worker is free.
    """
    receipt_id = _seed(store, tmp_path)
    inbox_json = Path(store) / "inbox" / f"{receipt_id}.json"

    class KilledMidCall(FakeLLM):
        def complete_json_vision(self, *a, **kw):
            self.vision_calls += 1
            raise KeyboardInterrupt("SIGTERM")

    llm = KilledMidCall()
    for run in (1, 2, 3):
        with pytest.raises(KeyboardInterrupt):
            _run(store, llm, FakeAudit())
        assert json.loads(inbox_json.read_text())["ocr_attempts"] == run
        assert inbox_json.exists()  # still wedged, as it was in production

    # Fourth run: over the cap, so it's filed instead of retried.
    audit = FakeAudit()
    _run(store, llm, audit)
    assert llm.vision_calls == 3  # no fourth OCR attempt
    assert list_inbox(store) == []
    assert audit.events_of("receipt_ocr_failed")[0]["attempts"] == 3


def test_attempts_over_the_cap_skip_ocr_entirely(store, tmp_path):
    receipt_id = _seed(store, tmp_path)
    inbox_json = Path(store) / "inbox" / f"{receipt_id}.json"
    meta = json.loads(inbox_json.read_text())
    meta["ocr_attempts"] = 3  # already used its allowance
    inbox_json.write_text(json.dumps(meta))

    llm, audit = FakeLLM(), FakeAudit()
    _run(store, llm, audit)

    # The whole point: no further LLM work is done for this receipt.
    assert llm.vision_calls == 0
    assert list_inbox(store) == []
    done = json.loads((Path(store) / "done" / f"{receipt_id}.json").read_text())
    assert done["status"] == "failed"
    assert "gave up after 3 attempts" in done["error"]

    failed = audit.events_of("receipt_ocr_failed")
    assert len(failed) == 1
    assert failed[0]["attempts"] == 3


def test_attempt_counter_is_written_before_ocr_runs(store, tmp_path):
    """A run killed mid-OCR writes nothing afterwards, so the count has to be
    persisted on the way in or the receipt retries forever."""
    receipt_id = _seed(store, tmp_path)
    inbox_json = Path(store) / "inbox" / f"{receipt_id}.json"

    class ExplodingLLM(FakeLLM):
        def complete_json_vision(self, *a, **kw):
            # Stands in for SIGTERM: nothing after this point gets to run.
            raise KeyboardInterrupt("killed mid-call")

    with pytest.raises(KeyboardInterrupt):
        _run(store, ExplodingLLM(), FakeAudit())

    assert json.loads(inbox_json.read_text())["ocr_attempts"] == 1


def test_cap_of_zero_means_unlimited(store, tmp_path):
    receipt_id = _seed(store, tmp_path)
    inbox_json = Path(store) / "inbox" / f"{receipt_id}.json"
    meta = json.loads(inbox_json.read_text())
    meta["ocr_attempts"] = 99
    inbox_json.write_text(json.dumps(meta))

    llm, audit = FakeLLM(), FakeAudit()
    _run(store, llm, audit, receipts_max_ocr_attempts=0)

    assert llm.vision_calls == 1  # still attempted despite the high count


def test_one_bad_receipt_does_not_block_the_next(store, tmp_path):
    """Receipts is step 4 of 6 — a receipt that throws must not stop the ones
    behind it, nor the pipeline steps that follow."""
    first = _seed(store, tmp_path)
    (tmp_path / "second.jpg").write_bytes(b"\xff\xd8\xff" + b"\x00" * 10)
    second = save_received(store, tmp_path / "second.jpg", "ios")

    class ExplodesOnFirst(FakeLLM):
        def complete_json_vision(self, *a, **kw):
            self.vision_calls += 1
            if self.vision_calls == 1:
                raise RuntimeError("boom")
            return {"error": "LLM call failure: timed out"}

    llm, audit = ExplodesOnFirst(), FakeAudit()
    _run(store, llm, audit)

    assert llm.vision_calls == 2  # the receipt behind the failure was still attempted
    assert list_inbox(store) == []
    errors = {e["receipt_id"]: e["error"] for e in audit.events_of("receipt_ocr_failed")}
    # list_inbox sorts by receipt id, so which one raises isn't fixed — only that
    # both were processed, one via the exception path and one via the error path.
    assert set(errors) == {first, second}
    assert sorted("boom" in e for e in errors.values()) == [False, True]
