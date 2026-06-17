"""Unit tests for the receipt HTTP receiver."""

import json

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def set_env(tmp_path, monkeypatch):
    monkeypatch.setenv("RECEIPT_RECEIVER_TOKEN", "test-token")
    monkeypatch.setenv("RECEIPTS_STORE_PATH", str(tmp_path / "receipts"))


@pytest.fixture
def client():
    # Import after env is patched
    from actual_cat.receipts.receiver import app
    return TestClient(app, raise_server_exceptions=True)


def _post_image(client, image_bytes=None, content_type="image/jpeg",
                token="test-token", extra_fields=None):
    if image_bytes is None:
        image_bytes = b"\xff\xd8\xff" + b"\x00" * 20  # minimal JPEG magic
    files = {"image": ("receipt.jpg", image_bytes, content_type)}
    data = extra_fields or {}
    headers = {"Authorization": f"Bearer {token}"}
    return client.post("/receipts/", files=files, data=data, headers=headers)


class TestAuth:
    def test_missing_token_returns_401(self, client):
        resp = client.post("/receipts/", files={"image": ("r.jpg", b"\xff\xd8", "image/jpeg")})
        assert resp.status_code == 401

    def test_wrong_token_returns_401(self, client):
        resp = _post_image(client, token="wrong")
        assert resp.status_code == 401

    def test_correct_token_accepted(self, client):
        resp = _post_image(client)
        assert resp.status_code == 201


class TestReceiptPost:
    def test_returns_receipt_id_and_status(self, client):
        resp = _post_image(client)
        body = resp.json()
        assert "receipt_id" in body
        assert body["status"] == "received"

    def test_image_file_written_to_inbox(self, client, tmp_path):
        image_bytes = b"\xff\xd8\xff" + b"\xAB" * 50
        resp = _post_image(client, image_bytes=image_bytes)
        receipt_id = resp.json()["receipt_id"]
        store = tmp_path / "receipts"
        image_path = store / "inbox" / f"{receipt_id}.jpg"
        assert image_path.exists()
        assert image_path.read_bytes() == image_bytes

    def test_meta_json_written_to_inbox(self, client, tmp_path):
        resp = _post_image(client)
        receipt_id = resp.json()["receipt_id"]
        meta_path = tmp_path / "receipts" / "inbox" / f"{receipt_id}.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["id"] == receipt_id
        assert meta["status"] == "received"
        assert meta["source"] == "ios"

    def test_hint_fields_stored_in_meta(self, client, tmp_path):
        resp = _post_image(
            client,
            extra_fields={"hint_merchant": "Trader Joe's", "hint_date": "2026-06-07"},
        )
        receipt_id = resp.json()["receipt_id"]
        meta = json.loads((tmp_path / "receipts" / "inbox" / f"{receipt_id}.json").read_text())
        assert meta["hint_merchant"] == "Trader Joe's"
        assert meta["hint_date"] == "2026-06-07"

    def test_png_gets_png_extension(self, client, tmp_path):
        png_bytes = b"\x89PNG" + b"\x00" * 20
        resp = _post_image(client, image_bytes=png_bytes, content_type="image/png")
        receipt_id = resp.json()["receipt_id"]
        assert (tmp_path / "receipts" / "inbox" / f"{receipt_id}.png").exists()

    def test_unsupported_content_type_returns_415(self, client):
        resp = _post_image(client, content_type="application/pdf")
        assert resp.status_code == 415

    def test_oversized_image_returns_413(self, client):
        big = b"\xff\xd8\xff" + b"\x00" * (21 * 1024 * 1024)
        resp = _post_image(client, image_bytes=big)
        assert resp.status_code == 413


def _post_text(client, text="TRADER JOE'S\nBananas 1.99\nTotal 1.99",
               token="test-token", extra_fields=None):
    data = {"text": text}
    if extra_fields:
        data.update(extra_fields)
    headers = {"Authorization": f"Bearer {token}"}
    return client.post("/receipts/text", data=data, headers=headers)


class TestReceiptTextPost:
    def test_missing_token_returns_401(self, client):
        resp = client.post("/receipts/text", data={"text": "x"})
        assert resp.status_code == 401

    def test_returns_receipt_id_and_status(self, client):
        resp = _post_text(client)
        assert resp.status_code == 201
        body = resp.json()
        assert "receipt_id" in body
        assert body["status"] == "received"

    def test_meta_written_with_text_input_kind(self, client, tmp_path):
        resp = _post_text(client, text="WHOLE FOODS\nApples 4.20\nTotal 4.20")
        receipt_id = resp.json()["receipt_id"]
        meta_path = tmp_path / "receipts" / "inbox" / f"{receipt_id}.json"
        meta = json.loads(meta_path.read_text())
        assert meta["status"] == "received"
        assert meta["source"] == "ios"
        assert meta["input_kind"] == "text"
        assert "WHOLE FOODS" in meta["text"]
        # No image file for a text receipt
        assert list((tmp_path / "receipts" / "inbox").glob("*.jpg")) == []

    def test_hint_fields_stored_in_meta(self, client, tmp_path):
        resp = _post_text(
            client, extra_fields={"hint_merchant": "Trader Joe's", "hint_date": "2026-06-07"}
        )
        receipt_id = resp.json()["receipt_id"]
        meta = json.loads((tmp_path / "receipts" / "inbox" / f"{receipt_id}.json").read_text())
        assert meta["hint_merchant"] == "Trader Joe's"
        assert meta["hint_date"] == "2026-06-07"

    def test_empty_text_returns_422(self, client):
        resp = _post_text(client, text="   ")
        assert resp.status_code == 422
