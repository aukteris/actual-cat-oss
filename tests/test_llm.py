"""Unit tests for LLMClient request construction and error handling."""

from typing import Any
from unittest.mock import MagicMock

from actual_cat.llm import LLMClient, LLMProfile

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_response(content: str) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _text_profile(**overrides: Any) -> LLMProfile:
    defaults: dict[str, Any] = dict(
        endpoint="http://local/v1",
        api_key="test-key",
        model="test-model",
        temperature=0.2,
        top_p=0.85,
        presence_penalty=1.0,
        json_mode=True,
        extra_body={"top_k": 20, "min_p": 0.05},
    )
    defaults.update(overrides)
    return LLMProfile(**defaults)


# ---------------------------------------------------------------------------
# complete_json — sampling params forwarded
# ---------------------------------------------------------------------------


def test_complete_json_passes_sampling_params() -> None:
    profile = _text_profile()
    client = LLMClient(profile)

    captured: list[dict[str, Any]] = []

    def fake_create(**kwargs: Any) -> MagicMock:
        captured.append(kwargs)
        return _mock_response('{"ok": true}')

    client._text_client.chat.completions.create = fake_create  # type: ignore[method-assign]

    result = client.complete_json("sys", "user")
    assert result == {"ok": True}
    kw = captured[0]
    assert kw["temperature"] == 0.2
    assert kw["top_p"] == 0.85
    assert kw["presence_penalty"] == 1.0


def test_complete_json_omits_none_sampling_params() -> None:
    profile = _text_profile(temperature=None, top_p=None, presence_penalty=None)
    client = LLMClient(profile)

    captured: list[dict[str, Any]] = []

    def fake_create(**kwargs: Any) -> MagicMock:
        captured.append(kwargs)
        return _mock_response('{"ok": true}')

    client._text_client.chat.completions.create = fake_create  # type: ignore[method-assign]

    client.complete_json("sys", "user")
    kw = captured[0]
    assert "temperature" not in kw
    assert "top_p" not in kw
    assert "presence_penalty" not in kw


# ---------------------------------------------------------------------------
# json_mode toggle
# ---------------------------------------------------------------------------


def test_json_mode_on_sends_response_format() -> None:
    profile = _text_profile(json_mode=True)
    client = LLMClient(profile)

    captured: list[dict[str, Any]] = []

    def fake_create(**kwargs: Any) -> MagicMock:
        captured.append(kwargs)
        return _mock_response('{"x": 1}')

    client._text_client.chat.completions.create = fake_create  # type: ignore[method-assign]
    client.complete_json("s", "u")
    assert captured[0].get("response_format") == {"type": "json_object"}


def test_json_mode_off_omits_response_format() -> None:
    profile = _text_profile(json_mode=False)
    client = LLMClient(profile)

    captured: list[dict[str, Any]] = []

    def fake_create(**kwargs: Any) -> MagicMock:
        captured.append(kwargs)
        return _mock_response('{"x": 1}')

    client._text_client.chat.completions.create = fake_create  # type: ignore[method-assign]
    client.complete_json("s", "u")
    assert "response_format" not in captured[0]


# ---------------------------------------------------------------------------
# extra_body passthrough
# ---------------------------------------------------------------------------


def test_extra_body_passed_when_non_empty() -> None:
    profile = _text_profile(extra_body={"top_k": 20, "min_p": 0.05})
    client = LLMClient(profile)

    captured: list[dict[str, Any]] = []

    def fake_create(**kwargs: Any) -> MagicMock:
        captured.append(kwargs)
        return _mock_response('{"x": 1}')

    client._text_client.chat.completions.create = fake_create  # type: ignore[method-assign]
    client.complete_json("s", "u")
    assert captured[0]["extra_body"] == {"top_k": 20, "min_p": 0.05}


def test_extra_body_omitted_when_empty() -> None:
    profile = _text_profile(extra_body={})
    client = LLMClient(profile)

    captured: list[dict[str, Any]] = []

    def fake_create(**kwargs: Any) -> MagicMock:
        captured.append(kwargs)
        return _mock_response('{"x": 1}')

    client._text_client.chat.completions.create = fake_create  # type: ignore[method-assign]
    client.complete_json("s", "u")
    assert "extra_body" not in captured[0]


# ---------------------------------------------------------------------------
# Text vs vision profiles use independent clients + models
# ---------------------------------------------------------------------------


def test_vision_uses_vision_profile_model() -> None:
    text_profile = _text_profile(model="text-model", endpoint="http://text/v1")
    vision_profile = _text_profile(model="vision-model", endpoint="http://vision/v1")
    client = LLMClient(text_profile, vision_profile)

    vision_captured: list[dict[str, Any]] = []

    def fake_vision_create(**kwargs: Any) -> MagicMock:
        vision_captured.append(kwargs)
        return _mock_response('{"v": 1}')

    client._vision_client.chat.completions.create = fake_vision_create  # type: ignore[method-assign]
    client.complete_json_vision("s", "u", "abc123", "image/png")
    assert vision_captured[0]["model"] == "vision-model"


def test_text_uses_text_profile_model() -> None:
    text_profile = _text_profile(model="text-model", endpoint="http://text/v1")
    vision_profile = _text_profile(model="vision-model", endpoint="http://vision/v1")
    client = LLMClient(text_profile, vision_profile)

    text_captured: list[dict[str, Any]] = []

    def fake_text_create(**kwargs: Any) -> MagicMock:
        text_captured.append(kwargs)
        return _mock_response('{"t": 1}')

    client._text_client.chat.completions.create = fake_text_create  # type: ignore[method-assign]
    client.complete_json("s", "u")
    assert text_captured[0]["model"] == "text-model"


def test_vision_defaults_to_text_profile_when_not_set() -> None:
    text_profile = _text_profile(model="text-model")
    client = LLMClient(text_profile)
    assert client.vision is text_profile


# ---------------------------------------------------------------------------
# Client reuse for same (endpoint, api_key)
# ---------------------------------------------------------------------------


def test_same_endpoint_key_shares_client() -> None:
    p1 = _text_profile(model="m1")
    p2 = _text_profile(model="m2")  # same endpoint and api_key
    client = LLMClient(p1, p2)
    assert client._text_client is client._vision_client


def test_different_endpoint_creates_separate_client() -> None:
    p1 = _text_profile(endpoint="http://a/v1")
    p2 = _text_profile(endpoint="http://b/v1")
    client = LLMClient(p1, p2)
    assert client._text_client is not client._vision_client


# ---------------------------------------------------------------------------
# Retry on JSON parse failure
# ---------------------------------------------------------------------------


def test_retries_on_json_parse_failure() -> None:
    profile = _text_profile()
    client = LLMClient(profile)

    call_count = 0

    def fake_create(**kwargs: Any) -> MagicMock:
        nonlocal call_count
        call_count += 1
        return _mock_response("not json")

    client._text_client.chat.completions.create = fake_create  # type: ignore[method-assign]
    result = client.complete_json("s", "u", retries=2)
    assert "error" in result
    assert call_count == 3  # 1 initial + 2 retries


# ---------------------------------------------------------------------------
# LLM call exception returns error dict
# ---------------------------------------------------------------------------


def test_llm_exception_returns_error_dict() -> None:
    profile = _text_profile()
    client = LLMClient(profile)

    def fake_create(**kwargs: Any) -> MagicMock:
        raise RuntimeError("connection refused")

    client._text_client.chat.completions.create = fake_create  # type: ignore[method-assign]
    result = client.complete_json("s", "u")
    assert result == {"error": "LLM call failure: connection refused"}


# ---------------------------------------------------------------------------
# Code-fence stripping in response
# ---------------------------------------------------------------------------


def test_code_fence_stripped_before_json_parse() -> None:
    profile = _text_profile(json_mode=False)
    client = LLMClient(profile)

    def fake_create(**kwargs: Any) -> MagicMock:
        return _mock_response("```json\n{\"ok\": true}\n```")

    client._text_client.chat.completions.create = fake_create  # type: ignore[method-assign]
    result = client.complete_json("s", "u")
    assert result == {"ok": True}
