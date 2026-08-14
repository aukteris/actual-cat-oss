"""Unit tests for load_config / _load_llm_profiles parsing logic."""

import textwrap
from pathlib import Path

import pytest

from actual_cat.config import _load_llm_profiles, load_config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MINIMAL_TOML_TEMPLATE = textwrap.dedent("""
    [actual]
    base_url = "http://test"
    file = "Test"

    {llm_block}

    [categorization]
    mode = "suggest"
    apply_confidence_threshold = "high"

    [transfers]
    mode = "suggest"
    window_days = 3
    apply_confidence_threshold = "high"

    [audit]
    log_path = "/tmp/test.jsonl"
""")


def _make_toml(tmp_path: Path, llm_block: str) -> Path:
    p = tmp_path / "config.toml"
    p.write_text(_MINIMAL_TOML_TEMPLATE.format(llm_block=llm_block))
    return p


# ---------------------------------------------------------------------------
# _load_llm_profiles — extra_body synthesis from legacy keys
# ---------------------------------------------------------------------------


def test_legacy_keys_synthesize_extra_body() -> None:
    import tomllib
    raw = tomllib.loads(textwrap.dedent("""
        endpoint = "http://local/v1"
        model = "my-model"
        top_k = 30
        min_p = 0.1
        enable_thinking = true
    """))
    text, vision = _load_llm_profiles(raw)
    assert text.extra_body == {
        "top_k": 30,
        "min_p": 0.1,
        "chat_template_kwargs": {"enable_thinking": True},
    }


def test_no_extra_body_no_legacy_keys_is_cloud_safe() -> None:
    # No [extra_body] block and no legacy top-level keys: send nothing
    # provider-specific, so cloud providers (e.g. Gemini) don't 400.
    import tomllib
    raw = tomllib.loads(textwrap.dedent("""
        endpoint = "https://generativelanguage.googleapis.com/v1beta/openai/"
        model = "gemini-2.5-flash"
    """))
    text, _ = _load_llm_profiles(raw)
    assert text.extra_body == {}


def test_partial_legacy_keys_still_synthesize() -> None:
    # A single legacy key present means the user is on the llama.cpp style;
    # synthesize the full block (missing keys fall back to defaults).
    import tomllib
    raw = tomllib.loads(textwrap.dedent("""
        endpoint = "http://local/v1"
        model = "my-model"
        min_p = 0.1
    """))
    text, _ = _load_llm_profiles(raw)
    assert text.extra_body == {
        "top_k": 20,
        "min_p": 0.1,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def test_explicit_extra_body_used_verbatim() -> None:
    import tomllib
    raw = tomllib.loads(textwrap.dedent("""
        endpoint = "http://local/v1"
        model = "my-model"

        [extra_body]
        custom_param = "foo"
        num_predict = 512
    """))
    text, _ = _load_llm_profiles(raw)
    assert text.extra_body == {"custom_param": "foo", "num_predict": 512}
    # legacy keys should be ignored when extra_body is explicit
    assert "top_k" not in text.extra_body


def test_explicit_extra_body_overrides_legacy_keys() -> None:
    import tomllib
    raw = tomllib.loads(textwrap.dedent("""
        endpoint = "http://local/v1"
        model = "my-model"
        top_k = 99
        min_p = 0.99

        [extra_body]
        only_this = true
    """))
    text, _ = _load_llm_profiles(raw)
    assert text.extra_body == {"only_this": True}


# ---------------------------------------------------------------------------
# _load_llm_profiles — vision fallback behaviour
# ---------------------------------------------------------------------------


def test_vision_inherits_text_when_not_configured() -> None:
    import tomllib
    raw = tomllib.loads(textwrap.dedent("""
        endpoint = "http://local/v1"
        model = "text-model"
    """))
    text, vision = _load_llm_profiles(raw)
    assert vision.endpoint == text.endpoint
    assert vision.model == text.model
    assert vision.extra_body == text.extra_body


def test_vision_overrides_endpoint_and_model() -> None:
    import tomllib
    raw = tomllib.loads(textwrap.dedent("""
        endpoint = "http://local/v1"
        model = "text-model"

        [vision]
        endpoint = "https://openrouter.ai/api/v1"
        model = "google/gemini-2.5-flash"
    """))
    text, vision = _load_llm_profiles(raw)
    assert vision.endpoint == "https://openrouter.ai/api/v1"
    assert vision.model == "google/gemini-2.5-flash"
    # non-overridden fields still inherit
    assert vision.temperature == text.temperature


def test_vision_explicit_extra_body() -> None:
    import tomllib
    raw = tomllib.loads(textwrap.dedent("""
        endpoint = "http://local/v1"
        model = "text-model"

        [vision]
        endpoint = "https://cloud/v1"
        model = "cloud-vision"

        [vision.extra_body]
        special = "val"
    """))
    _, vision = _load_llm_profiles(raw)
    assert vision.extra_body == {"special": "val"}


def test_vision_no_extra_body_key_inherits_empty_when_no_vision_config() -> None:
    """When [llm.vision] is present but no extra_body, default is {} (cloud-safe)."""
    import tomllib
    raw = tomllib.loads(textwrap.dedent("""
        endpoint = "http://local/v1"
        model = "text-model"

        [vision]
        endpoint = "https://cloud/v1"
        model = "cloud-vision"
    """))
    _, vision = _load_llm_profiles(raw)
    # Vision block present but no extra_body → cloud-safe empty dict
    assert vision.extra_body == {}


# ---------------------------------------------------------------------------
# _load_llm_profiles — API keys from environment
# ---------------------------------------------------------------------------


def test_api_key_default_when_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_VISION_API_KEY", raising=False)
    import tomllib
    raw = tomllib.loads('endpoint = "http://local/v1"\nmodel = "m"')
    text, vision = _load_llm_profiles(raw)
    assert text.api_key == "not-needed"
    assert vision.api_key == "not-needed"


def test_text_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "my-secret-key")
    monkeypatch.delenv("LLM_VISION_API_KEY", raising=False)
    import tomllib
    raw = tomllib.loads('endpoint = "http://local/v1"\nmodel = "m"')
    text, vision = _load_llm_profiles(raw)
    assert text.api_key == "my-secret-key"
    # vision falls back to text key when LLM_VISION_API_KEY not set
    assert vision.api_key == "my-secret-key"


def test_vision_api_key_separate_from_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "text-key")
    monkeypatch.setenv("LLM_VISION_API_KEY", "vision-key")
    import tomllib
    raw = tomllib.loads(textwrap.dedent("""
        endpoint = "http://local/v1"
        model = "m"
        [vision]
        endpoint = "https://cloud/v1"
        model = "cloud-vision"
    """))
    text, vision = _load_llm_profiles(raw)
    assert text.api_key == "text-key"
    assert vision.api_key == "vision-key"


# ---------------------------------------------------------------------------
# json_mode default
# ---------------------------------------------------------------------------


def test_json_mode_defaults_true() -> None:
    import tomllib
    raw = tomllib.loads('endpoint = "http://local/v1"\nmodel = "m"')
    text, _ = _load_llm_profiles(raw)
    assert text.json_mode is True


def test_json_mode_can_be_disabled() -> None:
    import tomllib
    raw = tomllib.loads('endpoint = "http://local/v1"\nmodel = "m"\njson_mode = false')
    text, _ = _load_llm_profiles(raw)
    assert text.json_mode is False


# ---------------------------------------------------------------------------
# load_config integration — llm_text and llm_vision on Config
# ---------------------------------------------------------------------------


def test_load_config_exposes_llm_text_and_vision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ACTUAL_PASSWORD", "pw")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_VISION_API_KEY", raising=False)
    toml_path = _make_toml(tmp_path, textwrap.dedent("""
        [llm]
        endpoint = "http://local/v1"
        model = "my-model"
    """))
    cfg = load_config(str(toml_path))
    assert cfg.llm_text.model == "my-model"
    assert cfg.llm_text.endpoint == "http://local/v1"
    assert cfg.llm_vision.model == "my-model"  # falls back to text


def test_load_config_vision_block_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ACTUAL_PASSWORD", "pw")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_VISION_API_KEY", raising=False)
    toml_path = _make_toml(tmp_path, textwrap.dedent("""
        [llm]
        endpoint = "http://local/v1"
        model = "text-model"

        [llm.vision]
        endpoint = "https://cloud/v1"
        model = "vision-model"
    """))
    cfg = load_config(str(toml_path))
    assert cfg.llm_text.model == "text-model"
    assert cfg.llm_vision.model == "vision-model"
    assert cfg.llm_vision.endpoint == "https://cloud/v1"


def test_load_config_rate_limit_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ACTUAL_PASSWORD", "pw")
    toml_path = _make_toml(tmp_path, textwrap.dedent("""
        [llm]
        endpoint = "http://local/v1"
        model = "my-model"
    """))
    cfg = load_config(str(toml_path))
    # Omitted [llm.rate_limit] → no throttle, SDK default retries.
    assert cfg.llm_requests_per_minute == 0
    assert cfg.llm_max_retries == 2


def test_load_config_rate_limit_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ACTUAL_PASSWORD", "pw")
    toml_path = _make_toml(tmp_path, textwrap.dedent("""
        [llm]
        endpoint = "https://generativelanguage.googleapis.com/v1beta/openai/"
        model = "gemini-2.5-flash"

        [llm.rate_limit]
        requests_per_minute = 15
        max_retries = 5
    """))
    cfg = load_config(str(toml_path))
    assert cfg.llm_requests_per_minute == 15
    assert cfg.llm_max_retries == 5


# ---------------------------------------------------------------------------
# [duplicates] — pending-duplicate pipeline
# ---------------------------------------------------------------------------

_LLM_BLOCK = textwrap.dedent("""
    [llm]
    endpoint = "http://local/v1"
    model = "my-model"
""")


def test_load_config_duplicates_absent_block_is_inert(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An existing install with no [duplicates] block keeps its old behavior."""
    monkeypatch.setenv("ACTUAL_PASSWORD", "pw")
    toml_path = _make_toml(tmp_path, _LLM_BLOCK)
    cfg = load_config(str(toml_path))
    assert cfg.duplicates_enabled is False
    # Crucially also off: the other pipelines keep seeing pending rows.
    assert cfg.duplicates_defer_pending is False
    assert cfg.duplicates_mode == "suggest"
    assert cfg.duplicates_window_days == 7
    assert cfg.duplicates_max_uplift_pct == 40
    assert cfg.duplicates_max_reduction_pct == 5
    assert cfg.duplicates_auth_hold_max_cents == 200
    assert cfg.duplicates_threshold == "high"


def test_load_config_duplicates_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ACTUAL_PASSWORD", "pw")
    p = tmp_path / "config.toml"
    p.write_text(
        _MINIMAL_TOML_TEMPLATE.format(llm_block=_LLM_BLOCK)
        + textwrap.dedent("""
            [duplicates]
            enabled = true
            mode = "apply"
            window_days = 5
            max_uplift_pct = 35
            max_reduction_pct = 3
            auth_hold_max_cents = 150
            apply_confidence_threshold = "medium"
            defer_pending = true
        """)
    )
    cfg = load_config(str(p))
    assert cfg.duplicates_enabled is True
    assert cfg.duplicates_mode == "apply"
    assert cfg.duplicates_window_days == 5
    assert cfg.duplicates_max_uplift_pct == 35
    assert cfg.duplicates_max_reduction_pct == 3
    assert cfg.duplicates_auth_hold_max_cents == 150
    assert cfg.duplicates_threshold == "medium"
    assert cfg.duplicates_defer_pending is True


def test_load_config_defer_pending_without_enabling_the_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """defer_pending stands on its own — it needs no matcher to be useful."""
    monkeypatch.setenv("ACTUAL_PASSWORD", "pw")
    p = tmp_path / "config.toml"
    p.write_text(
        _MINIMAL_TOML_TEMPLATE.format(llm_block=_LLM_BLOCK)
        + textwrap.dedent("""
            [duplicates]
            defer_pending = true
        """)
    )
    cfg = load_config(str(p))
    assert cfg.duplicates_enabled is False
    assert cfg.duplicates_defer_pending is True
