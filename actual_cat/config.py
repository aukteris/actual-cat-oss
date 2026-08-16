import os
import tomllib
from dataclasses import dataclass
from typing import cast

from dotenv import load_dotenv

from .llm import LLMProfile
from .receipts import ocr  # OCR time-bound defaults live with the code that enforces them


@dataclass(frozen=True)
class Config:
    base_url: str
    file: str
    password: str
    encryption_password: str | None  # None for plaintext budgets
    llm_text: LLMProfile
    llm_vision: LLMProfile           # falls back to llm_text when not configured
    llm_requests_per_minute: int     # client-side throttle; 0 = no throttle
    llm_max_retries: int             # OpenAI SDK retries on 429 (honors Retry-After)
    categorization_mode: str         # "suggest" | "apply"
    categorization_threshold: str    # "high" | "medium" | "low"
    transfer_mode: str
    transfer_window_days: int
    transfer_threshold: str
    audit_log_path: str
    ca_bundle: str | None
    # receipts pipeline
    receipts_enabled: bool
    receipts_store_path: str
    receipts_match_window_days: int
    receipts_expiry_days: int
    receipts_mode: str               # "suggest" | "apply"
    receipts_threshold: str
    receipts_ocr_request_timeout_seconds: float | None  # per vision pass; None = no cap
    receipts_ocr_budget_seconds: float | None           # all passes for one receipt
    receipts_max_ocr_attempts: int   # runs a receipt gets before it's failed; 0 = unlimited
    receipts_autocrop: bool          # crop to the receipt before OCR
    # pending-duplicate pipeline
    duplicates_enabled: bool
    duplicates_mode: str             # "suggest" | "apply"
    duplicates_window_days: int
    duplicates_max_uplift_pct: int
    duplicates_max_reduction_pct: int
    duplicates_auth_hold_max_cents: int
    duplicates_threshold: str
    duplicates_defer_pending: bool   # other pipelines ignore pending rows
    # scheduled bank sync
    bank_sync_enabled: bool
    bank_sync_interval_minutes: int
    bank_sync_grace_minutes: int     # absorbs systemd RandomizedDelaySec jitter
    bank_sync_max_runs_per_day: int  # 0 = uncapped
    bank_sync_accounts: list[str]        # empty = all sync-enabled accounts
    bank_sync_exclude_accounts: list[str]
    bank_sync_lookback_days: int     # 0 = actualpy default (per-account last transaction)
    bank_sync_allow_first_sync: bool # guard the auto "Starting Balance" reconciliation row
    state_path: str
    # email ingestion
    email_enabled: bool
    email_imap_host: str
    email_user: str
    email_mailbox: str
    # receipt HTTP receiver
    receiver_host: str
    receiver_port: int
    # historical-categorization hints
    history_enabled: bool
    history_payee_top_n: int
    history_item_top_n: int
    history_min_count: int


def _load_llm_profiles(llm: dict[str, object]) -> tuple[LLMProfile, LLMProfile]:
    """Parse [llm] (and optional [llm.vision]) into text + vision LLMProfile objects.

    Backward-compat: if [llm.extra_body] is absent, synthesize it from the
    legacy top-level top_k/min_p/enable_thinking keys so existing config.toml
    files keep working unchanged.

    API keys come from env only — never stored in config:
      LLM_API_KEY          text model (default "not-needed" for local servers)
      LLM_VISION_API_KEY   vision model (falls back to LLM_API_KEY)
    """
    text_api_key = os.environ.get("LLM_API_KEY") or "not-needed"
    vision_api_key = os.environ.get("LLM_VISION_API_KEY") or text_api_key

    # [llm.extra_body] takes precedence. Otherwise synthesize from legacy top-level
    # keys ONLY when they're actually present, so cloud providers (which omit both
    # the block and the legacy keys) get an empty extra_body and aren't sent
    # llama.cpp-specific params they'd reject with a 400.
    legacy_keys = ("top_k", "min_p", "enable_thinking")
    if "extra_body" in llm:
        text_extra: dict[str, object] = dict(cast(dict[str, object], llm["extra_body"]))
    elif any(k in llm for k in legacy_keys):
        text_extra = {
            "top_k": llm.get("top_k", 20),
            "min_p": llm.get("min_p", 0.05),
            "chat_template_kwargs": {"enable_thinking": llm.get("enable_thinking", False)},
        }
    else:
        text_extra = {}  # cloud-safe: nothing provider-specific to send

    text = LLMProfile(
        endpoint=str(llm["endpoint"]),
        api_key=text_api_key,
        model=str(llm["model"]),
        temperature=float(llm.get("temperature", 0.2)),  # type: ignore[arg-type]
        top_p=float(llm.get("top_p", 0.85)),  # type: ignore[arg-type]
        presence_penalty=float(llm.get("presence_penalty", 1.0)),  # type: ignore[arg-type]
        json_mode=bool(llm.get("json_mode", True)),
        extra_body=text_extra,
        timeout_seconds=cast(float | None, llm.get("timeout_seconds", 120.0)),
    )

    vision_raw: dict[str, object] = llm.get("vision", {})  # type: ignore[assignment]
    if "extra_body" in vision_raw:
        vision_extra: dict[str, object] = dict(cast(dict[str, object], vision_raw["extra_body"]))
    else:
        vision_extra = text_extra if not vision_raw else {}

    vision = LLMProfile(
        endpoint=str(vision_raw.get("endpoint", text.endpoint)),
        api_key=vision_api_key if vision_raw else text_api_key,
        model=str(vision_raw.get("model", text.model)),
        temperature=float(vision_raw.get("temperature", text.temperature)),  # type: ignore[arg-type]
        top_p=float(vision_raw.get("top_p", text.top_p)),  # type: ignore[arg-type]
        presence_penalty=float(vision_raw.get("presence_penalty", text.presence_penalty)),  # type: ignore[arg-type]
        json_mode=bool(vision_raw.get("json_mode", text.json_mode)),
        extra_body=vision_extra,
        # Inherited like the sampling params above. The receipts pipeline overrides
        # this per request anyway (a vision pass legitimately takes far longer than
        # a text one); this only bounds vision calls made outside that path.
        timeout_seconds=cast(
            float | None, vision_raw.get("timeout_seconds", text.timeout_seconds)
        ),
    )

    return text, vision


def load_config(path: str = "config.toml") -> Config:
    load_dotenv()
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    password = os.environ.get("ACTUAL_PASSWORD")
    if not password:
        raise ValueError("ACTUAL_PASSWORD not set in environment")

    encryption_password = os.environ.get("ACTUAL_ENCRYPTION_PASSWORD") or None

    llm_text, llm_vision = _load_llm_profiles(raw["llm"])
    rate_limit = raw["llm"].get("rate_limit", {})

    receipts = raw.get("receipts", {})
    duplicates = raw.get("duplicates", {})
    bank_sync = raw.get("bank_sync", {})
    state = raw.get("state", {})
    email = raw.get("email", {})
    receiver = raw.get("receiver", {})
    history = raw.get("history", {})

    return Config(
        base_url=raw["actual"]["base_url"],
        file=raw["actual"]["file"],
        password=password,
        encryption_password=encryption_password,
        llm_text=llm_text,
        llm_vision=llm_vision,
        llm_requests_per_minute=rate_limit.get("requests_per_minute", 0),
        llm_max_retries=rate_limit.get("max_retries", 2),
        categorization_mode=raw["categorization"]["mode"],
        categorization_threshold=raw["categorization"]["apply_confidence_threshold"],
        transfer_mode=raw["transfers"]["mode"],
        transfer_window_days=raw["transfers"]["window_days"],
        transfer_threshold=raw["transfers"]["apply_confidence_threshold"],
        audit_log_path=raw["audit"]["log_path"],
        ca_bundle=raw.get("paths", {}).get("ca_bundle"),
        receipts_enabled=receipts.get("enabled", False),
        receipts_store_path=receipts.get("store_path", "receipts"),
        receipts_match_window_days=receipts.get("match_window_days", 3),
        receipts_expiry_days=receipts.get("expiry_days", 30),
        receipts_mode=receipts.get("mode", "suggest"),
        receipts_threshold=receipts.get("apply_confidence_threshold", "high"),
        receipts_ocr_request_timeout_seconds=receipts.get(
            "ocr_request_timeout_seconds", ocr.DEFAULT_REQUEST_TIMEOUT_SECONDS
        ),
        receipts_ocr_budget_seconds=receipts.get(
            "ocr_budget_seconds", ocr.DEFAULT_BUDGET_SECONDS
        ),
        receipts_max_ocr_attempts=receipts.get("max_ocr_attempts", 3),
        receipts_autocrop=receipts.get("autocrop", True),
        # Absent [duplicates] block leaves an existing install exactly as it was:
        # the pipeline off, and the other pipelines still seeing pending rows.
        duplicates_enabled=duplicates.get("enabled", False),
        duplicates_mode=duplicates.get("mode", "suggest"),
        duplicates_window_days=duplicates.get("window_days", 7),
        duplicates_max_uplift_pct=duplicates.get("max_uplift_pct", 40),
        duplicates_max_reduction_pct=duplicates.get("max_reduction_pct", 5),
        duplicates_auth_hold_max_cents=duplicates.get("auth_hold_max_cents", 200),
        duplicates_threshold=duplicates.get("apply_confidence_threshold", "high"),
        duplicates_defer_pending=duplicates.get("defer_pending", False),
        # Absent [bank_sync] block leaves an existing install exactly as it was:
        # the pipeline off, sync left entirely to actual-server's own schedule.
        bank_sync_enabled=bank_sync.get("enabled", False),
        bank_sync_interval_minutes=bank_sync.get("interval_minutes", 360),
        bank_sync_grace_minutes=bank_sync.get("grace_minutes", 5),
        bank_sync_max_runs_per_day=bank_sync.get("max_runs_per_day", 0),
        bank_sync_accounts=bank_sync.get("accounts", []),
        bank_sync_exclude_accounts=bank_sync.get("exclude_accounts", []),
        bank_sync_lookback_days=bank_sync.get("lookback_days", 0),
        bank_sync_allow_first_sync=bank_sync.get("allow_first_sync", False),
        state_path=state.get("path", "state/actual-cat.json"),
        email_enabled=email.get("enabled", False),
        email_imap_host=email.get("imap_host", ""),
        email_user=email.get("user", ""),
        email_mailbox=email.get("mailbox", "INBOX"),
        receiver_host=receiver.get("host", "127.0.0.1"),
        receiver_port=receiver.get("port", 8001),
        history_enabled=history.get("enabled", True),
        history_payee_top_n=history.get("payee_top_n", 3),
        history_item_top_n=history.get("item_top_n", 3),
        history_min_count=history.get("min_count", 1),
    )
