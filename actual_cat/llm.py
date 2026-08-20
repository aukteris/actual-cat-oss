import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


@dataclass(frozen=True)
class LLMProfile:
    """One LLM target: where to call, how to authenticate, and how to sample.

    Text and vision can use independent profiles (different endpoint/model/key),
    so a local text model can coexist with a cloud vision model, or vice versa.

    - api_key: real key for cloud providers; "not-needed" for keyless local servers.
    - json_mode: send response_format={"type":"json_object"}. Disable for models
      that reject it — the code-fence stripping below is the fallback.
    - extra_body: provider-specific params sent verbatim (e.g. llama.cpp's top_k /
      min_p / chat_template_kwargs). Empty for standard OpenAI-compatible providers.
    - temperature / top_p / presence_penalty: omitted from the request when None,
      for models that reject them.
    - timeout_seconds: per-request cap. None leaves the SDK default (600s), which is
      long enough that a stalled call outlives the systemd unit's own timeout and
      gets SIGTERMed instead of surfacing as a logged error.
    """

    endpoint: str
    api_key: str
    model: str
    temperature: float | None = 0.2
    top_p: float | None = 0.85
    presence_penalty: float | None = 1.0
    json_mode: bool = True
    extra_body: dict[str, Any] = field(default_factory=dict)
    timeout_seconds: float | None = None


class LLMClient:
    def __init__(
        self,
        text: LLMProfile,
        vision: LLMProfile | None = None,
        *,
        requests_per_minute: int = 0,
        max_retries: int = 2,
    ) -> None:
        """requests_per_minute throttles outgoing calls client-side (0 = no throttle,
        for local servers). max_retries is handed to the OpenAI SDK, which retries
        429s with exponential backoff and honors Retry-After — useful for cloud free
        tiers. The throttle is shared across the text and vision profiles, since both
        draw on the same provider quota."""
        self.text = text
        self.vision = vision or text
        self._max_retries = max_retries
        self._min_interval = 60.0 / requests_per_minute if requests_per_minute > 0 else 0.0
        self._last_call: float | None = None
        self._clients: dict[tuple[str, str, float | None], OpenAI] = {}
        self._text_client = self._client_for(self.text)
        self._vision_client = self._client_for(self.vision)

    def _client_for(self, profile: LLMProfile) -> OpenAI:
        """One OpenAI client per (endpoint, api_key, timeout) — reused across profiles.

        The timeout is part of the key because text and vision commonly share an
        endpoint and key (one local server) but not necessarily a timeout.
        """
        key = (profile.endpoint, profile.api_key, profile.timeout_seconds)
        client = self._clients.get(key)
        if client is None:
            kwargs: dict[str, Any] = {
                "base_url": profile.endpoint,
                "api_key": profile.api_key,
                "max_retries": self._max_retries,
            }
            if profile.timeout_seconds is not None:
                kwargs["timeout"] = profile.timeout_seconds
            client = OpenAI(**kwargs)
            self._clients[key] = client
        return client

    def _throttle(self) -> None:
        """Block until at least _min_interval has elapsed since the last call, so a
        fast cloud endpoint doesn't blow past a free-tier requests-per-minute quota."""
        if self._min_interval <= 0.0:
            return
        if self._last_call is not None:
            wait = self._min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
        self._last_call = time.monotonic()

    def _complete(
        self,
        client: OpenAI,
        profile: LLMProfile,
        messages: list[dict[str, Any]],
        *,
        retries: int,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Send messages, expect JSON. Returns parsed dict, or {"error": ...}.

        Retries up to `retries` times on JSON parse failures before giving up.
        Sampling params are included only when set; response_format and
        extra_body only when the profile asks for them.

        `timeout` overrides the profile's default for this call and also disables
        SDK-level retries. Both halves matter: the SDK retries timeouts, so a
        `timeout` alone would silently mean `timeout * (1 + max_retries)`. Callers
        that hand down a deadline (receipt OCR) need the bound to be the bound.
        """
        kwargs: dict[str, Any] = {"model": profile.model, "messages": messages}
        if profile.temperature is not None:
            kwargs["temperature"] = profile.temperature
        if profile.top_p is not None:
            kwargs["top_p"] = profile.top_p
        if profile.presence_penalty is not None:
            kwargs["presence_penalty"] = profile.presence_penalty
        if profile.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if profile.extra_body:
            kwargs["extra_body"] = profile.extra_body

        if timeout is not None:
            client = client.with_options(timeout=timeout, max_retries=0)

        last_err: dict[str, Any] = {}
        for _attempt in range(retries + 1):
            content = ""
            try:
                self._throttle()
                resp = client.chat.completions.create(**kwargs)
                content = _CODE_FENCE_RE.sub("", resp.choices[0].message.content or "").strip()
                parsed = json.loads(content)
                if not isinstance(parsed, dict):
                    return {"error": f"LLM returned non-object JSON: {parsed!r}"}
                return parsed
            except json.JSONDecodeError as e:
                last_err = {"error": f"JSON parse failure: {e}", "raw": content}
            except Exception as e:
                return {"error": f"LLM call failure: {e}"}
        return last_err

    def complete_json(self, system: str, user: str, *, retries: int = 2) -> dict[str, Any]:
        """Send messages, expect a JSON response. Returns parsed dict.

        On any error returns {"error": "..."} so callers can log-and-skip
        individual transactions without crashing the whole run.
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return self._complete(self._text_client, self.text, messages, retries=retries)

    def complete_json_vision(
        self,
        system: str,
        user: str,
        image_b64: str,
        media_type: str,
        *,
        retries: int = 2,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Like complete_json but attaches a base64-encoded image via image_url.

        media_type should be the MIME type, e.g. "image/jpeg". Uses the vision
        profile (which falls back to the text profile when not configured).
        `timeout` caps this single call — see `_complete`.
        """
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{media_type};base64,{image_b64}"},
                    },
                ],
            },
        ]
        return self._complete(
            self._vision_client, self.vision, messages, retries=retries, timeout=timeout
        )
