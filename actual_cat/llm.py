import json
import re
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
    """

    endpoint: str
    api_key: str
    model: str
    temperature: float | None = 0.2
    top_p: float | None = 0.85
    presence_penalty: float | None = 1.0
    json_mode: bool = True
    extra_body: dict[str, Any] = field(default_factory=dict)


class LLMClient:
    def __init__(self, text: LLMProfile, vision: LLMProfile | None = None) -> None:
        self.text = text
        self.vision = vision or text
        self._clients: dict[tuple[str, str], OpenAI] = {}
        self._text_client = self._client_for(self.text)
        self._vision_client = self._client_for(self.vision)

    def _client_for(self, profile: LLMProfile) -> OpenAI:
        """One OpenAI client per (endpoint, api_key) — reused across profiles."""
        key = (profile.endpoint, profile.api_key)
        client = self._clients.get(key)
        if client is None:
            client = OpenAI(base_url=profile.endpoint, api_key=profile.api_key)
            self._clients[key] = client
        return client

    def _complete(
        self,
        client: OpenAI,
        profile: LLMProfile,
        messages: list[dict[str, Any]],
        *,
        retries: int,
    ) -> dict[str, Any]:
        """Send messages, expect JSON. Returns parsed dict, or {"error": ...}.

        Retries up to `retries` times on JSON parse failures before giving up.
        Sampling params are included only when set; response_format and
        extra_body only when the profile asks for them.
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

        last_err: dict[str, Any] = {}
        for _attempt in range(retries + 1):
            content = ""
            try:
                resp = client.chat.completions.create(**kwargs)
                content = _CODE_FENCE_RE.sub("", resp.choices[0].message.content or "").strip()
                return json.loads(content)  # type: ignore[no-any-return]
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
        self, system: str, user: str, image_b64: str, media_type: str, *, retries: int = 2
    ) -> dict[str, Any]:
        """Like complete_json but attaches a base64-encoded image via image_url.

        media_type should be the MIME type, e.g. "image/jpeg". Uses the vision
        profile (which falls back to the text profile when not configured).
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
        return self._complete(self._vision_client, self.vision, messages, retries=retries)
