"""Receipt OCR via vision LLM — parse image to structured line items."""

from __future__ import annotations

import base64
import io
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .parse import compute_confidence, validate_receipt

if TYPE_CHECKING:
    from ..llm import LLMClient

# Back-compat aliases — the normalizer/scorer moved to parse.py (source-neutral).
# Existing imports of these private names (e.g. tests) keep working.
_compute_confidence = compute_confidence
_validate_response = validate_receipt

# Rotation candidates tried when the initial (EXIF-normalized) pass scores low.
# PIL rotates CCW; [90, 270, 180] covers the common sideways-photo cases first.
_ROTATIONS = [90, 270, 180]

# Defaults for the OCR time bounds. One vision pass over a phone-camera receipt
# costs ~100s against a large local model, so four passes overrun a 600s systemd
# TimeoutStartSec — the run gets killed mid-call and logs nothing at all.
DEFAULT_REQUEST_TIMEOUT_SECONDS = 180.0
DEFAULT_BUDGET_SECONDS = 420.0

# Rotation-gate thresholds. Below this many line items, or above this fraction of
# unreadable amounts, the model plausibly couldn't read the image at all and a
# rotation is worth trying. Calibrated against a real 76-item receipt that scored
# low with 1 unreadable amount (1.3%): an "any unreadable amount" rule would have
# burned three useless ~100s rotations on it.
_MIN_ITEMS_FOR_LEGIBLE_READ = 3
_MAX_UNREADABLE_FRACTION = 0.5


def _normalize_image(image_bytes: bytes) -> tuple[bytes, Any]:
    """Apply EXIF orientation and return (jpeg_bytes, PIL_Image).

    Returns (original_bytes, None) if Pillow can't open the image.
    """
    try:
        from PIL import Image, ImageOps

        img: Image.Image = Image.open(io.BytesIO(image_bytes))
        img = ImageOps.exif_transpose(img)
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        return buf.getvalue(), img
    except Exception:
        return image_bytes, None


def _rotate_image(img: Any, degrees: int) -> bytes:
    """Rotate a PIL Image CCW by degrees and return JPEG bytes."""
    rotated = img.rotate(degrees, expand=True)
    if rotated.mode in ("RGBA", "LA", "P"):
        rotated = rotated.convert("RGB")
    buf = io.BytesIO()
    rotated.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _encode_image(image_bytes: bytes) -> tuple[str, str]:
    """Return (base64_string, media_type). Sniffs JPEG vs PNG by magic bytes."""
    if image_bytes[:4] == b"\x89PNG":
        media_type = "image/png"
    else:
        media_type = "image/jpeg"
    return base64.b64encode(image_bytes).decode(), media_type


def _rotation_rank(result: dict[str, Any], total_cents: int) -> tuple[int, int]:
    """Scoring key for picking the best rotation candidate.

    Lower is better: (0=high/1=low/2=error, abs_diff).
    """
    if "error" in result:
        return (2, 0)
    conf = result.get("confidence", "low")
    readable = [
        i["amount_cents"] for i in result.get("line_items", [])
        if isinstance(i.get("amount_cents"), int)
    ]
    diff = total_cents - sum(readable)
    conf_rank = 0 if conf == "high" else 1
    return (conf_rank, abs(diff))


def _rotation_could_help(result: dict[str, Any]) -> bool:
    """Would re-OCRing this image at another orientation plausibly do better?

    Only when the pass shows the model couldn't read the image: a structural
    failure, almost no line items recovered, or most amounts unreadable. A read
    that recovered a coherent receipt whose amounts merely don't reconcile is
    evidence the orientation was fine — rotating it just pays for the same bad
    read three more times.
    """
    if "error" in result:
        return True

    items = result.get("line_items", [])
    if len(items) < _MIN_ITEMS_FOR_LEGIBLE_READ:
        return True

    unreadable: int = result.get(
        "unreadable_count",
        sum(1 for i in items if not isinstance(i.get("amount_cents"), int)),
    )
    return unreadable / len(items) > _MAX_UNREADABLE_FRACTION


def ocr_receipt(
    image_bytes: bytes,
    llm: "LLMClient",
    system_prompt: str,
    schema_text: str,
    *,
    request_timeout_seconds: float | None = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    budget_seconds: float | None = DEFAULT_BUDGET_SECONDS,
) -> dict[str, Any]:
    """OCR a receipt image, with a bounded rotation retry on unreadable results.

    Returns a dict with keys: merchant, date, total_cents, line_items, confidence.
    On failure returns {"error": "..."}.

    Strategy:
    1. Apply EXIF orientation correction.
    2. OCR the corrected image.
    3. If confidence is high, return immediately (1 LLM call, normal path).
    4. Otherwise rotate ONLY if the read looks illegible (see _rotation_could_help)
       — a merely-unreconciled read returns as-is.
    5. Try rotations [90°, 270°, 180°] and pick the best-scoring result.
       Best = highest confidence, then smallest sum-vs-total diff.

    `budget_seconds` caps total OCR wall time across every pass, and each request
    is capped at the smaller of `request_timeout_seconds` and the budget remaining.
    Together these make the ceiling absolute, so one slow receipt can't run long
    enough for the service manager to kill the whole run. Pass None to either to
    disable that bound.
    """
    user = (
        f"Budget category schema:\n\n{schema_text}\n\n"
        "Please extract the receipt data from the attached image."
    )

    normalized_bytes, pil_img = _normalize_image(image_bytes)
    deadline = time.monotonic() + budget_seconds if budget_seconds is not None else None

    def _remaining() -> float | None:
        return None if deadline is None else deadline - time.monotonic()

    def _attempt(img_bytes: bytes) -> dict[str, Any]:
        remaining = _remaining()
        timeout = request_timeout_seconds
        if remaining is not None:
            timeout = remaining if timeout is None else min(timeout, remaining)
        b64, media_type = _encode_image(img_bytes)
        raw = llm.complete_json_vision(system_prompt, user, b64, media_type, timeout=timeout)
        try:
            return _validate_response(raw)
        except ValueError as e:
            return {"error": f"OCR parse error: {e}"}

    best = _attempt(normalized_bytes)
    if best.get("confidence") == "high":
        return best
    if not _rotation_could_help(best):
        return best

    # Illegible read — try rotations and keep the best result
    total_cents = best.get("total_cents", 0)
    candidates: list[dict[str, Any]] = [best]

    if pil_img is not None:
        for deg in _ROTATIONS:
            remaining = _remaining()
            if remaining is not None and remaining <= 0:
                break  # out of budget; go with what we have
            try:
                rotated_bytes = _rotate_image(pil_img, deg)
            except Exception:
                continue
            result = _attempt(rotated_bytes)
            candidates.append(result)
            if result.get("confidence") == "high":
                break  # can't do better

    return min(candidates, key=lambda r: _rotation_rank(r, total_cents))


def ocr_receipt_from_path(
    image_path: Path,
    llm: "LLMClient",
    system_prompt: str,
    schema_text: str,
    *,
    request_timeout_seconds: float | None = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    budget_seconds: float | None = DEFAULT_BUDGET_SECONDS,
) -> dict[str, Any]:
    return ocr_receipt(
        image_path.read_bytes(),
        llm,
        system_prompt,
        schema_text,
        request_timeout_seconds=request_timeout_seconds,
        budget_seconds=budget_seconds,
    )
