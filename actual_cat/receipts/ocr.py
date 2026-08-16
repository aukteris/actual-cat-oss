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

# Receipt-detection tuning. Calibrated on a real 4284x5712 photo of a Safeway
# receipt lying on a grey floor, where the receipt was 24% of the frame width.
_DETECT_MAX_EDGE = 1000      # detect on a downscale; 24MP through a median is far too slow
_MEDIAN_SIZE = 5             # despeckle before thresholding
_BRIGHT_THRESHOLD = 150      # post-autocontrast luminance counted as "paper"
_COL_FRACTION = 0.45         # a receipt column is bright down much of the frame
_ROW_FRACTION = 0.5          # measured within the column band, not the full width
_PAD_X, _PAD_Y = 0.05, 0.005  # a little margin so edge glyphs aren't shaved
_MIN_CROP_PX = 64            # below this the "receipt" is noise
_MIN_CROP_FRACTION = 0.05    # smaller: detection latched onto a speck
_MAX_CROP_FRACTION = 0.90    # larger: nothing to gain, don't risk clipping


def _profile(mask: Any, axis: str) -> list[int]:
    """Mean brightness per column ("x") or per row ("y"), 0-255.

    Box resampling to a 1px-thick strip averages each column/row — the
    dependency-free equivalent of a numpy axis mean. numpy is deliberately not a
    dependency of this project.

    Reads the strip with tobytes() rather than getdata(): for an "L" image those
    are the same bytes, and getdata() is deprecated in Pillow 14.
    """
    from PIL import Image

    w, h = mask.size
    size = (w, 1) if axis == "x" else (1, h)
    return list(mask.resize(size, Image.Resampling.BOX).convert("L").tobytes())


def _receipt_bbox(img: Any) -> tuple[int, int, int, int] | None:
    """Locate the receipt within the frame: bright paper against a darker surface.

    Returns a full-resolution (x0, y0, x1, y1) box, or None to use the whole frame.

    Receipts are commonly photographed as a narrow strip on a table or floor — the
    motivating one occupied 24% of the frame width, so ~78% of the vision model's
    fixed token budget was spent rendering empty background.
    """
    from PIL import ImageFilter, ImageOps

    W, H = img.size
    # Detect on a downscale: 24MP through a median filter is far too slow, and
    # locating a large bright region needs nothing like that resolution.
    scale = _DETECT_MAX_EDGE / max(W, H)
    if scale >= 1.0:
        scale, small = 1.0, img.convert("L")
    else:
        small = img.convert("L").resize((max(1, round(W * scale)), max(1, round(H * scale))))
    # Median kills specks (the real photo has debris on the floor); autocontrast
    # keeps the threshold from being tied to one lighting condition.
    small = ImageOps.autocontrast(small.filter(ImageFilter.MedianFilter(_MEDIAN_SIZE)))
    mask = small.point(lambda p: 255 if p > _BRIGHT_THRESHOLD else 0)
    sw, sh = mask.size

    cols = _profile(mask, "x")
    xs = [i for i, v in enumerate(cols) if v > _COL_FRACTION * 255]
    if not xs:
        return None
    x0, x1 = xs[0], xs[-1]

    # Judge rows only WITHIN the receipt's column band. A narrow strip never clears
    # a full-width brightness threshold, so a naive whole-row profile collapses the
    # box vertically — it reported 1658px of height for a receipt 5024px tall.
    rows = _profile(mask.crop((x0, 0, x1 + 1, sh)), "y")
    ys = [i for i, v in enumerate(rows) if v > _ROW_FRACTION * 255]
    if not ys:
        return None
    y0, y1 = ys[0], ys[-1]

    padx = (x1 - x0) * _PAD_X
    pady = (y1 - y0) * _PAD_Y
    box = (
        max(0, int((x0 - padx) / scale)),
        max(0, int((y0 - pady) / scale)),
        min(W, int(round((x1 + 1 + padx) / scale))),
        min(H, int(round((y1 + 1 + pady) / scale))),
    )
    cw, ch = box[2] - box[0], box[3] - box[1]
    if cw < _MIN_CROP_PX or ch < _MIN_CROP_PX:
        return None
    fraction = (cw * ch) / (W * H)
    # Too small: detection latched onto a speck. Too large: nothing to gain, and
    # cropping only risks clipping the receipt.
    if not (_MIN_CROP_FRACTION <= fraction <= _MAX_CROP_FRACTION):
        return None
    return box


def _autocrop_receipt(img: Any) -> tuple[Any, float | None]:
    """Crop to the receipt. Returns (image, retained_area_fraction).

    The fraction is None when no crop was applied. Any failure falls back to the
    full frame — a bad crop that clips the total would be worse than the wasted
    tokens this is trying to reclaim.
    """
    try:
        box = _receipt_bbox(img)
        if box is None:
            return img, None
        cropped = img.crop(box)
        W, H = img.size
        return cropped, (cropped.size[0] * cropped.size[1]) / (W * H)
    except Exception:
        return img, None


def _normalize_image(image_bytes: bytes, autocrop: bool = True) -> tuple[bytes, Any, float | None]:
    """Apply EXIF orientation, optionally crop to the receipt, and encode.

    Returns (jpeg_bytes, PIL_Image, crop_fraction). Cropping happens before the
    encode so the first pass and any rotation retry share the same image.
    Returns (original_bytes, None, None) if Pillow can't open the image.
    """
    try:
        from PIL import Image, ImageOps

        img: Image.Image = Image.open(io.BytesIO(image_bytes))
        img = ImageOps.exif_transpose(img)
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGB")
        crop_fraction = None
        if autocrop:
            img, crop_fraction = _autocrop_receipt(img)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        return buf.getvalue(), img, crop_fraction
    except Exception:
        return image_bytes, None, None


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


def _self_rank(result: dict[str, Any]) -> tuple[int, int]:
    """Score a read against its *own* total — lower is better.

    Used to compare two reads of the same receipt (cropped vs full frame) when
    neither total can be assumed correct: a crop that clipped the total line
    reports a different total than the full frame does, so ranking both against
    one shared number would be meaningless.
    """
    return _rotation_rank(result, result.get("total_cents") or 0)


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
    autocrop: bool = True,
) -> dict[str, Any]:
    """OCR a receipt image, with a bounded rotation retry on unreadable results.

    Returns a dict with keys: merchant, date, total_cents, line_items, confidence.
    On failure returns {"error": "..."}.

    Strategy:
    1. Apply EXIF orientation correction, then crop to the receipt.
    2. OCR the corrected image.
    3. If confidence is high, return immediately (1 LLM call, normal path).
    4. If the read looks illegible and we cropped, retry once on the full frame —
       a mis-detected crop is the first thing to rule out.
    5. Otherwise rotate ONLY if the read looks illegible (see _rotation_could_help)
       — a merely-unreconciled read returns as-is.
    6. Try rotations [90°, 270°, 180°] and pick the best-scoring result.
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

    normalized_bytes, pil_img, crop_fraction = _normalize_image(image_bytes, autocrop=autocrop)
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

    def _tag(result: dict[str, Any], fraction: float | None) -> dict[str, Any]:
        """Record what the model was actually shown, so production can tell whether
        cropping is firing and how often it falls back."""
        if "error" not in result:
            result["crop_fraction"] = fraction
            result["cropped"] = fraction is not None
        return result

    best = _attempt(normalized_bytes)
    if best.get("confidence") == "high":
        return _tag(best, crop_fraction)

    # Cropping is an optimization *attempt*, never to be trusted on its own.
    # Detection can put the box in the wrong place — a receipt held at an angle,
    # or another bright object in frame — and a misplaced crop returns a perfectly
    # coherent read of the part it kept. Nothing about that read looks wrong, so no
    # legibility check can catch it; the only reliable test is to read the full
    # frame too and keep whichever reconciles better against its own total.
    if crop_fraction is not None:
        remaining = _remaining()
        if remaining is None or remaining > 0:
            full_bytes, full_img, _ = _normalize_image(image_bytes, autocrop=False)
            uncropped = _attempt(full_bytes)
            if _self_rank(uncropped) < _self_rank(best):
                best, pil_img, crop_fraction = uncropped, full_img, None
            if best.get("confidence") == "high":
                return _tag(best, crop_fraction)

    if not _rotation_could_help(best):
        return _tag(best, crop_fraction)

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

    return _tag(min(candidates, key=lambda r: _rotation_rank(r, total_cents)), crop_fraction)


def ocr_receipt_from_path(
    image_path: Path,
    llm: "LLMClient",
    system_prompt: str,
    schema_text: str,
    *,
    request_timeout_seconds: float | None = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    budget_seconds: float | None = DEFAULT_BUDGET_SECONDS,
    autocrop: bool = True,
) -> dict[str, Any]:
    return ocr_receipt(
        image_path.read_bytes(),
        llm,
        system_prompt,
        schema_text,
        request_timeout_seconds=request_timeout_seconds,
        budget_seconds=budget_seconds,
        autocrop=autocrop,
    )
