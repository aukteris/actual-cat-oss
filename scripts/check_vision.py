"""
Smoke-test the configured LLM's vision capability before enabling the receipts pipeline.

Generates a minimal synthetic PNG (no Pillow needed — raw bytes assembled with
stdlib struct/zlib), sends it to the configured LLM endpoint via complete_json_vision,
and prints the response.

Exit 0: response has no "error" key (model accepted the image).
Exit 1: LLM error or JSON parse failure.

Run:
    venv/bin/python scripts/check_vision.py
"""

import os
import struct
import sys
import tomllib
import zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _minimal_png(width: int = 4, height: int = 4) -> bytes:
    """Build a tiny white PNG entirely with stdlib — no Pillow dependency."""
    def chunk(tag: bytes, data: bytes) -> bytes:
        length = struct.pack(">I", len(data))
        crc = struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        return length + tag + data + crc

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit RGB

    # Build raw image rows: filter byte 0x00 + white pixels (0xFF 0xFF 0xFF)
    raw_row = b"\x00" + b"\xFF\xFF\xFF" * width
    raw_data = raw_row * height
    idat = zlib.compress(raw_data)

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", idat)
        + chunk(b"IEND", b"")
    )


def main() -> int:
    with open("config.toml", "rb") as f:
        cfg = tomllib.load(f)

    llm_cfg = cfg["llm"]
    ca_bundle = cfg.get("paths", {}).get("ca_bundle")
    if ca_bundle:
        os.environ["REQUESTS_CA_BUNDLE"] = ca_bundle
        os.environ["SSL_CERT_FILE"] = ca_bundle

    from actual_cat.config import _load_llm_profiles
    from actual_cat.llm import LLMClient

    _, vision_profile = _load_llm_profiles(llm_cfg)
    llm = LLMClient(vision_profile, vision_profile)

    import base64
    png_bytes = _minimal_png()
    b64 = base64.b64encode(png_bytes).decode()

    print(f"Model  : {vision_profile.model}")
    print(f"Endpoint: {vision_profile.endpoint}")
    print(f"Image  : {len(png_bytes)} bytes (synthetic 4×4 white PNG)")
    print("Sending vision request...")

    system = (
        "You analyze images. If given a receipt, extract data as JSON. "
        'If given a test image, return {"test": "ok"}.'
    )
    user = (
        'This is a vision capability test. '
        'Return JSON {"test": "ok", "model": "<your model name>"}.'
    )

    result = llm.complete_json_vision(system, user, b64, "image/png")
    import json
    print("\nResponse:", json.dumps(result, indent=2))

    if "error" in result:
        print(
            "\nFAIL — model returned an error. "
            "Check that the endpoint supports image_url input.",
            file=sys.stderr,
        )
        return 1

    print("\nPASS — vision endpoint is functional.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
