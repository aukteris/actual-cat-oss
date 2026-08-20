"""Generate the scrubbed stand-in for a real emailed PDF receipt (ISSUE-001).

The four receipts lost to ISSUE-001 were jsPDF 2.5.1 share-sheet exports with a
very specific shape: one page, no text layer, and exactly one /DCTDecode image
XObject holding a baseline JPEG. That shape is the whole point of the fixture —
it is what routes a PDF to vision OCR rather than the text parser.

The real receipts are personal financial records and this repository is public,
so the committed fixture is generated here instead: same structure, synthetic
receipt image. Run this only to regenerate the fixture.

    venv/bin/python tests/fixtures/pdf/make_receipt_fixture.py
"""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).parent / "jspdf_image_only_receipt.pdf"

# Matches the real receipts: 1482 px wide, portrait, rendered onto US Letter.
IMG_W, IMG_H = 1482, 1400
PAGE_W, PAGE_H = 612, 792

_LINES = [
    ("GENERIC GROCERY CO", 64),
    ("123 Example Street", 36),
    ("Anytown, CA 90000", 36),
    ("", 24),
    ("2026-08-19  13:42", 36),
    ("", 24),
    ("BANANAS               3.18", 40),
    ("OAT MILK              4.99", 40),
    ("COFFEE BEANS         12.50", 40),
    ("SPARKLING WATER       6.29", 40),
    ("", 24),
    ("SUBTOTAL             26.96", 40),
    ("TAX                   2.16", 40),
    ("TOTAL                29.12", 48),
    ("", 24),
    ("VISA ****1234", 36),
    ("THANK YOU", 36),
]


def _receipt_jpeg() -> bytes:
    img = Image.new("RGB", (IMG_W, IMG_H), "white")
    draw = ImageDraw.Draw(img)
    y = 80
    for text, step in _LINES:
        if text:
            draw.text((90, y), text, fill="black")
        y += step
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def build_pdf(jpeg: bytes) -> bytes:
    """Hand-assemble a one-page PDF wrapping `jpeg` as a /DCTDecode XObject.

    Written out by hand rather than through a PDF library so the structure the
    fixture exists to exercise is explicit and cannot drift: no text operators
    anywhere, exactly one image.
    """
    content = f"q {PAGE_W} 0 0 {PAGE_H} 0 0 cm /I0 Do Q\n".encode()
    objects = [
        b"<</Type /Catalog /Pages 2 0 R>>",
        b"<</Type /Pages /Kids [3 0 R] /Count 1>>",
        (
            f"<</Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_W} {PAGE_H}] "
            f"/Resources <</XObject <</I0 4 0 R>>>> /Contents 5 0 R>>"
        ).encode(),
        (
            f"<</Type /XObject /Subtype /Image /Width {IMG_W} /Height {IMG_H} "
            f"/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode "
            f"/Length {len(jpeg)}>>\nstream\n"
        ).encode() + jpeg + b"\nendstream",
        f"<</Length {len(content)}>>\nstream\n".encode() + content + b"endstream",
    ]

    out = bytearray(b"%PDF-1.3\n")
    offsets = []
    for n, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{n} 0 obj\n".encode() + body + b"\nendobj\n"

    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<</Size {len(objects) + 1} /Root 1 0 R>>\n"
        f"startxref\n{xref_at}\n%%EOF\n"
    ).encode()
    return bytes(out)


if __name__ == "__main__":
    OUT.write_bytes(build_pdf(_receipt_jpeg()))
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
