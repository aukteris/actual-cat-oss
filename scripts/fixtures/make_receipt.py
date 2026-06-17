"""
Generate the synthetic receipt fixture used by `seed_dev_budget.py --receipts`.

This produces a fully fabricated warehouse-style receipt — no real membership
number, payment card, store address, or photo EXIF (GPS/device/timestamp). The
line items are tuned to total $317.37 so the staged receipt matches the
-$317.37 transaction the seed helper creates, exercising the OCR → match flow
end to end.

Run:
    venv/bin/python scripts/fixtures/make_receipt.py
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).parent / "costco_receipt.jpg"
TARGET_TOTAL = 317.37

# Fabricated line items. A final weighted item absorbs the remainder so the
# subtotal lands exactly on TARGET_TOTAL (weighed produce/meat carry odd cents
# on real receipts, so this stays plausible).
ITEMS = [
    ("ORG BANANAS", 5.99),
    ("ROTISSERIE CHKN", 4.99),
    ("PAPER TOWELS 12", 21.99),
    ("OLIVE OIL 2L", 18.99),
    ("COFFEE 3LB", 16.99),
    ("EGGS 24CT", 8.49),
    ("ATLANTIC SALMON", 34.99),
    ("LAUNDRY DETERG", 19.99),
    ("CHEDDAR 2LB", 13.99),
    ("MIXED NUTS", 19.99),
    ("DISH SOAP", 11.99),
    ("TOILET PAPER 30", 21.99),
    ("FROZEN BERRIES", 13.99),
    ("PARMESAN WEDGE", 16.99),
    ("GREEK YOGURT", 7.49),
    ("SPARKLING WTR 24", 11.99),
    ("TRASH BAGS 200", 18.99),
]
_remainder = round(TARGET_TOTAL - sum(p for _, p in ITEMS), 2)
ITEMS.append(("GROUND BEEF /LB", _remainder))

assert round(sum(p for _, p in ITEMS), 2) == TARGET_TOTAL


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", size)


def main() -> None:
    W = 700
    margin = 50
    line_h = 34
    header_lines = 6
    footer_lines = 8
    H = margin * 2 + (header_lines + len(ITEMS) + footer_lines) * line_h

    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    body = font(24)
    big = font(40)

    y = margin
    d.text((margin, y), "COSTCO WHOLESALE", font=big, fill="black")
    y += line_h + 12
    for ln in ("Warehouse #000", "123 Example St", "Anytown, ST 00000",
               "Member 000000000000", "-" * 34):
        d.text((margin, y), ln, font=body, fill="black")
        y += line_h

    for name, price in ITEMS:
        d.text((margin, y), name, font=body, fill="black")
        amt = f"{price:.2f}"
        w = d.textlength(amt, font=body)
        d.text((W - margin - w, y), amt, font=body, fill="black")
        y += line_h

    y += 6
    for label, val in (("-" * 34, ""), ("SUBTOTAL", f"{TARGET_TOTAL:.2f}"),
                       ("TAX", "0.00"), ("TOTAL", f"{TARGET_TOTAL:.2f}"),
                       ("-" * 34, "")):
        d.text((margin, y), label, font=body, fill="black")
        if val:
            w = d.textlength(val, font=body)
            d.text((W - margin - w, y), val, font=body, fill="black")
        y += line_h
    d.text((margin, y), "XXXXXXXXXXXX0000  CREDIT", font=body, fill="black")
    y += line_h
    d.text((margin, y), "Thank you - synthetic test receipt", font=body, fill="black")

    # save with no exif metadata
    img.save(OUT, "JPEG", quality=88)
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes), total ${TARGET_TOTAL:.2f}")


if __name__ == "__main__":
    main()
