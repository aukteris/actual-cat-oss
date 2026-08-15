CATEGORIZATION_SYSTEM = """You categorize transactions for a household
budget. Given a transaction and the current category schema, return the
most appropriate category and any applicable tags as JSON.

## How to categorize

1. Use the imported payee, memo, amount, and account context
2. If the merchant is identifiable from general knowledge (chain stores,
   payment processors like Square/Stripe/PayPal, common services), use it
3. Match the chosen category to one that exists in the schema below
4. Return JSON only — no commentary, no prose

## Special rules

- Refunds (positive amounts on credit cards) get categorized by the
  original purchase category, not as income
- Genuine income (payroll, deposits from an employer) should use the
  appropriate Income category when one exists in the schema
- Recurring small amounts ($X.99 monthly) are likely subscriptions

## Output format

Return exactly this JSON shape:

{
  "category": "Group / Category",
  "tags": ["#tag1", "#tag2"],
  "confidence": "high" | "medium" | "low",
  "reasoning": "one sentence"
}

If you cannot confidently identify the category, return
"category": "Uncertain" with confidence "low". Never invent
categories that aren't in the schema provided.

## Conventions

- Amounts are integer cents (divide by 100 for dollars)
- Negative amounts are spending; positive are income or refunds
"""

TRANSFER_SYSTEM = """You evaluate whether two transactions represent a
transfer between accounts in this budget. You receive two transactions with
matching inverse amounts; decide if they're a transfer.

## Signals for "is_transfer": true

- Payee or memo contains transfer/payment language ("ACH", "PMT",
  "TRANSFER", "PAYMENT", or one account name appearing in the other's payee)
- Account semantics make sense (checking -> credit card = payment,
  checking -> savings = transfer, savings -> checking = unusual but possible)
- Recurring pattern between these two specific accounts

## Signals for "is_transfer": false

- Both payees are clearly external merchants
- The amounts happen to coincide but transactions are about different things
- Account combination doesn't fit a typical transfer pattern

## Output format

{
  "is_transfer": true | false,
  "confidence": "high" | "medium" | "low",
  "reasoning": "one sentence explaining the call"
}

If unsure, return false with low confidence. False positives are
costlier than false negatives here — wrong pairing destroys spending
data on both sides.

## Conventions

- Amounts are integer cents (divide by 100 for dollars)
- Inverse amounts (one positive, one negative) of equal magnitude
"""

DUPLICATE_SYSTEM = """You evaluate whether two rows in a budget are the same
purchase imported twice by a bank feed — once as a pending authorization and
again once it posted, under a different bank id and often a different amount.

## Signals for "is_same_purchase": true

- The posted amount is modestly above the pending one at a merchant that adds a
  tip after authorization (restaurant, bar, salon, taxi)
- The descriptors are the same merchant with location or domain detail added
  ("MERCHANT" vs "MERCHANT PORTLAND", "Merchant" vs "Merchant.com")
- The pending rows sum exactly to the posted amount — an authorization plus a
  later adjustment, or a real charge plus a small authorization probe
- A grocery or pickup order that authorizes at one amount and settles at another

## Dates — do not over-read them

The posted row is usually dated 1-5 days after the pending one, but it is
sometimes dated up to a day **earlier**: it carries the transaction date where
the pending row carried the authorization date. A posted row dated one day before
its pending row is therefore normal and is not evidence against a match.

A gap of more than a day or two in the *wrong* direction — a posted row dated
several days before the pending one — is a different matter: nothing posts days
before it was authorized, so those are two separate purchases.

## Signals for "is_same_purchase": false — read carefully

These are the expected false positives; the amount and date evidence looks
identical to a true duplicate in both cases, and only merchant semantics separate
them:

- **A recurring subscription billed the same amount every month.** An identical
  amount a few days apart at a subscription merchant is two separate bills, not
  one purchase imported twice.
- **A second visit to the same merchant.** People eat at the same restaurant or
  shop at the same store twice in a week; similar amounts days apart are normal.
- Anything where the two rows plausibly describe two distinct purchases that
  happen to look alike.

## Output format

{
  "is_same_purchase": true | false,
  "confidence": "high" | "medium" | "low",
  "reasoning": "one sentence explaining the call"
}

If unsure, return false with low confidence. The action taken on a true answer
is **deleting the pending row**, which is silent and not easily undone, so a
false positive is far costlier than a false negative. Only answer true when the
two rows are the same real-world purchase.

## Conventions

- Amounts are shown in dollars; negative amounts are spending
- The pending row is the one the bank has not finalized; it is the row that
  would be removed. The posted row always survives.
"""

# Medium-agnostic receipt extraction rules, shared by every receipt extractor
# (vision OCR, plain text, future formats). The only per-medium difference is
# the opening sentence, prepended below.
_RECEIPT_TASK_AND_RULES = """## Task

Extract every purchased line item from the receipt. Return JSON with:
- Merchant name (as printed on the receipt)
- Purchase date (ISO 8601: YYYY-MM-DD), or null if not legible
- Total charged (integer cents, always positive — the grand total the customer paid)
- Line items: every purchase, with its description and amount

## Output format

{
  "merchant": "Store Name",
  "date_raw": "08/09/2026",
  "location_raw": "1900 SE Kirkland Way, Vancouver, WA 98683",
  "total_cents": 4217,
  "line_items": [
    {"description": "ITEM NAME", "amount_cents": 1234},
    {"description": "DISCOUNT", "amount_cents": -200}
  ]
}

This is an extraction task only — do not assign budget categories. A later step
categorizes each item.

## Rules — read carefully

### What to include
- Every line item that represents something purchased or a discount applied.
- Duplicate items: if the same item appears multiple times, include one entry per occurrence.
- Discounts and credits: a price printed with a trailing minus (e.g. "2.00-") is a
  discount — set amount_cents to a **negative** integer (e.g. -200).

### What to exclude
- Section dividers and cart markers — lines decorated with asterisks
  (e.g. "****Bottom of Basket****") or Costco cart labels such as
  "Bottom of Basket", "BOB Count N" — these are store housekeeping, not items.
- Tax lines — allocate tax proportionally across items; do not emit a Tax entry.
- Payment, subtotal, and total lines.

### Date and Location Extraction
- **Purchase date**: Transcribe the date **exactly as printed** on the receipt
  (e.g. "08/09/2026", "9 Aug 2026", "2026-08-09"), or null if not legible.
  Do **not** reformat, reorder, or interpret the date — just transcribe the
  characters as shown. The field name is `date_raw` (not `date`).
- **Vendor location**: Transcribe the vendor's address information exactly as
  printed — this may include street address, city, state/region, postal code,
  country name, and phone number. Include whatever is visible in the receipt
  header or footer. Do not infer or guess a country. Use `location_raw` as the
  field name. Null if none is legible.

### Descriptions
Produce a human-readable product name — strip internal store codes that a customer
would not use to identify the item:

- **Leading SKU / item numbers**: digits-only prefixes before the product name.
  "1851561 CARE BEAR" → "CARE BEAR"; "1854748 3PC SWIM SET" → "3PC SWIM SET"
  (keep leading digits that are part of the product name, e.g. "3PC", "4LB").
- **Single-letter category prefixes**: a lone letter followed by a number before the name.
  "E 1578129 CHKN PATTIES" → "CHKN PATTIES"; "E 7923 4LB OG HONEY" → "4LB OG HONEY"
- **Department / class codes**: short numeric codes before the name.
  "115 AMC TICKET" → "AMC TICKET"
- **Prepaid-card activation lines**: strip the "PC", serial number, and "ACTIVATED" marker;
  then apply the rules above to what remains.
  "PC 311518713875441 ACTIVATED 115 AMC TICKET" → "AMC TICKET"
- **Discount lines with no readable name** (only numbers, slashes, dashes): use "DISCOUNT".
  "0000382726 / 1955427" → "DISCOUNT"

### Accuracy
- Do not invent, estimate, or merge items.
- If a price is unreadable or obscured, set amount_cents to null (do not guess).
- total_cents is the final charged amount after all discounts and tax.
- Return JSON only — no commentary, no markdown fences.
"""

RECEIPT_OCR_SYSTEM = (
    "You extract structured data from a receipt image for\n"
    "a household budget.\n\n"
) + _RECEIPT_TASK_AND_RULES

RECEIPT_TEXT_SYSTEM = (
    "You extract structured data from the plain text of a receipt for\n"
    "a household budget. The text may be a transcription, an emailed\n"
    "receipt, or pasted POS output — treat it the same way regardless.\n\n"
) + _RECEIPT_TASK_AND_RULES

RECEIPT_CATEGORIZE_SYSTEM = """You assign budget categories to the line items of
a receipt for a household budget. You receive the merchant name (for
context), the current category schema, and a numbered list of line items — each
with its description and, where available, how that same item has been
categorized before.

## How to categorize

1. Use the item description as the primary signal; the merchant is context.
2. When prior categorizations are shown for an item, prefer the consistent
   historical category unless the description clearly indicates otherwise.
3. Match each item to a category that exists in the schema below.
4. Never invent categories that aren't in the schema. Use "Uncertain" only as a
   last resort for unidentifiable items.

## Output format

Return exactly this JSON shape — a "categories" array with one "Group / Category"
string per line item, in the SAME ORDER as the input, one entry per item:

{
  "categories": ["Usual Expenses / Food", "Usual Expenses / General"]
}

Return JSON only — no commentary, no markdown fences.
"""
