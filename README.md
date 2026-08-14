# actual-cat

AI-assisted transaction categorization worker for [Actual Budget](https://actualbudget.org/).

`actual-cat` fills the gaps left by Actual's built-in rule engine:

1. **Categorization** — transactions from new or unknown payees that deterministic
   rules can't match are categorized by an LLM.
2. **Transfer detection** — inverse-amount pairs across accounts that rules didn't
   pair are detected and linked.
3. **Receipt splitting** — a receipt photo posted via iOS Shortcut (or email) is
   OCR'd by the vision LLM, matched to the corresponding bank transaction by exact
   amount, and the transaction is either tagged for review (suggest) or split into
   categorized child rows (apply).
4. **Pending-duplicate resolution** — when a card authorization and its posted
   counterpart both land in the budget as separate rows (a tip added after
   authorization defeats the importer's amount match), the pending row is
   detected and tagged for review, or deleted in apply mode. Off by default.

It runs the built-in rule engine first, then duplicate resolution, then the
transfer pipeline, then the receipt pipeline, then categorization on whatever
remains. By default all LLM pipelines run in **suggest mode**: they write tags to
transaction notes and never touch category fields, create splits, or delete rows
until advanced to apply mode.

## How it works

```
rule engine  →  pending duplicates  →  transfer detection  →  receipt splitting  →  categorization
 (Actual's)      (LLM, off)             (LLM, suggest)        (LLM, suggest)        (LLM, suggest)
```

- Duplicate resolution runs first, before any other pipeline: its apply action is
  deletion, and going first means nothing has been spent on — or written to — a row
  that is about to be removed. With `defer_pending`, every other pipeline also skips
  rows the bank hasn't finalized, so a pending charge is categorized once it clears
  rather than twice.
- Transfers run before categorization so a transfer pair never gets miscategorized.
- Receipt splitting runs before categorization and overrides any existing category —
  this is the one intentional exception to the "don't touch already-categorized
  transactions" rule. Receipts are processed in two passes: the LLM first transcribes
  line items (description + amount), then a second pass assigns each item a category.
- Both the categorization and receipt-categorization passes are **history-aware**: they
  read how similar things were categorized before from the live budget — a payee's prior
  categories for single transactions, and prior categories for the same item description
  for receipt line items — and pass them to the LLM as hints. Because the source is the
  budget itself, your manual corrections feed back in over time. Configurable via the
  `[history]` block (set `enabled = false` to disable).
- Any OpenAI-compatible endpoint works: local llama.cpp, OpenRouter, Google Gemini,
  or OpenAI itself. Text and vision can use different endpoints (`[llm.vision]`).
  API keys come from `.env` (`LLM_API_KEY`, `LLM_VISION_API_KEY`); local servers
  need no key. Provider-specific sampling params go in `[llm.extra_body]`; omit it
  entirely for standard cloud providers. For cloud HTTPS leave `[paths].ca_bundle`
  unset so the system trust store validates public certificates. On a free tier,
  `[llm.rate_limit]` throttles requests (`requests_per_minute`) and tunes 429
  retries (`max_retries`).
- Every decision is written to a JSONL audit log for later review.
- All writes are tags prepended to the notes field; original notes are preserved.

## Modes

Each pipeline has an independent `mode` (`suggest` | `apply`), flipped in
`config.toml` with no redeploy:

| Mode | Behavior |
|---|---|
| `suggest` | Writes `#ai:<slug>` tag to notes only. Category field untouched. |
| `apply` | Sets the category when confidence meets the threshold, tags `#ai-assisted`. Falls back to suggest below threshold. |
| `apply` (receipts) | Creates child split rows, sets their categories and notes, clears parent category. |
| `apply` (duplicates) | Deletes the pending row (soft delete, as the Actual UI does). Only ever a row the bank hasn't finalized; never the posted survivor. |

`Uncertain` results are always tagged, never applied.

## Receipt delivery

Three ingestion paths are supported. All produce the same internal receipt record;
a later extraction step dispatches on `input_kind` so image and text receipts share
one categorization and matching pipeline.

### iOS Shortcut — image (tested)
POST the photo directly from the iOS Photos Shortcut to the HTTP receiver:

```bash
# Start the receiver (exposes POST /receipts/ and POST /receipts/text)
RECEIPT_RECEIVER_TOKEN=<token> venv/bin/python -m actual_cat.receipts.receiver
```

`POST /receipts/` — **Form** body, field `image` (type: File).
Supported formats: JPEG, PNG, WebP, HEIC/HEIF, up to 20 MB.
Optional hint fields: `hint_merchant`, `hint_date`.

### iOS Shortcut — plain text
`POST /receipts/text` — **Form** body, field `text` (type: Text), up to 1 MB.
Useful for forwarding emailed receipts or pasting POS output from a Shortcut.
Same optional hint fields as the image endpoint.

### IMAP email (implemented, not yet end-to-end tested)
Configure `[email]` in `config.toml`. The worker polls the mailbox on each run:
- **Image attachment** → vision OCR path (same as iOS image upload).
- **No image attachment** → plain-text body is treated as a text receipt.

**Issues are expected** — this path has not been exercised against a live mailbox
and likely has rough edges around attachment handling and IMAP edge cases.

## Idempotency

Re-running is always safe:

- The categorization pipeline only sees transactions where `category_id is None`.
- Any transaction already bearing an `#ai:`, `#ai-assisted`, `#ai-suggested-transfer`,
  `#ai-receipt-split` / `#ai-suggested-split`, or `#ai-suggested-duplicate` /
  `#ai-duplicate-review` marker is skipped.
- Off-budget accounts are excluded (they don't use budget categories).
- Pending receipts that go unmatched for more than `expiry_days` are marked expired
  and skipped on future runs.

## Configuration

Copy the examples and fill them in:

```bash
cp config.example.toml config.toml   # non-secret settings
cp .env.example .env                  # secrets only
chmod 600 .env
```

`config.toml` holds the budget name, LLM endpoint, modes, and paths.
Secrets (`ACTUAL_PASSWORD`, `ACTUAL_ENCRYPTION_PASSWORD`, and LLM API keys) live
**only** in `.env` — both files are gitignored.

**LLM configuration** (`[llm]`): any OpenAI-compatible provider works.

| Scenario | What to set |
|---|---|
| Local llama.cpp | `endpoint`, `model`, and `[llm.extra_body]` with llama.cpp-specific params (`top_k`, `min_p`, `chat_template_kwargs`) |
| Cloud (OpenRouter, Gemini, OpenAI) | `endpoint`, `model`, `json_mode = true` — omit `[llm.extra_body]` entirely |
| Split text/vision providers | Add `[llm.vision]` with its own `endpoint`/`model`; fields fall back to `[llm]` when absent |

Standard sampling params (`temperature`, `top_p`, `presence_penalty`) are always
sent. Provider-specific params go in `[llm.extra_body]` and are passed verbatim —
cloud providers that reject unknown fields should omit this table entirely. With no
`[llm.extra_body]` block and no legacy top-level `top_k`/`min_p`/`enable_thinking`
keys, nothing provider-specific is sent (cloud-safe).

For cloud free tiers, an optional `[llm.rate_limit]` table caps request volume:
`requests_per_minute` (0/omitted = no throttle) spaces calls out client-side, and
`max_retries` (default 2) controls how many times the OpenAI SDK retries a 429 with
exponential backoff, honoring the provider's `Retry-After`.

`json_mode = true` (default) sends `response_format: json_object`. Set false for
models that reject it — the response parser strips code fences as a fallback.

API keys for cloud providers come from `.env`:
- `LLM_API_KEY` — text model (omit for keyless local servers)
- `LLM_VISION_API_KEY` — vision model; falls back to `LLM_API_KEY` when unset

TLS to the Actual server is trusted via `paths.ca_bundle`, which the worker exports
as `REQUESTS_CA_BUNDLE` / `SSL_CERT_FILE`. Leave unset for cloud LLM providers so
the system trust store validates their public certificates.

**Pending duplicates** (`[duplicates]`): off unless `enabled = true`, and omitting
the block entirely leaves every other pipeline exactly as it was.

| Key | Meaning |
|---|---|
| `mode` | `suggest` tags the pending row; `apply` **deletes** it at or above the threshold |
| `window_days` | Symmetric search window — a posted row is sometimes dated a day *earlier* |
| `max_uplift_pct` / `max_reduction_pct` | Tip band: how far the posted amount may sit above or below the pending one |
| `auth_hold_max_cents` | Ceiling for treating a small pending row as an authorization probe |
| `defer_pending` | Independent of the matcher: the other pipelines skip rows the bank hasn't finalized |

Before enabling it, replay the matcher over the budget's own history — hand-deleted
pending rows are ground truth for what it should propose, never-pending rows for
what it should not:

```bash
ACTUAL_PASSWORD=... venv/bin/python scripts/backtest_duplicates.py   # read-only
```

## Install & run

```bash
python3 -m venv venv
venv/bin/pip install -e ".[dev]"
venv/bin/pytest -q

# run the worker
venv/bin/python -m actual_cat
```

Lint, type-check, and test via the `Makefile`:

```bash
make lint    # ruff + mypy
make test    # pytest
make check   # lint + test
```

## Deployment

A typical deployment runs the worker on a schedule via a systemd oneshot service + timer.
See [`docs/deployment.md`](docs/deployment.md) for the end-to-end guide (worker,
receipt receiver, reverse proxy, validation), and
[`docs/observability.md`](docs/observability.md) for shipping the audit log to ELK.

## Docs

- [`docs/architecture.md`](docs/architecture.md) — design rationale, pipeline
  architecture, and the suggest→apply verification gates.
- [`docs/deployment.md`](docs/deployment.md) — production deployment and receipt
  splitting setup.
- [`docs/observability.md`](docs/observability.md) — optional ELK/Kibana audit
  dashboards.

## Layout

```
actual_cat/
├── __main__.py        # orchestration
├── config.py          # TOML + .env loading (LLMProfile, Config)
├── llm.py             # OpenAI-compatible client — LLMProfile + LLMClient
├── prompts.py         # system prompts: categorization, transfer, duplicate, receipt extraction/categorization
├── schema.py          # renders the live category schema for the LLM
├── categorization.py  # find_uncategorized + categorization pipeline
├── history.py         # payee + item-description history from the live budget
├── transfers.py       # transfer candidate finder + pairing
├── duplicates.py      # pending-duplicate clustering, rules R1-R5, adjudication, deletion
├── sync_meta.py       # raw_synced_data parsing: is_pending, bank id, descriptor tokens
├── tags.py            # tag markers, slugify, idempotent prepend
├── audit.py           # JSONL audit logger
└── receipts/
    ├── __init__.py
    ├── extract.py     # extractor registry — dispatches inbox meta to ocr.py or text.py
    ├── ocr.py         # vision OCR (image bytes → raw LLM JSON)
    ├── text.py        # plain-text receipt parsing (text → raw LLM JSON)
    ├── parse.py       # source-neutral validation, confidence scoring, amount nudging
    ├── categorize.py  # pass 2: assigns budget categories to extracted line items
    ├── store.py       # JSON-file-per-receipt state machine (inbox→pending→done)
    ├── match.py       # amount-based transaction matching + split application
    ├── receiver.py    # FastAPI HTTP receiver (/receipts/ image, /receipts/text)
    └── ingest.py      # IMAP email poller (not yet end-to-end tested)
```
