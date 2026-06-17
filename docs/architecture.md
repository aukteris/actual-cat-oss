# Architecture & Design Rationale

Design notes for `actual-cat`, an AI-assisted transaction categorization worker
for [Actual Budget](https://actualbudget.org). This document covers *why* the
worker is built the way it is. For the shipped module layout see the
[README](../README.md); for deployment see [deployment.md](deployment.md).

## What this project does

`actual-cat` is a Python worker that fills the gaps left by Actual Budget's
deterministic rule engine. It runs periodically (hourly via systemd timer in a
typical deployment), connects to an Actual server via the `actualpy` library,
runs Actual's existing rule engine first, then applies LLM reasoning only to the
residuals:

1. **Categorizes transactions from new/unknown payees** the rule engine couldn't handle.
2. **Detects transfers between accounts** that weren't paired by rules.
3. **Splits receipts** matched to a bank transaction by exact amount.

Output is either written to transaction notes as suggestions (the safe default)
or applied directly to the category field after a validation period. The worker is
designed for personal/household use; it is not multi-tenant.

## Why AI here, and where it does NOT belong

Actual Budget already has a strong deterministic rule engine that handles
same-payee matching cleanly. **AI is not a replacement for rules** — same-payee
categorization is the rule engine's job. AI's value is in the cases rules can't
handle:

- **New merchants** the user has never transacted with before (no payee-default
  category exists yet).
- **Cryptic payee strings** from payment processors (`SQ *MERCHANT_NAME`,
  `STRIPE*ID`, `PP*MERCHANT`) where merchant identity requires inference.
- **Variable-amount transfers** that don't match a fixed-amount rule pattern.
- **Cross-account transfer pairs** where one side has a custom payee that doesn't
  match the other.

## Explicitly out of V1 scope

- **Payee cleanup / consolidation.** When the imported payee string varies for the
  same merchant (`WHOLE FOODS MKT 12345` vs `WHOLEFDS #5678`), AI could merge them
  into a clean canonical payee. This is a distinct use case with its own design
  considerations (semantic similarity, conservative thresholds for destructive
  operations) and stays manual for V1.
- **Web lookup for unknown merchants.** When the LLM can't identify a merchant from
  training knowledge alone, a web search could provide enrichment. Adds privacy
  considerations (leaking transaction info externally) and infrastructure.
- **Bank sync triggering.** `actualpy` exposes `run_bank_sync()`, but the worker
  stays focused on categorizing what's already in the budget. Sync runs through
  actual-server's own schedule.

## Pipeline architecture

The worker runs these pipelines in order per invocation:

```
Periodic trigger
  │
  ├── 1. Run Actual's rule engine on uncategorized transactions
  │      (handles same-payee cases; no LLM involvement)
  │
  ├── 2. Transfer detection pipeline
  │      - Find inverse-amount pairs across accounts within window
  │      - LLM evaluates each candidate pair
  │      - Pair high-confidence matches (in apply mode), else tag for review
  │
  ├── 3. Receipt splitting pipeline
  │      - OCR/parse pending receipts, match to a bank txn by exact amount
  │      - Split into categorized child rows (apply) or tag the match (suggest)
  │
  ├── 4. Categorization pipeline
  │      - Re-fetch transactions still uncategorized after the steps above
  │      - LLM infers category from payee + memo + amount + account
  │      - Apply category (apply mode, high confidence) or tag for review
  │
  └── Commit changes back to Actual server + write structured audit log
```

Transfer detection runs before categorization so a transfer pair never gets
miscategorized as an expense. Receipt splitting runs before categorization and is
the one intentional exception to "don't touch already-categorized transactions."

## Architectural decisions

- **`actualpy` library** for Actual server interaction. Downloads a local SQLite
  copy of the budget via SQLAlchemy, supports E2EE budgets via
  `encryption_password`, syncs writes back via `actual.commit()`.
- **OpenAI-compatible client** for the LLM. Any OpenAI-compatible endpoint works —
  local llama.cpp, OpenRouter, Gemini, or OpenAI itself. Text and vision can use
  different endpoints.
- **systemd timer + venv**, not a Docker container. The worker is short-running
  batch code that benefits from fast iteration. Container deployment can be added
  later if isolation matters.
- **Two-mode operation: `suggest` and `apply`,** configurable per-pipeline. Each
  pipeline starts in `suggest` for a validation period (~2 weeks), then advances to
  `apply` independently based on observed agreement rates.
- **Idempotency via marker tags in notes.** The worker prepends markers (`#ai:<slug>`,
  `#ai-assisted`, `#ai-suggested-transfer`, `#ai-receipt-split` / `#ai-suggested-split`)
  to transaction notes and skips any transaction already bearing one. Re-running is
  always safe; original notes are preserved.
- **Dynamic schema from Actual at runtime, not static in code.** The worker fetches
  the current category tree each run and embeds it in the system prompt, preventing
  staleness when the user adds/renames categories.
- **Structured JSON output from the LLM.** Parsed deterministically; no regex
  extraction (with a code-fence-stripping fallback for models that won't honor
  `response_format`).
- **History-aware prompting.** The categorization and receipt-categorization passes
  read how similar things were categorized before from the live budget and pass that
  to the LLM as hints, so manual corrections feed back over time.
- **JSONL audit log** for every decision (category chosen, transfer evaluated,
  confidence, reasoning, mode). Useful standalone for review during the validation
  period, and shippable to Elasticsearch later (see [observability.md](observability.md)).

## Configuration

The runtime config lives in `config.toml`; secrets live only in `.env`. Both are
gitignored, with `config.example.toml` / `.env.example` checked in as templates.
See the [README](../README.md#configuration) for the full configuration reference.
A minimal sketch:

```toml
[actual]
base_url = "https://actual.example.com"   # Actual server URL
file = "My Budget"                         # Budget file name
# password and encryption_password come from .env, never config.toml

[llm]
endpoint = "http://llm-host:8900/v1"       # OpenAI-compatible endpoint
model = "your-model"
temperature = 0.2                          # Low for deterministic JSON output

[categorization]
mode = "suggest"                           # "suggest" or "apply"
apply_confidence_threshold = "high"        # "high" | "medium" | "low"

[transfers]
mode = "suggest"
window_days = 3                            # Match window for inverse-amount candidates
apply_confidence_threshold = "high"

[audit]
log_path = "logs/actual-cat.jsonl"         # Relative in dev; absolute in prod

[paths]
ca_bundle = "/etc/ssl/certs/ca-certificates.crt"  # CA bundle for Actual TLS; omit for cloud
```

Secrets in `.env`:

```bash
ACTUAL_PASSWORD=...                # Actual server password
ACTUAL_ENCRYPTION_PASSWORD=...     # E2EE budget password
```

### Dev vs production differences

| Aspect | Dev | Production |
|---|---|---|
| `actual.base_url` | Local dev Actual instance (`http://localhost:5006`) or a separate dev budget file on the prod server | `https://actual.example.com` |
| `llm.endpoint` | Same endpoint (if reachable) or a mock | `http://llm-host:8900/v1` |
| `paths.ca_bundle` | Often unused (HTTP dev server) | Set for TLS validation, or omit for cloud LLMs |
| `audit.log_path` | Relative to project root | Absolute, e.g. `/opt/actual-cat/logs/actual-cat.jsonl` |
| Run mechanism | `python -m actual_cat` | systemd timer |

## Module layout

The shipped code lives under `actual_cat/` (orchestration in `__main__.py`, config
loading in `config.py`, the OpenAI-compatible client in `llm.py`, prompts in
`prompts.py`, the live-schema renderer in `schema.py`, the categorization and
transfer pipelines, history hints, tag helpers, the audit logger, and the
`receipts/` subpackage). The [README](../README.md#layout) has the full annotated
tree — prefer reading the modules themselves over duplicating skeletons here.

## Testing strategy

### Unit tests

Cover the pure-logic modules that don't need external services — marker-tag
detection and append idempotency, schema-text rendering (tombstoned/income groups
excluded), prompt rendering with missing optional fields, and the
apply/suggest/idempotency branches of each pipeline driven by a mocked LLM client.
A simple mock that pops queued responses off a list is sufficient:

```python
class MockLLM:
    def __init__(self, responses: list[dict]):
        self.responses = responses
        self.calls = []

    def complete_json(self, system: str, user: str) -> dict:
        self.calls.append((system, user))
        return self.responses.pop(0)
```

### Integration tests

Run against a real Actual server in dev:

1. Spin up an Actual server instance (Docker, no E2EE for simplicity).
2. Create a test budget with known accounts and categories.
3. Add 10-20 deliberately-crafted transactions covering clear categorization
   candidates, ambiguous ones, obvious transfer pairs, and false-positive amount
   matches.
4. Run `python -m actual_cat` with `mode = "suggest"`.
5. Inspect the resulting notes in the Actual UI and assert expected tags.

Don't mock the LLM in integration tests — actually call your LLM endpoint. The
point is verifying the model's behavior matches expectations.

## Verification gates (post-deployment)

Three stages, advancing independently for each pipeline:

**Stage 1 — Suggest-mode validation (~2 weeks).** Worker runs on schedule without
crashes; `#ai:*` notes appear on otherwise-uncategorized transactions. Tally
agreement between AI suggestion and human categorization. Passes when
categorization agreement ≥90% and transfer detection agreement ≥95%.

**Stage 2 — Categorization apply mode.** Switch `categorization.mode = "apply"`;
transfers stay in suggest. Monitor override rate weekly. Passes when override rate
≤5% over 4 weeks.

**Stage 3 — Transfer apply mode.** Only advance after Stage 2 has been stable for a
month. Switch `transfers.mode = "apply"`. Continue monitoring; the bar is higher
because mis-pairs are destructive. Passes when override rate stays low and no
false-positive transfer pairs land.

Different thresholds reflect different blast radii: a wrong category is annoying; a
wrong transfer pairing corrupts spending data on both sides.

## Risks and mitigations

- **LLM picks wrong category for a class of merchants.** The suggest-mode period
  catches this. If a pattern emerges, add a deterministic rule in Actual's rule
  engine — the worker only invokes the LLM for residuals, so rules supersede.
- **Schema drift.** The worker fetches the live schema each run, so new categories
  are picked up on the next run.
- **Cents-vs-dollars hallucination.** Explicit prompt instruction plus structured
  JSON output. Spot-check early decisions in suggest mode.
- **Concurrent writes** between the worker and human edits in the UI. `actualpy`
  uses CRDT sync, so concurrent edits should merge. Test in dev.
- **False-positive transfer pairings** — the worst failure mode. Mitigations: high
  confidence threshold for apply mode, a transfer-specific validation period, and
  enabling transfer apply mode only after categorization apply mode is stable.
- **LLM unreachable.** The client returns an error result rather than raising; the
  pipelines log and skip individual failures, completing the run partially rather
  than crashing.

## Future enhancements (post-V1)

- **Payee cleanup pipeline** — detect that an imported payee is a variant of an
  existing cleaned payee; suggest or apply the merge (conservative, suggest-only by
  default).
- **Web lookup for unknown merchants** — when confidence is low, query a search or
  merchant-identification API with just the merchant identifier, then re-prompt.
- **Pattern detection across transactions** — surface recurring subscriptions the
  user may not be aware of.
