# Deployment Guide

End-to-end guide for deploying `actual-cat` to a production host — the
categorization/transfer worker on a timer, plus the optional receipt-splitting
feature. For design rationale see [architecture.md](architecture.md); for shipping
logs to ELK see [observability.md](observability.md).

Paths below assume the worker is installed at `/opt/actual-cat` and runs as a
dedicated `actual-cat` system user. Adjust to taste. A common pattern is to
co-locate the worker on the same host as actual-server.

## 1. Worker deploy

```bash
sudo mkdir -p /opt/actual-cat
sudo git clone https://github.com/<your-org>/actual-cat.git /opt/actual-cat
sudo useradd --system --no-create-home --shell /usr/sbin/nologin actual-cat
sudo chown -R actual-cat:actual-cat /opt/actual-cat

cd /opt/actual-cat
sudo -u actual-cat python3 -m venv venv
sudo -u actual-cat venv/bin/pip install -e . -q

sudo -u actual-cat cp config.example.toml config.toml
sudo -u actual-cat nano config.toml    # set base_url, file (budget name), endpoint, absolute log path
sudo -u actual-cat nano .env           # ACTUAL_PASSWORD + ACTUAL_ENCRYPTION_PASSWORD
sudo chmod 600 .env
sudo -u actual-cat mkdir -p logs
```

The budget `file` must match the budget's API name exactly (including spaces) —
verify with `actual.list_user_files()` if a run can't find it.

### systemd service + timer

`/etc/systemd/system/actual-cat.service`:

```ini
[Unit]
Description=Actual Budget AI Categorization Worker
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
User=actual-cat
WorkingDirectory=/opt/actual-cat
EnvironmentFile=/opt/actual-cat/.env
Environment="REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt"
Environment="SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt"
ExecStart=/opt/actual-cat/venv/bin/python -m actual_cat
StandardOutput=journal
StandardError=journal
TimeoutStartSec=600
```

> The `REQUESTS_CA_BUNDLE` / `SSL_CERT_FILE` lines are only needed when the Actual
> server presents a certificate from a private CA. For a public HTTPS endpoint,
> leave `paths.ca_bundle` unset and drop both `Environment=` lines.

`/etc/systemd/system/actual-cat.timer`:

```ini
[Unit]
Description=Run actual-cat hourly

[Timer]
OnCalendar=hourly
Persistent=true
RandomizedDelaySec=120

[Install]
WantedBy=timers.target
```

Do a manual dry run **before** enabling the timer, then enable it:

```bash
sudo systemctl daemon-reload

# Manual dry run first
sudo -u actual-cat /opt/actual-cat/venv/bin/python -m actual_cat
sudo -u actual-cat tail /opt/actual-cat/logs/actual-cat.jsonl

# Enable the hourly timer
sudo systemctl enable --now actual-cat.timer
sudo systemctl list-timers actual-cat.timer
```

## 2. Suggest → apply validation

Both pipelines start in **suggest mode**. The timer runs hourly, writing `#ai:`
tags to transaction notes without touching the category field. Monitor agreement:

```bash
# Count suggestions made
sudo -u actual-cat grep '"action": "suggested"' /opt/actual-cat/logs/actual-cat.jsonl | wc -l

# Inspect failures
sudo -u actual-cat grep '"event": "failure"' /opt/actual-cat/logs/actual-cat.jsonl | python3 -m json.tool
```

Advancing to apply mode needs **no redeploy** — just edit `config.toml`:

```toml
[categorization]
mode = "apply"        # was "suggest"
```

Transfer apply mode advances separately and later: a mis-paired transfer corrupts
spending data on both sides, so only flip `transfers.mode = "apply"` after
categorization apply mode has been stable for a month. See the staged
[verification gates](architecture.md#verification-gates-post-deployment).

### A note on bank import structure

Real bank imports typically split a friendly merchant name from a raw descriptor.
For example, `payee.name` may hold the institution's cleaned merchant name while
`notes` holds the raw bank descriptor with a processor prefix and address
(`WHOLEFDS #1234 ... ANYTOWN ST USA`). The worker's prompt passes both, since the
raw descriptor often carries the most identifying signal. Verify the exact field
mapping against your own bank's import format before deploying.

## 3. Receipt splitting (optional)

Adds three components, all running as the `actual-cat` user:

| Component | Type | Purpose |
|---|---|---|
| `actual-cat-receiver` | systemd service (always-on) | HTTP endpoint for iOS Shortcut POSTs |
| IMAP poll | hourly worker (existing timer) | Ingest receipt emails |
| Receipt pipeline | hourly worker (existing timer) | OCR → match → suggest/apply splits |

The receiver is the only new always-on surface. Run it loopback-bound, behind a
reverse proxy on your existing Actual vhost, with a bearer token in `.env`.

### Step 0 — Confirm vision capability

Before deploying, verify the model handles image input:

```bash
python3 - <<'EOF'
import base64, json, urllib.request

img = open("/tmp/test-receipt.jpg", "rb").read()
b64 = base64.b64encode(img).decode()

payload = json.dumps({
    "model": "your-vision-model",
    "messages": [{
        "role": "user",
        "content": [
            {"type": "text", "text": "What merchant is on this receipt? Reply with JSON {\"merchant\": \"...\"}"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        ]
    }],
    "response_format": {"type": "json_object"}
}).encode()

req = urllib.request.Request("http://llm-host:8900/v1/chat/completions",
    data=payload, headers={"Content-Type": "application/json"})
resp = json.loads(urllib.request.urlopen(req).read())
print(resp["choices"][0]["message"]["content"])
EOF
```

**Gate:** the response contains a parseable merchant name. If the model errors on
image input, the OCR layer won't work — switch to a vision-capable model first.

### Step 1 — Config blocks

Add to `/opt/actual-cat/config.toml`:

```toml
[receipts]
enabled = true
store_path = "/opt/actual-cat/receipts"
match_window_days = 3
expiry_days = 30
mode = "suggest"                  # flip to "apply" after validation
apply_confidence_threshold = "high"

[email]
enabled = true
imap_host = "mail.example.com"    # your IMAP server
user = "receipts@example.com"     # the dedicated email alias
mailbox = "INBOX"
# IMAP_PASSWORD in .env

[receiver]
host = "127.0.0.1"
port = 8001
# RECEIPT_RECEIVER_TOKEN in .env
```

Add to `/opt/actual-cat/.env`:

```
IMAP_PASSWORD=<imap-password-here>
RECEIPT_RECEIVER_TOKEN=<generate: openssl rand -hex 32>
```

```bash
chmod 600 /opt/actual-cat/.env
```

### Step 2 — Receiver systemd unit

`/etc/systemd/system/actual-cat-receiver.service`:

```ini
[Unit]
Description=actual-cat receipt HTTP receiver
After=network.target

[Service]
Type=simple
User=actual-cat
WorkingDirectory=/opt/actual-cat
EnvironmentFile=/opt/actual-cat/.env
ExecStart=/opt/actual-cat/venv/bin/python -m actual_cat.receipts.receiver
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
systemctl daemon-reload
systemctl enable --now actual-cat-receiver
systemctl status actual-cat-receiver

# Smoke test (401 = auth broken; 405 = working, GET just isn't defined)
source /opt/actual-cat/.env
curl -s -o /dev/null -w "%{http_code}" \
  -H "Authorization: Bearer $RECEIPT_RECEIVER_TOKEN" \
  http://127.0.0.1:8001/receipts/
```

### Step 3 — Reverse proxy (nginx example)

Add to your Actual server vhost (inside the `server {}` block):

```nginx
location /receipts/ {
    proxy_pass http://127.0.0.1:8001;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    client_max_body_size 25M;
}
```

```bash
nginx -t && systemctl reload nginx

# External smoke test from a LAN/VPN device (expects 405)
source /opt/actual-cat/.env
curl -s -o /dev/null -w "%{http_code}" \
  -H "Authorization: Bearer $RECEIPT_RECEIVER_TOKEN" \
  https://actual.example.com/receipts/
```

### Step 4 — iOS Shortcut

Create a Shortcut that:

1. **Get input** — "Shortcut Input" → Image (from Share Sheet or Camera).
2. **Get contents of URL**
   - URL: `https://actual.example.com/receipts/`
   - Method: POST
   - Headers: `Authorization: Bearer <RECEIPT_RECEIVER_TOKEN>`
   - Request body: Form
     - `image`: Shortcut Input (file, name `receipt.jpg`, type `image/jpeg`)
3. **Show result** — display the response JSON (shows `receipt_id` on success).

Add the shortcut to the Share Sheet so it appears when sharing a photo from the
Camera Roll. **Test:** photograph a receipt, share it, and confirm the response is
`{"receipt_id": "...", "status": "received"}` and that
`/opt/actual-cat/receipts/inbox/` shows two new files (`<id>.jpg` + `<id>.json`).

### Step 5 — Email alias (optional)

Configure an email alias (`receipts@example.com`) to deliver to the IMAP mailbox
the worker polls. Ensure the alias accepts mail from your sending addresses and that
IMAP access is enabled with an app password (set `IMAP_PASSWORD` to the app
password, not the account password). Test:

```bash
cd /opt/actual-cat
source .env
venv/bin/python -c "
from actual_cat.config import load_config
from actual_cat.audit import AuditLogger
from actual_cat.receipts.ingest import poll_email

cfg = load_config()
audit = AuditLogger(cfg.audit_log_path)
n = poll_email(cfg, audit)
print(f'Ingested {n} image(s)')
"
```

### Step 6 — Suggest-mode validation

Run for ~2 weeks in `mode = "suggest"`. Each hourly run polls email → inbox, OCRs
inbox images → pending, matches pending receipts to bank transactions, and tags
`#ai-suggested-split` on the matched transaction (no split rows created). Review by
filtering transactions with `#ai-suggested-split` in the Actual UI: is the split
breakdown correct, and is the matched transaction the right one? When correct ≥90%:

```toml
[receipts]
mode = "apply"
```

In apply mode the worker sets `is_parent = 1`, clears the single category, and
creates categorized child split rows (the parent shows a split icon in the UI).

### Receipt splitting — known limitations

- **Tipped charges never match** — the charged amount differs from the receipt
  total. By design (exact-amount matching keeps split sums trivially correct);
  tipped transactions fall through to the normal categorization pipeline.
- **Ambiguous matches** — if two transactions of the same amount fall within the
  date window, the receipt is held and logged as `receipt_ambiguous`. Resolve by
  lowering `match_window_days` or handling it in the UI.
- **HEIC images** — the receiver accepts HEIC, but the LLM must support it. If OCR
  fails on HEIC, convert to JPEG in the iOS Shortcut (the "Convert Image" action)
  before posting.
- **Off-budget accounts** — never matched (consistent with all other pipelines).

## Troubleshooting

Issues commonly hit during a first deploy, with fixes:

- **Wrong budget name.** The budget's API name may differ from what you expect
  (e.g. a space vs a hyphen). Find the exact name via `actual.list_user_files()`.
- **Missing logs dir.** Create it as the `actual-cat` user: `mkdir -p logs`.
- **Malformed production rule.** A rule with an empty category ID can fail pydantic
  validation in `get_ruleset`; the rule-engine pass is wrapped in try/except (the
  server runs rules on sync regardless). See commit `e41b1e1`.
- **Off-budget "Transfer" hallucination.** Off-budget accounts don't use budget
  categories; they're excluded from `find_uncategorized` so the LLM isn't asked to
  categorize them. See commit `8972b3b`.
- **`No module named 'fastapi'`** when starting the receiver. The venv was created
  before the receipt dependencies were added — pull the latest commit and
  re-install:
  ```bash
  cd /opt/actual-cat
  git pull
  venv/bin/pip install -e .
  sudo systemctl restart actual-cat-receiver
  ```
- **`Actual(cert=...)` expects PEM content, not a path.** Rely on
  `REQUESTS_CA_BUNDLE` / `SSL_CERT_FILE` env vars (set in the service unit) and omit
  `cert`.

## Idempotency / safety notes

- The worker only sees transactions where `category_id is None` — anything you've
  already categorized in the UI is invisible to it.
- `#ai:`, `#ai-assisted`, `#ai-suggested-transfer`, `#ai-receipt-split`, and
  `#ai-suggested-split` markers cause a transaction to be skipped on re-run.
  Re-running is always safe.
- Original notes content is always preserved; tags are prepended.
- The transfer pipeline runs before categorization so transfers aren't
  miscategorized.
