# Observability: shipping logs to ELK

A reusable pattern for shipping `actual-cat`'s logs (and the surrounding stack's)
into an existing Elastic stack and building audit dashboards in Kibana. This is
optional — the JSONL audit log is useful standalone (`tail -f`, `jq`, `grep`). The
example below assumes an 8.x ELK stack already running on a host referred to here as
`elk-host`, with a Logstash beats input.

The headline artifact is the **LLM/tool-call audit trail**: every categorization
decision and (if you run an MCP server in front of Actual) every budget-mutating
tool call, searchable and chartable.

## Outcome

Filebeat on the worker host ships four log streams — nginx access, MCP tool-calls,
the `actual-cat` JSONL audit, and the worker's journald stderr — to a central
Logstash, which parses each by `log_type` and forwards to a shared
`logs-%{+YYYY.MM.dd}` index. Two Kibana dashboards read it:

- **MCP Tool-Call Audit** — tool-call volume, calls by tool, a budget-write
  ("mutating") signal, and a recent-calls table.
- **Categorization Monitoring** — decision volume, by pipeline/action, confidence
  distribution, top categories, failures, and a recent-decisions table.

Both dashboards aggregate via **runtime keyword fields** on a dedicated data view
(see Step 4 for why).

## Bank-sync audit events

`bank_sync.py` writes four event types into the same `actual-cat.jsonl` stream as
every other pipeline decision, so they need no new Filebeat input or Logstash
filter — they arrive through the existing `actualcat-audit` path and land under
the `actualcat.*` namespace.

| Event | Fields | Meaning |
|---|---|---|
| `bank_sync_ok` | `accounts_synced`, `imported_count`, `per_account` | A completed sync pass (possibly zero accounts if none were due) |
| `bank_sync_skipped` | `reason`: `disabled` \| `interval` \| `daily_cap` | The gate blocked the run before touching any account |
| `bank_sync_account_failed` | `account`, `error_type`, `status`, `reason` | One account's `run_bank_sync()` raised `ActualBankSyncError`; other accounts still ran |
| `bank_sync_failed` | `error` | The whole stage raised outside the per-account loop (e.g. a network-level failure) |

**`bank_sync_account_failed` is the one worth a Kibana alert.** Expired bank
credentials are the normal failure shape here, they're persistent (every
subsequent run fails the same way until someone re-links the account), and
without an alert the only symptom is "transactions stopped appearing" — usually
noticed a week later. Once this stream is flowing, add a Kibana alert rule (or a
panel on the Categorization Monitoring dashboard) on
`actualcat.event:bank_sync_account_failed` count > 0 over a rolling window.

## Architecture

```
worker host                                      ELK host
┌───────────────────────────────┐
│ nginx :443  → access.log ──┐   │
│ MCP server  → json-file ───┤   │   Filebeat        Logstash         Elasticsearch
│ actual-cat  → *.jsonl ─────┼──▶│  (4 inputs) ─────▶ beats ─────────▶  logs-* index
│ actual-cat  → journald ────┘   │   tag log_type     branch on            │
└───────────────────────────────┘                    [fields][log_type]    ▼
                                                                          Kibana
                                                              MCP audit + categorization dashboards
```

## Key decisions

| Decision | Choice | Rationale |
|---|---|---|
| Sources in scope | nginx access + MCP tool-calls + actual-cat audit + worker stderr | Full audit coverage |
| Ingest path | Filebeat → central Logstash → ES | Centralizes parsing in Logstash |
| Logstash input | Reuse an existing `beats {}` input | No new input needed if one already exists |
| Logstash config | Add a filter file + (if needed) an output branch | Single concatenated `conf.d/*.conf` pipeline; slot in by filename order |
| Index naming | Shared `logs-%{+YYYY.MM.dd}` | Reuse the general index for a single-user stack |
| actual-cat parsing | `json` filter (no grok) | The audit log is already clean JSON Lines |
| nginx geoip | Skip | LAN/VPN-only traffic |

## Step 0 — Inventory the existing stack

Before adding anything, confirm what's already there on `elk-host`:

- ES / Logstash version (this pattern targets an 8.x stack — ingest pipelines,
  `grok`, NDJSON dashboard export all available).
- Whether a beats input already exists (e.g. in
  `/etc/logstash/conf.d/01-filebeat-input.conf`) — reuse it; add no new input.
- The pipeline layout — typically a single `main` pipeline over
  `/etc/logstash/conf.d/*.conf`, loaded in filename order.
- Existing indices and the output routing in `50-output.conf` (or equivalent), so
  you know where un-tagged events land and whether you need an output change.

In a typical single-user setup, finance events carry no special tag and fall to the
general `logs-*` index, so **no output change is needed** — you only add a filter.

## Step 1 — MCP tool-call logging (the long pole — do first)

If you run an MCP server in front of Actual (so an LLM client can read/write the
budget), its tool-call log is the centerpiece of the audit and the one unknown.
Inspect the container first:

```bash
docker ps                                       # find the MCP container name
docker logs --tail=200 <mcp-container>          # is anything per-tool-call logged?
docker inspect <mcp-container> --format '{{json .HostConfig.LogConfig}}'   # logging driver
docker inspect <mcp-container> --format '{{json .Config.Env}}'             # log-level env vars?
```

A representative MCP server (e.g. a community `actual-mcp-server` image) logs with
the `json-file` driver to
`/var/lib/docker/containers/<cid>/<cid>-json.log`, in winston text format
(`YYYY-MM-DD HH:mm:ss.SSS <level>: <message>`). The two records that matter:

- `debug: [TOOL CALL] <tool> args={...json...}` — **tool name + arguments**
- `info: [TOOL RESULT] <tool>: {...json...}` — tool name + full result payload

Ship the container log via a Filebeat `filestream` + `container` parser, filter to
the MCP container, and in Logstash keep only `[TOOL CALL]` / `[TOOL RESULT]` lines.

### Caveats that shape the config

1. **`[TOOL CALL]` args are `debug`-level.** If debug is disabled, arguments are
   lost (tool names survive via the `info`-level `[TOOL RESULT]`). Pin
   `LOG_LEVEL=debug` (or the image's equivalent) in the MCP container env.
2. **`[TOOL RESULT]` payloads are large** (full budget blobs). Index tool name +
   status only; drop/truncate the result body to avoid ES bloat.
3. **No user attribution** with a single bearer token. Best granularity is
   per-tool-call, not per-user — so replace any "top callers" panel with
   **mutating-tool-call highlighting** (budget writes are the real risk signal).
4. **Secrets:** the log lines themselves contain no secrets. Note that
   `docker inspect .Config.Env` *does* expose passwords — that's an inspect-only
   exposure, not a log one.

## Step 2 — Install & configure Filebeat on the worker host

Filebeat is not in Debian's repos — add Elastic's 8.x APT repo (match your stack):

```bash
wget -qO - https://artifacts.elastic.co/GPG-KEY-elasticsearch \
  | sudo gpg --dearmor -o /usr/share/keyrings/elastic-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/elastic-keyring.gpg] https://artifacts.elastic.co/packages/8.x/apt stable main" \
  | sudo tee /etc/apt/sources.list.d/elastic-8.x.list
sudo apt-get update
sudo apt-get install -y filebeat
filebeat version          # confirm it matches your stack
# do NOT enable the service yet — configure + test first (below)
```

`/etc/filebeat/filebeat.yml` — Logstash output, four inputs, each tagged with a
`log_type` so Logstash can branch:

```yaml
filebeat.inputs:
  - type: filestream
    id: nginx-access
    paths: ["/var/log/nginx/access.log"]
    fields: { log_type: nginx-access }
    fields_under_root: false

  - type: filestream
    id: actualcat-audit
    paths: ["/opt/actual-cat/logs/actual-cat.jsonl"]   # absolute path
    fields: { log_type: actualcat-audit }
    # ship raw; Logstash decodes JSON (keeps parsing centralized)

  # MCP tool-call logs — Docker json-file driver (Step 1). Read all container logs,
  # keep only the MCP container via add_docker_metadata + drop_event.
  - type: filestream
    id: mcp-toolcall
    paths: ["/var/lib/docker/containers/*/*-json.log"]
    parsers:
      - container: {}        # unwraps the json-file envelope -> message + stream + ts
    fields: { log_type: mcp-toolcall }
    processors:
      - add_docker_metadata: ~
      - drop_event:
          when:
            not:
              equals: { container.name: "<mcp-container>" }

  # actual-cat worker stderr (WARNING/FATAL from __main__.py) for failure visibility
  - type: journald
    id: actualcat-stderr
    include_matches:
      - _SYSTEMD_UNIT=actual-cat.service
    fields: { log_type: actualcat-stderr }

output.logstash:
  hosts: ["elk-host:5044"]   # match your beats input port; ssl => false for plaintext LAN

# no output.elasticsearch — Logstash is the only hop
```

Validate locally before relying on it:

```bash
sudo filebeat test config
sudo filebeat test output           # must reach Logstash
sudo systemctl enable --now filebeat
# confirm the Filebeat user can read the audit file:
sudo test -r /opt/actual-cat/logs/actual-cat.jsonl && echo readable
```

## Step 3 — Logstash filters

The pipeline is a single concatenated `conf.d/*.conf`. **Add filters only** — if an
existing output already ships beats events to `logs-*`, do not add a new `output{}`
(that would duplicate the ES output). Drop a new
`/etc/logstash/conf.d/20-finance-filters.conf` (number `20` puts it after inputs,
before output):

```ruby
filter {
  if [fields][log_type] == "nginx-access" {
    grok { match => { "message" => "%{HTTPD_COMBINEDLOG}" } }
    date { match => ["timestamp", "dd/MMM/yyyy:HH:mm:ss Z"] }
    # skip geoip (LAN-only); useragent {} optional
  }
  else if [fields][log_type] == "actualcat-audit" {
    # decode into its own namespace — the audit's top-level "event":"decision"
    # would otherwise collide with ECS's event{} object and get rejected by ES.
    json { source => "message" target => "actualcat" }
    # -> actualcat.event, actualcat.pipeline, actualcat.mode, actualcat.action,
    #    actualcat.run_id, actualcat.llm_response.confidence, etc.
  }
  else if [fields][log_type] == "mcp-toolcall" {
    # winston colorizes the log level — strip ANSI escapes BEFORE grok
    mutate { gsub => ["message", "\x1B\[[0-9;]*m", ""] }
    # keep only the two audit records; drop the rest of the chatty winston output
    if "[TOOL CALL]" not in [message] and "[TOOL RESULT]" not in [message] {
      drop { }
    }
    grok {
      match => {
        "message" => [
          "%{TIMESTAMP_ISO8601:mcp_ts} %{LOGLEVEL:mcp_level}: \[TOOL CALL\] %{DATA:mcp_tool} args=%{GREEDYDATA:mcp_args}",
          "%{TIMESTAMP_ISO8601:mcp_ts} %{LOGLEVEL:mcp_level}: \[TOOL RESULT\] %{DATA:mcp_tool}: %{GREEDYDATA:mcp_result_raw}"
        ]
      }
    }
    # parse args (small) into a structured field; leave the huge result body out
    if [mcp_args] {
      mutate { strip => ["mcp_args"] }                  # drop trailing newline
      json { source => "mcp_args" target => "mcp_args_json" }
    }
    if [mcp_result_raw] {
      mutate { add_field => { "mcp_result_status" => "ok" } }   # presence of result = success
      truncate { fields => ["mcp_result_raw"] length_bytes => 512 }  # cap, don't index full blob
    }
    # flag budget-mutating tools (writes are the prompt-injection risk signal)
    if [mcp_tool] =~ /(?i)(create|update|delete|import|set|add|remove)/ {
      mutate { add_field => { "mcp_mutating" => "true" } }
    }
  }
}
```

**ANSI gotcha:** the MCP container's winston output colorizes the log level
(`[32minfo[39m` in the stored `message`), so the `mutate gsub` must run *before* the
grok — without it `%{LOGLEVEL}` fails to match. Inspect the raw ES `_source`, not the
prettified `docker logs` view, to catch this.

Reload Logstash per your convention (SIGHUP / `config.reload.automatic` / restart).

## Step 4 — Kibana: data view + dashboards

### The mapping gotcha that drives the design

A general `logs-*` index template often maps string fields as **`text` only — no
`.keyword` subfields** (confirm via `_field_caps`). Text fields aren't aggregatable,
so terms aggregations ("calls by tool", "by category") can't run on them directly,
and adding keyword multifields would require a reindex of the shared index.

**Solution:** a dedicated data view (e.g. `Finance Audit (logs-*)`, title `logs-*`)
carrying **runtime keyword fields** computed from `_source` — e.g. `mcp_tool_kw`,
`mcp_mutating_kw`, `actualcat_pipeline_kw`, `actualcat_action_kw`,
`actualcat_confidence_kw`, `actualcat_category_kw`, `actualcat_account_kw`. Runtime
fields evaluate at query time, so they work on existing and future data with no
reindex, and the dedicated view leaves any shared data view untouched.

### Dashboards

Author with legacy aggregation-based visualizations (more import-robust than Lens)
and a self-contained data view. Import via **Stack Management → Saved Objects →
Import**.

1. **MCP Tool-Call Audit** — tool-call volume over time, calls by tool
   (`mcp_tool_kw`), a mutating-call metric (`mcp_mutating:true`, the budget-write
   risk signal — replaces a "top callers" panel, meaningless with a single bearer
   token), and a recent-calls table (tool / args / result status / mutating).
2. **Categorization Monitoring** — decision volume, by pipeline
   (`actualcat_pipeline_kw`), by action, confidence distribution, top categories, a
   failures metric (`actualcat.event:failure`), and a recent-decisions table.

### Caveats

- `mcp_mutating` / `mcp_mutating_kw` only exist once a budget-write tool has
  actually run; until then the mutating panel reads 0 (read-only tools don't set it).
- Any decisions shipped as raw `message` *before* the filter existed lack the
  `actualcat.*` namespace and won't appear in the categorization aggregations. New
  decisions populate it forward.

## Risks / considerations

- **MCP debug level must stay on.** `[TOOL CALL]` args are `debug`-level; pin
  `LOG_LEVEL=debug` in the MCP container env so an image/default change can't
  silently drop argument granularity. (Tool names survive at `info` via
  `[TOOL RESULT]`.)
- **Financial data lands in `logs-*` in cleartext (accept consciously).** Budget
  data stays E2EE end-to-end, but the MCP process is the one place plaintext lives;
  the actual-cat audit (payees, amounts) and MCP arguments become searchable in the
  shared index. Acceptable for a single-user Logstash; revisit (dedicated index,
  access restriction, descriptor redaction) if the stack ever gains additional users.
- **Log rotation for `actual-cat.jsonl`.** It appends unbounded. The worker is a
  short-lived hourly oneshot that re-opens the file per write, so a plain logrotate
  between runs is safe (no held FD; `copytruncate` not required):

  ```
  /opt/actual-cat/logs/actual-cat.jsonl {
      weekly
      rotate 8
      compress
      missingok
      notifempty
  }
  ```

## Verification

1. `filebeat test config` → `Config OK`; `filebeat test output` → reaches the server
   (a TLS warning is expected for plaintext beats).
2. Generate one event per source and confirm in ES (`logs-*`):
   - nginx → parsed to ECS fields (`url`, `http.response.status_code`).
   - MCP → `mcp_tool` parsed from a real tool call.
   - actual-cat → `actualcat.event` namespace decoded after a worker run.
3. grok/json failures on the MCP stream: **0**
   (`tags:_grokparsefailure AND fields.log_type:mcp-toolcall` → 0 hits).
4. Both dashboards render the live data.

## Common issues (and fixes)

1. **Filebeat not in Debian repos** — `apt-get install filebeat` fails (`Unable to
   locate package`). Add the Elastic 8.x APT repo (Step 2).
2. **Placeholder pasted as a literal** — a `<<port>>`-style token copied into
   `filebeat.yml` fails `filebeat test output` with `unknown port`. Replace
   placeholders with real values before testing.
3. **Winston ANSI color codes** — break `%{LOGLEVEL}` grok; strip ANSI with a
   `mutate gsub` before grok (Step 3).
4. **ECS `event` field collision** — the audit JSON's top-level `"event":"decision"`
   clashes with Filebeat/ECS's `event{}` object (object-vs-string mapping
   rejection). Decode the audit into its own namespace:
   `json { source => "message" target => "actualcat" }`.
5. **Text-only string mapping** — `logs-*` maps strings as `text` with no `.keyword`,
   so terms aggregations can't run. Use runtime keyword fields on a dedicated data
   view (Step 4) — no reindex, works on existing data.
