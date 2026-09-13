# Groww Weekly Review Pulse

An AI agent that turns public Google Play reviews into a weekly product insight report — clustered themes, verified real quotes, actionable ideas, and a fee explainer for the charge users are most confused about.

Reviews go in. A one-page summary lands in an email draft, backed by a detailed PDF in Google Drive. Nothing is sent until a human reviews it and clicks approve.

---

## What it does

Every run walks the same path:

1. **Ingest** — scrape the last 8 weeks of public Google Play reviews (~5,000 for Groww)
2. **Clean** — normalise, strip HTML and URLs, filter to English, drop short entries
3. **Deduplicate** — exact by review ID, near-duplicates via MinHash LSH at Jaccard 0.90
4. **Scrub** — redact phone numbers, emails, account IDs and names *before* any text reaches a language model
5. **Embed** — chunk and embed with `jina-embeddings-v3` (1024 dimensions)
6. **Cluster** — UMAP dimensionality reduction, then HDBSCAN, falling back to K-Means if too few clusters emerge
7. **Label** — an LLM names each theme from a sample of its reviews
8. **Report** — a ≤250-word structured note: top themes, three real quotes, key observation, three action ideas
9. **Fee explainer** — identifies the most recurring fee confusion and writes a neutral, source-linked explanation support can reuse
10. **Validate** — fuzzy-matches every quoted line back to its source review; retries generation if too many fail
11. **PDF** — a detailed report with charts, the full theme table, severity, version breakdown and methodology
12. **Deliver** — uploads the PDF to Drive and creates a Gmail draft, *after approval*

Reviews are treated strictly as data, never as instructions to the model.

---

## Pulse — the web UI

`src/web/` is a small FastAPI app: the front door for the whole system.

| | |
|---|---|
| **Run now** | Triggers a pipeline run in the background with a live progress line |
| **Review** | Metrics, themes with severity ratings, the rendered report, the fee explainer, and the PDF to download |
| **Approve / Reject** | Approving uploads to Drive and drafts the email. Rejecting sends nothing, permanently |
| **History** | Every past run with its status, counts and Drive link |

The key property: the generate phase calls `build_graph(include_delivery=False)`, so the delivery node is **absent from the graph entirely**. There is no code path to Google before approval — it isn't a step that gets skipped.

Run states: `queued → running → awaiting_approval → delivering → delivered`, with `rejected` and `failed` as the other terminals.

Access is HTTP Basic, and it **fails closed** — with `PULSE_PASSWORD` unset, protected routes return 503 rather than serving openly.

---

## The custom MCP server

Google Docs, Drive and Gmail access lives in a **separate service**, not in this repo:

**[nikhil1102-ai/custom_mcp_server](https://github.com/nikhil1102-ai/custom_mcp_server)**

It is a FastAPI app exposing three MCP-style tool endpoints, each gated behind an operator approval check:

| Endpoint | Purpose |
|---|---|
| `POST /upload_to_drive` | Upload the weekly PDF, return a shareable link |
| `POST /create_email_draft` | Create a Gmail draft (plain text + rich HTML) |
| `POST /append_to_doc` | Append text to a Google Doc *(no longer on the delivery path)* |

### Why it is separate

The server holds the Google OAuth credentials and token. Keeping it apart means **this repository needs no Google secrets at all** — only `MCP_SERVER_URL`. It also has a genuinely different shape: the MCP server is a long-running service that must stay reachable, while the pipeline is bursty work that runs and finishes.

It requests three OAuth scopes, deliberately narrow:

- `documents` — Google Docs read/write
- `gmail.compose` — draft creation only, never send
- `drive.file` — access **only to files the app itself creates**, never your existing Drive

### A note on the name

Despite the name and the `MCP_SERVER_URL` variable, there is no MCP or SSE transport involved. `src/mcp_client.py` speaks plain JSON over HTTPS with `httpx`. The name is historical — an earlier version used the MCP protocol, and the naming stuck through the refactor to REST.

### Delivery flow

```
pipeline ──► POST /upload_to_drive ──► Drive API ──► shareable link
         └─► POST /create_email_draft ──► Gmail API ──► draft (never sent)
```

Binary PDFs travel base64-encoded in `content_b64`. Uploads are given "anyone with the link can view" so recipients can open them without requesting access; `LINK_SHARING_ENABLED = False` in the server's `drive_tool.py` turns that off. `DRIVE_FOLDER_ID` keeps every report in one folder instead of the Drive root.

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env      # then fill it in
```

### Configuration

| Variable | Required | Notes |
|---|---|---|
| `OPENAI_API_KEY` | yes | A **Groq** key — `OPENAI_API_BASE` defaults to Groq's endpoint |
| `JINA_API_KEY` | yes | Embeddings |
| `MCP_SERVER_URL` | for delivery | Base URL of the MCP server, no trailing slash |
| `REPORT_RECIPIENTS` | for delivery | Comma-separated addresses |
| `PULSE_PASSWORD` | for the UI | Unset means every protected route returns 503 |
| `PULSE_USERNAME` | no | Defaults to `admin` |
| `PULSE_DB_PATH` | no | Defaults to `data/pulse.db` |

`src/config.py` is the only place environment variables are read. Every other module imports typed constants from it.

---

## Running

### The UI

```bash
python -m uvicorn src.web.app:app --port 8080
```

Then open <http://127.0.0.1:8080>.

### The CLI

```bash
# Full run, delivering immediately
python -m src.main --backfill

# Explicit window
python -m src.main --product Groww --week-start 2026-07-13 --week-end 2026-09-07

# Two-stage: build everything, review it, deliver separately
python -m src.main --stage generate --backfill
python -m src.main --stage deliver
```

---

## Deployment

Two services, because they have different shapes:

| Service | Deploys from | Role |
|---|---|---|
| **Pulse** | this repo | Long-running web UI |
| **MCP server** | `custom_mcp_server` | Long-running Google integration |

The `Dockerfile` serves Pulse on `$PORT` (default 8080). When generating a Railway domain, the listening port is whatever uvicorn reports in the deploy logs:

```
INFO:     Uvicorn running on http://0.0.0.0:8080
```

**Attach a persistent volume.** Railway wipes the filesystem on every deploy, so without one you lose the run history, the generated PDFs, and the embedding cache. Point `PULSE_DB_PATH` and `DATA_DIR` inside it.

Set `PULSE_PASSWORD` *before* generating the domain — the service is public the moment the domain exists.

---

## Idempotency and audit

Re-running the same product and week must not duplicate reports or emails. Three independent mechanisms:

- **Embedding cache** — `data/embeddings/` is keyed by week, so a re-run of the same window does not re-pay for embeddings
- **Report hash** — `validate` writes `report_<sha256(product_weekstart_weekend)[:12]>.md` and checks for it first
- **Audit log** — `deliver` short-circuits entirely if an existing audit record has `email_sent: true`

Each run records product, window, review and theme counts, report hash, PDF path and URL, delivery status, and validation warnings.

---

## Tests

```bash
pytest                                   # 244 tests
pytest tests/test_clustering.py           # one file
pytest tests/test_web.py::TestAuth        # one class
pytest --cov=src                          # with coverage
```

Tests mirror pipeline phases rather than modules. All network and LLM calls are mocked; nothing in the suite touches Google, Groq or Jina.

---

## Project layout

```
src/
├── config.py          all environment variables, read once
├── state.py           the TypedDict passed between nodes
├── graph.py           build_graph(include_delivery) -> compiled LangGraph
├── main.py            CLI with --stage generate|deliver|all
├── mcp_client.py      REST client for the MCP server
├── analytics.py       metrics derived for the PDF
├── charts.py          matplotlib -> PNG
├── pdf_report.py      ReportLab layout
├── nodes/             one file per pipeline node (16 of them)
└── web/               Pulse: store, runner, app, auth, static UI
docs/
├── architecture.md    the original phase-by-phase design
└── problemStatement.md
```

`docs/architecture.md` predates the PDF and Pulse work and still describes delivery as a Google Doc append. The code is the source of truth; `CLAUDE.md` documents the current architecture.

---

## Stack

Python 3.12 · LangGraph · Groq (`openai/gpt-oss-120b`) · Jina embeddings v3 · UMAP + HDBSCAN · ReportLab · matplotlib · FastAPI · SQLite
