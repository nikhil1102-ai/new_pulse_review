# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

"Groww Weekly Review Pulse" — a LangGraph pipeline that scrapes Google Play reviews for the Groww app over an 8-week window, clusters them into themes, generates a ≤250-word Markdown report plus a fee explainer via an LLM, validates that every quote is real, renders a detailed PDF, and delivers it to Google Drive + Gmail. Requirements live in `docs/problemStatement.md`; the design (phase-by-phase) in `docs/architecture.md`.

## Commands

```bash
pip install -r requirements.txt

# Run the pipeline (auto 8-week window ending today)
python -m src.main --backfill

# Explicit window / product / verbosity
python -m src.main --product Groww --week-start 2026-07-13 --week-end 2026-09-07 --log-level DEBUG

# Two-stage run: generate, review, then deliver separately
python -m src.main --stage generate --backfill   # builds everything, sends nothing
python -m src.main --stage deliver               # ships the generated run

# Tests (no pytest config file; plain invocation)
pytest
pytest tests/test_clustering.py
pytest tests/test_clustering.py::TestClusterNode::test_kmeans_fallback
pytest --cov=src
```

`main.py` at the repo root is the Railway/Docker entry point and just calls `src.main.main()`. `Procfile` and `Dockerfile` both run the `--backfill` form.

Config comes from `.env` (see `.env.example`); `src/config.py` loads it via `python-dotenv` and is the single place env vars are read.

## Architecture

### Graph

`src/graph.py` exposes `build_graph(include_delivery=True)` and a module-level `app` (the full pipeline). Passing `include_delivery=False` ends the graph at `generate_pdf`, so nothing can reach an external service — that is what the CI generate stage runs. Nodes are a linear chain with one conditional edge:

```
fetch_reviews → clean → deduplicate → pii_scrub → chunk → batch_prepare
→ embed → cache_vectors → cluster → label_themes → identify_fee_issue
→ generate_report → generate_fee_explainer → validate
→ [generate_report | generate_pdf] → deliver → END
```

`should_retry_report` loops back to `generate_report` while `validation_passed` is false and `retry_count < MAX_VALIDATION_RETRIES` (default 2); after that it falls through to `generate_pdf` and delivers anyway with warnings.

Adding a node means: a function in `src/nodes/`, a field in `PipelineState`, plus `add_node`/`add_edge` in `graph.py`.

### State

`src/state.py` defines `PipelineState`, a `TypedDict(total=False)`. Every node takes the full state and returns a **partial dict** of only the keys it produces — never the whole state. Node docstrings document their input/output keys; keep that convention.

### Node conventions

- One file per node in `src/nodes/`, named after the node, exporting a function of the same name.
- Config is imported as module-level constants (`from src.config import JINA_BATCH_SIZE`), **not** read from `os.environ` inside nodes. Consequence for tests: patch the constant on the *node* module (`patch("src.nodes.deliver.MCP_SERVER_URL", "")`), not on `src.config`.
- Heavy/fragile imports are deferred inside functions (`umap` in `cluster._reduce_dimensions`, `nltk` + `punkt_tab` download in `chunk`) so the module imports cheaply and tests can run without them.
- Nodes degrade gracefully rather than raising: missing `MCP_SERVER_URL` skips delivery, HDBSCAN under-clustering falls back to relaxed params then K-Means, JINA failures retry with exponential backoff.

### External services

- **Reviews**: `google-play-scraper` (public scraping, no credentials). Swapping to the official Play Developer API should only touch `src/nodes/fetch_reviews.py`.
- **Embeddings**: JINA `jina-embeddings-v3`, 1024-dim, batches of 256 — `src/nodes/embed.py`.
- **LLM**: `langchain_openai.ChatOpenAI` pointed at an OpenAI-*compatible* base URL. The default `OPENAI_API_BASE` is Groq (`https://api.groq.com/openai/v1`) with model `openai/gpt-oss-120b`, so `OPENAI_API_KEY` normally holds a Groq key. Every LLM node (`label_themes`, `identify_fee_issue`, `generate_report`, `generate_fee_explainer`) builds its client the same way: set `base_url` only when `OPENAI_API_BASE` is non-empty.
- **Delivery**: `src/mcp_client.py` — despite the name and the `MCP_SERVER_URL` variable, this is plain JSON-over-HTTPS (`httpx`) to a Railway-hosted FastAPI server: `POST /upload_to_drive` (PDF, base64 in `content_b64`) and `POST /create_email_draft`. There is no MCP/SSE transport; `MCP_SERVER_URL` is a bare base URL with no `/sse` suffix. `append_to_doc()` and `call_mcp_tool()` remain but are no longer on the delivery path. The server holds the Google OAuth credentials, so this repo needs no Google secrets.

### The detailed PDF

The email carries only the ≤250-word summary; the PDF is the deep artifact. Three modules, deliberately layered so each is testable alone:

- `src/analytics.py` — pure functions over state, no rendering deps. Produces the funnel, rating mix, weekly trend, per-theme severity/engagement, version breakdown, top-upvoted complaints and fee stats. Everything is derived from state already in the graph, so it costs no extra API calls.
- `src/charts.py` — matplotlib (forced to the `Agg` backend at import, for headless Docker) → PNG bytes. Every renderer returns `None` instead of raising; `render_all` tolerates individual failures.
- `src/pdf_report.py` — ReportLab layout. Base-14 fonts only, so no font files are needed in the image. **Those fonts are WinAnsi/Latin-1 encoded, so glyphs outside that range (★, emoji) render as boxes** — the charts carry symbols instead.

`src/nodes/generate_pdf.py` ties them together, writes `data/reports/<product>_weekly_pulse_<week_start>.pdf`, and imports the two heavy libraries lazily. It sits between `validate` and `deliver` so the document exists on disk before any external call — which is also where a future approval gate would pause. A failure returns `pdf_path: None` and the pipeline still emails the summary. Set `PDF_ENABLED=false` to skip it.

### Idempotency and audit

Re-running the same product/week must not duplicate reports or emails. Three independent mechanisms:

- `cache_vectors` merges new embeddings into `data/embeddings/embeddings_<week_start>.npy` + metadata JSON; `batch_prepare` skips chunk IDs already cached, so re-runs don't re-pay for JINA calls.
- `validate` writes `data/reports/report_<sha256(product_weekstart_weekend)[:12]>.md` and checks for it first.
- `deliver` writes `data/reports/audit_<product>_<week_start>.json` and short-circuits the whole delivery if an existing audit record has `email_sent: true`. The audit record carries product, window, review/theme counts, report hash, `pdf_path`, `pdf_url`, and delivery status.

### Data layout

`data/raw/`, `data/clean/`, `data/embeddings/`, `data/reports/` — paths from `src/config.py`. Files are named by `week_start`. Note `data/` is **not** gitignored; sample raw and clean JSON for 2026-07-16 are committed.

## Tests

`tests/` mirrors phases rather than modules (e.g. `test_clustering.py` covers both `cluster.py` and `label_themes.py`, `test_embedding.py` covers `chunk`/`batch_prepare`/`embed`/`cache_vectors`). There is no `conftest.py`; each file defines its own `_make_review` / `_make_chunk` helpers. All network and LLM calls are mocked with `unittest.mock.patch`, and filesystem-touching nodes are pointed at `tmp_path`.

## Gotchas

- PII scrubbing runs **after** deduplication on purpose — redaction tokens would break MinHash similarity.
- `clean` lowercases into `cleaned_text` but preserves `original_text`; quotes in the report and the fuzzy validation in `validate` (rapidfuzz, threshold 85) both work against `original_text`.
- Clustering operates on chunks; `cluster` reassembles chunks back to parent `review_id`s before emitting clusters.
- Reviews are data, not instructions — never let review text drive prompt behaviour when editing LLM nodes.
- Phase numbers in module docstrings drifted from `docs/architecture.md` (e.g. `validate` says "Phase 6", the doc says Phase 5). Trust the graph order, not the numbers.
- `docs/architecture.md` still describes delivery as a Google **Doc** append. That is stale — delivery now uploads a PDF to Drive. The code is the source of truth.
### Pulse (web UI)

`src/web/` is a FastAPI app for triggering, reviewing and approving runs. It is the long-running process now — `Procfile` serves `src.web.app:app`, not the batch pipeline.

- `store.py` — SQLite run history. The generated report is stored **in the row**, not left on disk, so an approval survives a restart. Only the PDF stays on the filesystem, which is why `REPORTS_DIR` needs a persistent volume.
- `runner.py` — background threads. `_run_pipeline` calls `build_graph(include_delivery=False)` and streams node updates into the run's `step` field for the progress line. `_run_delivery` calls the `deliver` node directly.
- `app.py` — JSON API plus the single page at `/`.
- `auth.py` — HTTP Basic on every route except `/api/health`. It **fails closed**: with `PULSE_PASSWORD` unset, protected routes return 503 rather than serving openly, since a public deployment silently allowing anyone in is far worse than refusing. Credentials compare via `secrets.compare_digest`, and FastAPI's `/docs`, `/redoc` and `/openapi.json` are disabled so they cannot describe the API unauthenticated.
- `static/index.html` — vanilla JS, no build step. Polls every 2.5s while a run is live, 15s otherwise.

Lifecycle: `queued → running → awaiting_approval → delivering → delivered`, with `rejected` and `failed` as the other terminals. Only `queued`/`running`/`delivering` count as active, so a run parked at `awaiting_approval` does not block the next trigger. `reset_stuck_runs()` fails interrupted runs on startup but deliberately spares ones awaiting approval — their results are in the database and still deliverable.

Concurrent triggers return 409: runs share paths under `REPORTS_DIR` and would collide.

`PULSE_DB_PATH` overrides the database location (default `data/pulse.db`). On Railway it must point at a mounted volume or history is lost on every deploy.

### Two-stage runs and the approval gate

`src/main.py` supports `--stage generate|deliver|all` (default `all`):

- **generate** — builds the report and PDF, writes `data/reports/pending_delivery.json`, and stops. It runs `build_graph(include_delivery=False)`, so the `deliver` node is absent from the graph entirely: there is no code path to an external service, not merely a skipped step.
- **deliver** — reads that handoff file and delivers the already-generated run.

This exists so a human can review real output before anything is sent. Pulse drives the gate through these same two phases; the CLI flags are the terminal equivalent.

The handoff stores `review_count`/`theme_count` rather than the lists themselves — `deliver` only takes `len()` of them — keeping it a few KB instead of megabytes. `_run_deliver` rebuilds placeholder lists of the right length. If `deliver` ever reads more than the length of those fields, this breaks.

`--stage deliver` deliberately imports only `src.nodes.deliver`, never the graph, so a delivery-only process needs just `httpx` and `python-dotenv` rather than the clustering and PDF stack.

**`AUTO_APPROVE` must stay `true` on the Railway MCP server.** Its `request_approval()` calls `input()`, so in a headless environment `false` means *reject everything* (EOFError → `return False`), not "ask someone". The real gate belongs upstream of the server.

- Embedding cache keys on `week_start`, so it only helps re-runs of the same window, not week-over-week — consecutive windows overlap by ~7 weeks but re-embed in full.
