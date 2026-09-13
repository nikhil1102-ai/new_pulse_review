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

`src/graph.py` builds the whole `StateGraph` and exports the compiled `app`. Nodes are a linear chain with one conditional edge:

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
- There is no human approval gate before delivery. `app.compile()` takes no checkpointer and no `interrupt_before`, so a run writes to Drive and Gmail unattended. Gmail only ever receives a *draft*, so nothing is auto-sent.
