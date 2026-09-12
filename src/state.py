"""
Pipeline state definition for the LangGraph orchestration.

This TypedDict defines the complete state schema passed between all nodes
in the Groww Weekly Review Pulse pipeline.
"""

from typing import TypedDict, Optional


class PipelineState(TypedDict, total=False):
    """
    Typed state shared across all LangGraph nodes.

    Fields are grouped by the pipeline phase that produces them.
    Using ``total=False`` so nodes can return partial state updates.
    """

    # ── Inputs ────────────────────────────────────────────────
    product: str                        # e.g. "Groww"
    week_start: str                     # ISO date  (YYYY-MM-DD)
    week_end: str                       # ISO date  (YYYY-MM-DD)

    # ── Phase 1 — Ingestion ──────────────────────────────────
    raw_reviews: list[dict]             # fetched reviews from Google Play API
    cleaned_reviews: list[dict]         # after clean + dedup + PII scrub

    # ── Phase 2 — Embedding ──────────────────────────────────
    chunks: list[dict]                  # chunked review segments
    embeddings: list[list[float]]       # JINA embedding vectors
    embedding_ids: list[str]            # aligned chunk_ids

    # ── Phase 3 — Clustering ─────────────────────────────────
    clusters: list[dict]                # {label, review_ids, centroid_quotes}
    noise_reviews: list[dict]           # outlier reviews from HDBSCAN

    # ── Phase 4 — Report ─────────────────────────────────────
    report_markdown: str                # generated Markdown report

    # ── Phase 5 — Validation ─────────────────────────────────
    validation_passed: bool
    validation_errors: list[str]
    retry_count: int                    # track report-generation retries

    # ── Phase 3b — Fee/Charge Pain Point Detection ──────────
    fee_pain_point: str                 # identified fee/charge confusion
    fee_related_reviews: list[dict]     # reviews related to fee issues

    # ── Phase 4b — Fee Explainer ─────────────────────────────
    fee_explainer: str                  # generated fee explanation (≤6 bullets)

    # ── Phase 6 — Delivery ───────────────────────────────────
    doc_url: Optional[str]              # Google Doc link
    email_sent: bool
    audit_record: dict                  # full run metadata
