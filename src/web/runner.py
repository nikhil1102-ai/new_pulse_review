"""
Background execution of pipeline runs for Pulse.

A run takes minutes — scraping, embedding thousands of reviews, clustering,
several LLM calls — so it cannot happen inside an HTTP request. The trigger
endpoint starts a worker thread and returns immediately; the UI polls for
status.

Two phases, matching the two-stage CLI:

  start_run()  builds everything and stops at AWAITING_APPROVAL. It invokes
               the graph with delivery excluded, so there is no code path to
               Google at all — not merely a skipped step.
  approve()    delivers a run that a human has reviewed.
"""

from __future__ import annotations

import logging
import threading

from src.web import store

logger = logging.getLogger(__name__)

# Node name -> human-readable progress line for the UI.
_STEP_LABELS = {
    "fetch_reviews": "Fetching reviews from Google Play",
    "clean": "Cleaning review text",
    "deduplicate": "Removing duplicates",
    "pii_scrub": "Scrubbing personal information",
    "chunk": "Chunking reviews for embedding",
    "batch_prepare": "Preparing embedding batches",
    "embed": "Generating embeddings",
    "cache_vectors": "Caching embedding vectors",
    "cluster": "Clustering into themes",
    "label_themes": "Labelling themes",
    "identify_fee_issue": "Identifying fee confusion",
    "generate_report": "Writing the weekly summary",
    "generate_fee_explainer": "Writing the fee explainer",
    "validate": "Validating quotes and structure",
    "generate_pdf": "Building the PDF report",
}


def _summarise_themes(clusters: list[dict], insights: dict | None) -> list[dict]:
    """Build the compact theme list the UI shows on the review screen.

    Prefers the analytics breakdown, which carries severity, and falls back to
    raw clusters when the PDF stage did not run.
    """
    if insights and insights.get("themes"):
        return [
            {
                "label": t.get("label", ""),
                "count": t.get("count", 0),
                "share_pct": t.get("share_pct", 0),
                "avg_rating": t.get("avg_rating", 0),
                "quotes": (t.get("quotes") or [])[:2],
            }
            for t in insights["themes"]
        ]

    return [
        {
            "label": c.get("label") or f"Unnamed theme {i + 1}",
            "count": len(c.get("review_ids", [])),
            "share_pct": 0,
            "avg_rating": 0,
            "quotes": (c.get("centroid_quotes") or [])[:2],
        }
        for i, c in enumerate(clusters or [])
    ]


def _run_pipeline(run_id: str, product: str, week_start: str, week_end: str) -> None:
    """Worker body: generate everything, then park the run for approval."""
    store.update_run(run_id, status=store.RUNNING, step="Starting pipeline")

    try:
        # Deferred: these pull in umap, hdbscan, reportlab and matplotlib.
        from src.graph import build_graph

        graph = build_graph(include_delivery=False)
        initial_state = {
            "product": product,
            "week_start": week_start,
            "week_end": week_end,
        }

        final_state: dict = {}

        # stream() reports each node as it completes, which drives the
        # progress line in the UI.
        for update in graph.stream(initial_state, stream_mode="updates"):
            for node_name, node_output in update.items():
                label = _STEP_LABELS.get(node_name, node_name)
                logger.info("Run %s: %s", run_id, label)
                store.update_run(run_id, step=label)
                if isinstance(node_output, dict):
                    final_state.update(node_output)

        insights = final_state.get("insights") or {}
        clusters = final_state.get("clusters") or []

        store.update_run(
            run_id,
            status=store.AWAITING_APPROVAL,
            step="Ready for review",
            review_count=len(final_state.get("cleaned_reviews", [])),
            theme_count=len(clusters),
            validation_passed=final_state.get("validation_passed", False),
            validation_errors=final_state.get("validation_errors", []),
            themes=_summarise_themes(clusters, insights),
            report_markdown=final_state.get("report_markdown", ""),
            fee_explainer=final_state.get("fee_explainer", ""),
            fee_pain_point=final_state.get("fee_pain_point", ""),
            pdf_path=final_state.get("pdf_path"),
        )
        logger.info("Run %s is awaiting approval.", run_id)

    except Exception as exc:
        logger.exception("Run %s failed", run_id)
        store.update_run(
            run_id, status=store.FAILED, step="Failed", error=f"{type(exc).__name__}: {exc}"
        )


def _run_delivery(run_id: str) -> None:
    """Worker body: deliver an approved run."""
    run = store.get_run(run_id)
    if run is None:
        logger.error("Run %s vanished before delivery.", run_id)
        return

    store.update_run(run_id, status=store.DELIVERING, step="Uploading PDF and drafting email")

    try:
        from src.config import REPORT_RECIPIENTS
        from src.nodes.deliver import deliver

        # deliver() only takes len() of these two, so placeholders sized from
        # the stored counts are sufficient and avoid re-reading the corpus.
        state = {
            "product": run["product"],
            "week_start": run["week_start"],
            "week_end": run["week_end"],
            "report_markdown": run["report_markdown"] or "",
            "fee_explainer": run["fee_explainer"] or "",
            "fee_pain_point": run["fee_pain_point"] or "",
            "pdf_path": run["pdf_path"],
            "validation_passed": run["validation_passed"],
            "validation_errors": run["validation_errors"],
            "cleaned_reviews": [{}] * int(run["review_count"] or 0),
            "clusters": [{}] * int(run["theme_count"] or 0),
        }

        result = deliver(state)

        store.update_run(
            run_id,
            status=store.DELIVERED,
            step="Delivered",
            pdf_url=result.get("pdf_url"),
            email_sent=result.get("email_sent", False),
            recipients=list(REPORT_RECIPIENTS),
        )
        logger.info("Run %s delivered. PDF: %s", run_id, result.get("pdf_url"))

    except Exception as exc:
        logger.exception("Delivery of run %s failed", run_id)
        store.update_run(
            run_id,
            status=store.FAILED,
            step="Delivery failed",
            error=f"{type(exc).__name__}: {exc}",
        )


def _spawn(target, *args) -> None:
    """Run *target* on a daemon thread so it never blocks shutdown."""
    threading.Thread(target=target, args=args, daemon=True).start()


def start_run(product: str, week_start: str, week_end: str) -> str:
    """Queue a pipeline run and start it in the background.

    Returns:
        The new run id.

    Raises:
        RuntimeError: If another run is already occupying the pipeline.
    """
    existing = store.active_run()
    if existing:
        raise RuntimeError(
            f"Run {existing['id']} is already {existing['status']}. "
            "Wait for it to finish before starting another."
        )

    run_id = store.create_run(product, week_start, week_end)
    _spawn(_run_pipeline, run_id, product, week_start, week_end)
    return run_id


def approve(run_id: str) -> None:
    """Approve a reviewed run and deliver it in the background.

    Raises:
        LookupError: If the run id is unknown.
        RuntimeError: If the run is not awaiting approval.
    """
    run = store.get_run(run_id)
    if run is None:
        raise LookupError(f"No run with id {run_id}.")

    if run["status"] != store.AWAITING_APPROVAL:
        raise RuntimeError(
            f"Run {run_id} is '{run['status']}', not awaiting approval."
        )

    from src.web.store import _now

    store.update_run(run_id, approved_at=_now())
    _spawn(_run_delivery, run_id)


def reject(run_id: str) -> None:
    """Reject a reviewed run so it is never delivered.

    Raises:
        LookupError: If the run id is unknown.
        RuntimeError: If the run is not awaiting approval.
    """
    run = store.get_run(run_id)
    if run is None:
        raise LookupError(f"No run with id {run_id}.")

    if run["status"] != store.AWAITING_APPROVAL:
        raise RuntimeError(
            f"Run {run_id} is '{run['status']}', not awaiting approval."
        )

    store.update_run(run_id, status=store.REJECTED, step="Rejected - not delivered")
    logger.info("Run %s rejected; nothing was sent.", run_id)
