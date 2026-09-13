"""
LangGraph pipeline definition for the Groww Weekly Review Pulse.

Assembles all node functions into a StateGraph with typed state,
wires sequential edges and the conditional validation → retry loop,
and exports the compiled graph as ``app``.

Pipeline flow::

    fetch_reviews → clean → deduplicate → pii_scrub
    → chunk → batch_prepare → embed → cache_vectors
    → cluster → label_themes → identify_fee_issue
    → generate_report → generate_fee_explainer
    → validate → [retry or generate_pdf] → deliver → END

When ``include_delivery`` is False the graph ends at ``generate_pdf``. That
is what the approval-gated CI workflow runs: everything is produced and can
be reviewed, but nothing is sent until a human approves the second stage.
"""

import logging
from langgraph.graph import StateGraph, END

from src.state import PipelineState
from src.config import MAX_VALIDATION_RETRIES

# ── Node imports ──────────────────────────────────────────────
from src.nodes.fetch_reviews import fetch_reviews
from src.nodes.clean import clean
from src.nodes.deduplicate import deduplicate
from src.nodes.pii_scrub import pii_scrub
from src.nodes.chunk import chunk
from src.nodes.batch_prepare import batch_prepare
from src.nodes.embed import embed
from src.nodes.cache_vectors import cache_vectors
from src.nodes.cluster import cluster
from src.nodes.label_themes import label_themes
from src.nodes.identify_fee_issue import identify_fee_issue
from src.nodes.generate_report import generate_report
from src.nodes.generate_fee_explainer import generate_fee_explainer
from src.nodes.validate import validate
from src.nodes.generate_pdf import generate_pdf
from src.nodes.deliver import deliver

logger = logging.getLogger(__name__)

# ── Conditional edge: validation → retry or continue ─────────
def should_retry_report(state: PipelineState) -> str:
    """Route back to report generation if validation failed and retries remain."""
    if not state.get("validation_passed", False):
        retry_count = state.get("retry_count", 0)
        if retry_count < MAX_VALIDATION_RETRIES:
            logger.warning(
                "Validation failed (attempt %d/%d). Retrying report generation.",
                retry_count + 1,
                MAX_VALIDATION_RETRIES,
            )
            return "generate_report"
        logger.error(
            "Validation failed after %d retries. Continuing with warnings.",
            MAX_VALIDATION_RETRIES,
        )
    return "generate_pdf"


# ── Build the graph ───────────────────────────────────────────


def build_graph(include_delivery: bool = True):
    """Assemble and compile the pipeline graph.

    Args:
        include_delivery: When True the graph ends with the ``deliver`` node.
            When False it ends at ``generate_pdf``, producing every artifact
            without making any external call — the generate stage of the
            approval-gated workflow.

    Returns:
        The compiled LangGraph application.
    """
    workflow = StateGraph(PipelineState)

    # Register nodes
    workflow.add_node("fetch_reviews", fetch_reviews)
    workflow.add_node("clean", clean)
    workflow.add_node("deduplicate", deduplicate)
    workflow.add_node("pii_scrub", pii_scrub)
    workflow.add_node("chunk", chunk)
    workflow.add_node("batch_prepare", batch_prepare)
    workflow.add_node("embed", embed)
    workflow.add_node("cache_vectors", cache_vectors)
    workflow.add_node("cluster", cluster)
    workflow.add_node("label_themes", label_themes)
    workflow.add_node("identify_fee_issue", identify_fee_issue)
    workflow.add_node("generate_report", generate_report)
    workflow.add_node("generate_fee_explainer", generate_fee_explainer)
    workflow.add_node("validate", validate)
    workflow.add_node("generate_pdf", generate_pdf)

    # ── Sequential edges ─────────────────────────────────────
    workflow.set_entry_point("fetch_reviews")
    workflow.add_edge("fetch_reviews", "clean")
    workflow.add_edge("clean", "deduplicate")
    workflow.add_edge("deduplicate", "pii_scrub")
    workflow.add_edge("pii_scrub", "chunk")
    workflow.add_edge("chunk", "batch_prepare")
    workflow.add_edge("batch_prepare", "embed")
    workflow.add_edge("embed", "cache_vectors")
    workflow.add_edge("cache_vectors", "cluster")
    workflow.add_edge("cluster", "label_themes")
    workflow.add_edge("label_themes", "identify_fee_issue")
    workflow.add_edge("identify_fee_issue", "generate_report")
    workflow.add_edge("generate_report", "generate_fee_explainer")
    workflow.add_edge("generate_fee_explainer", "validate")

    workflow.add_conditional_edges("validate", should_retry_report)

    if include_delivery:
        # PDF generation sits between validation and delivery, so the
        # document exists on disk before any external call is made.
        workflow.add_node("deliver", deliver)
        workflow.add_edge("generate_pdf", "deliver")
        workflow.add_edge("deliver", END)
    else:
        workflow.add_edge("generate_pdf", END)

    return workflow.compile()


# Default application: the full pipeline, delivery included.
app = build_graph(include_delivery=True)
