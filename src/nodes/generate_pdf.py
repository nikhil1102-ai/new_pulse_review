"""
Phase 6b — Build the detailed PDF report.

Runs after validation so the document only ever embeds a report that passed
(or exhausted) the quote-authenticity checks. Computes the analytics bundle,
renders the charts, lays out the PDF, and writes it to ``data/reports/``.

Delivery is deliberately *not* done here — :mod:`src.nodes.deliver` uploads
the file. Keeping generation separate means the PDF exists on disk before any
external call, which is what a future approval gate would pause on.
"""

import logging
import os

from src.analytics import compute_insights
from src.config import PDF_ENABLED, REPORTS_DIR
from src.state import PipelineState

logger = logging.getLogger(__name__)


def _pdf_filename(product: str, week_start: str) -> str:
    """Filesystem-safe PDF name for this product and window."""
    slug = "".join(
        c if c.isalnum() or c in "-_" else "_" for c in str(product).lower()
    ).strip("_") or "report"
    return f"{slug}_weekly_pulse_{week_start}.pdf"


def generate_pdf(state: PipelineState) -> dict:
    """LangGraph node: render the detailed PDF report.

    **Input** (from state):
        - ``report_markdown``     — validated ≤250-word summary
        - ``fee_explainer``       — fee explainer Markdown, if generated
        - ``cleaned_reviews``     — corpus for the analytics
        - ``clusters``            — labelled themes
        - ``raw_reviews``, ``chunks``, ``noise_reviews`` — funnel counts
        - ``fee_related_reviews``, ``fee_pain_point`` — fee analysis
        - ``validation_passed``, ``validation_errors``, ``retry_count``

    **Output**::

        {
            "insights": dict,
            "pdf_path": str | None,
        }

    A failure here is logged and returns ``pdf_path: None`` — the pipeline
    still delivers the Markdown summary by email rather than aborting.
    """
    product = state.get("product", "Groww")
    week_start = state.get("week_start", "unknown")

    insights = compute_insights(state)

    if not PDF_ENABLED:
        logger.info("PDF_ENABLED is false — skipping PDF generation.")
        return {"insights": insights, "pdf_path": None}

    try:
        # Imported lazily: reportlab and matplotlib are heavy, and this keeps
        # the rest of the graph importable without them.
        from src.charts import render_all
        from src.pdf_report import build_pdf

        charts = render_all(insights)
        pdf_bytes = build_pdf(
            insights=insights,
            report_markdown=state.get("report_markdown", ""),
            fee_explainer=state.get("fee_explainer", ""),
            charts=charts,
        )
    except Exception:
        logger.exception("PDF generation failed — continuing without a PDF.")
        return {"insights": insights, "pdf_path": None}

    os.makedirs(REPORTS_DIR, exist_ok=True)
    pdf_path = os.path.join(REPORTS_DIR, _pdf_filename(product, week_start))

    try:
        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)
    except OSError:
        logger.exception("Could not write PDF to %s", pdf_path)
        return {"insights": insights, "pdf_path": None}

    logger.info("PDF report written to %s (%.1f KB)", pdf_path, len(pdf_bytes) / 1024)

    return {"insights": insights, "pdf_path": pdf_path}
