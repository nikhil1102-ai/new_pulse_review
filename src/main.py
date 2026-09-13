"""
CLI entry point for the Groww Weekly Review Pulse pipeline.

Usage:
    python -m src.main --product groww --week-start 2026-07-13 --week-end 2026-09-07
    python -m src.main --product groww --backfill         # auto-compute 8-week window

Stages (``--stage``):
    all        run everything, delivering immediately (default)
    generate   produce the report and PDF, then stop before any external call
    deliver    deliver a previously generated run

The split exists for the approval-gated CI workflow: ``generate`` runs
unattended and writes a handoff file, a human reviews the result, and only
then does ``deliver`` run. Nothing reaches Google Drive or Gmail in between.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone

from src.config import REPORTS_DIR, REVIEW_WINDOW_WEEKS

# Handoff file written by the generate stage and consumed by the deliver
# stage. It carries only what ``deliver`` reads, so it stays a few KB rather
# than embedding thousands of reviews.
PENDING_FILENAME = "pending_delivery.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Groww Weekly Review Pulse — AI-powered weekly review agent",
    )
    parser.add_argument(
        "--product",
        type=str,
        default="Groww",
        help="Product name (default: Groww)",
    )
    parser.add_argument(
        "--week-start",
        type=str,
        default=None,
        help="Start of the review window (YYYY-MM-DD). Defaults to 8 weeks ago.",
    )
    parser.add_argument(
        "--week-end",
        type=str,
        default=None,
        help="End of the review window (YYYY-MM-DD). Defaults to today.",
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Automatically compute an 8-week window ending today.",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", "generate", "deliver"],
        help=(
            "Which part of the pipeline to run. 'generate' stops before any "
            "external call; 'deliver' ships a previously generated run."
        ),
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    return parser.parse_args()


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _pending_path() -> str:
    """Location of the generate -> deliver handoff file."""
    return os.path.join(REPORTS_DIR, PENDING_FILENAME)


def _save_pending(state: dict, logger: logging.Logger) -> str:
    """Persist the fields ``deliver`` needs from a completed generate run.

    ``deliver`` only ever takes ``len()`` of ``cleaned_reviews`` and
    ``clusters``, so counts are stored instead of the lists themselves and
    rebuilt as placeholders on resume. That keeps the handoff small enough to
    travel as a CI artifact.
    """
    payload = {
        "product": state.get("product", ""),
        "week_start": state.get("week_start", ""),
        "week_end": state.get("week_end", ""),
        "report_markdown": state.get("report_markdown", ""),
        "fee_explainer": state.get("fee_explainer", ""),
        "fee_pain_point": state.get("fee_pain_point", ""),
        "pdf_path": state.get("pdf_path"),
        "validation_passed": state.get("validation_passed", False),
        "validation_errors": state.get("validation_errors", []),
        "review_count": len(state.get("cleaned_reviews", [])),
        "theme_count": len(state.get("clusters", [])),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    os.makedirs(REPORTS_DIR, exist_ok=True)
    path = _pending_path()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logger.info("Pending delivery written to %s", path)
    return path


def _run_deliver(logger: logging.Logger) -> None:
    """Deliver a run previously produced by the generate stage.

    Imports only the delivery node, so this stage does not need the heavy
    clustering and PDF dependencies.
    """
    path = _pending_path()
    if not os.path.exists(path):
        logger.error(
            "No pending delivery found at %s. Run --stage generate first.", path
        )
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        pending = json.load(f)

    logger.info(
        "Delivering %s (%s to %s), generated %s",
        pending.get("product"),
        pending.get("week_start"),
        pending.get("week_end"),
        pending.get("generated_at", "unknown"),
    )

    state = dict(pending)
    # deliver() only measures these, so placeholders of the right length are
    # sufficient and keep the handoff file small.
    state["cleaned_reviews"] = [{}] * int(pending.get("review_count", 0))
    state["clusters"] = [{}] * int(pending.get("theme_count", 0))

    from src.nodes.deliver import deliver  # deferred: keeps imports light

    try:
        result = deliver(state)
    except Exception:
        logger.exception("Delivery failed")
        sys.exit(1)

    if result.get("pdf_url"):
        logger.info("PDF URL   : %s", result["pdf_url"])
    logger.info("Email sent: %s", result.get("email_sent", False))
    logger.info("Delivery complete.")


def main() -> None:
    args = _parse_args()
    _setup_logging(args.log_level)
    logger = logging.getLogger("main")

    # ── Deliver-only: no date computation or graph needed ────
    if args.stage == "deliver":
        _run_deliver(logger)
        return

    # ── Compute date window ──────────────────────────────────
    today = datetime.now(timezone.utc).date()

    if args.week_end:
        week_end = datetime.strptime(args.week_end, "%Y-%m-%d").date()
    else:
        week_end = today

    if args.week_start:
        week_start = datetime.strptime(args.week_start, "%Y-%m-%d").date()
    elif args.backfill:
        week_start = week_end - timedelta(weeks=REVIEW_WINDOW_WEEKS)
    else:
        week_start = week_end - timedelta(weeks=REVIEW_WINDOW_WEEKS)

    logger.info("Product   : %s", args.product)
    logger.info("Window    : %s → %s", week_start.isoformat(), week_end.isoformat())

    # ── Build initial state ──────────────────────────────────
    initial_state = {
        "product": args.product,
        "week_start": week_start.isoformat(),
        "week_end": week_end.isoformat(),
    }

    # ── Run the LangGraph pipeline ───────────────────────────
    from src.graph import build_graph  # deferred import to avoid circular deps

    include_delivery = args.stage == "all"
    if not include_delivery:
        logger.info("Stage 'generate': stopping before delivery.")

    graph = build_graph(include_delivery=include_delivery)

    logger.info("Starting pipeline …")
    try:
        final_state = graph.invoke(initial_state)
    except Exception:
        logger.exception("Pipeline failed")
        sys.exit(1)

    # ── Summary ──────────────────────────────────────────────
    logger.info("Pipeline complete.")
    if final_state.get("pdf_path"):
        logger.info("Detailed PDF: %s", final_state["pdf_path"])
    if final_state.get("pdf_url"):
        logger.info("PDF URL   : %s", final_state["pdf_url"])
    if final_state.get("email_sent"):
        logger.info("Email sent successfully.")
    if not final_state.get("validation_passed"):
        logger.warning("Report generated with validation warnings.")

    if not include_delivery:
        _save_pending(final_state, logger)
        logger.info("Nothing delivered yet - awaiting approval.")


if __name__ == "__main__":
    main()
