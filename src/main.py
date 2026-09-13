"""
CLI entry point for the Groww Weekly Review Pulse pipeline.

Usage:
    python -m src.main --product groww --week-start 2026-07-13 --week-end 2026-09-07
    python -m src.main --product groww --backfill         # auto-compute 8-week window
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta

from src.config import REVIEW_WINDOW_WEEKS


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


def main() -> None:
    args = _parse_args()
    _setup_logging(args.log_level)
    logger = logging.getLogger("main")

    # ── Compute date window ──────────────────────────────────
    today = datetime.utcnow().date()

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
    from src.graph import app  # deferred import to avoid circular deps

    logger.info("Starting pipeline …")
    try:
        final_state = app.invoke(initial_state)
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
        logger.warning("Report was delivered with validation warnings.")


if __name__ == "__main__":
    main()
