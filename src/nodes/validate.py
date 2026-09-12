"""
Phase 6 — Validate report quotes, structure, and theme coverage.

Checks:
1. Quote authenticity — fuzzy-match blockquotes against source reviews
2. Theme coverage    — every large cluster (≥ 30 reviews) appears in the report
3. Formatting        — required Markdown sections are present
4. Idempotency       — check for existing report with same composite key

If quote validation fails > 50%, sets ``validation_passed = False`` to
trigger the retry edge back to ``generate_report`` (max 2 retries).
"""

import hashlib
import json
import logging
import os
import re
from datetime import datetime, timezone

from rapidfuzz import fuzz

from src.config import (
    FUZZY_MATCH_THRESHOLD,
    MAX_REPORT_WORDS,
    REPORTS_DIR,
)
from src.state import PipelineState

logger = logging.getLogger(__name__)

# Minimum cluster size to require theme coverage in the report
_COVERAGE_MIN_REVIEWS = 30

# Required Markdown sections (case-insensitive search)
_REQUIRED_SECTIONS = [
    "Summary of Top Themes",
    "Supporting User Quotes",
    "Key Observation",
    "Action Ideas",
]


# ── Helpers ──────────────────────────────────────────────────


def _extract_blockquotes(markdown: str) -> list[str]:
    """Extract all blockquote lines from a Markdown string.

    Matches lines starting with ``>`` (with optional leading whitespace).
    Strips the ``>`` prefix and leading/trailing whitespace.
    """
    quotes: list[str] = []
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith(">"):
            text = stripped.lstrip(">").strip()
            # Skip empty blockquotes and blockquote-style admonitions
            if text and not text.startswith("[!"):
                quotes.append(text)
    return quotes


def _check_quote_authenticity(
    quotes: list[str],
    cleaned_reviews: list[dict],
    threshold: int = FUZZY_MATCH_THRESHOLD,
) -> tuple[list[str], int, int]:
    """Fuzzy-match each extracted quote against source reviews.

    Returns (errors, matched_count, total_quotes).
    """
    if not quotes:
        return [], 0, 0

    source_texts = [
        r.get("original_text", r.get("cleaned_text", ""))
        for r in cleaned_reviews
        if r.get("original_text") or r.get("cleaned_text")
    ]

    errors: list[str] = []
    matched = 0

    for quote in quotes:
        best_score = 0
        for source in source_texts:
            score = fuzz.partial_ratio(quote, source)
            if score > best_score:
                best_score = score
            if score >= threshold:
                break  # found a good match, stop early

        if best_score >= threshold:
            matched += 1
        else:
            errors.append(
                f"Quote not found in source data (best match {best_score}%): "
                f"\"{quote[:80]}{'…' if len(quote) > 80 else ''}\""
            )

    return errors, matched, len(quotes)


def _check_theme_coverage(
    report_markdown: str,
    clusters: list[dict],
    min_reviews: int = _COVERAGE_MIN_REVIEWS,
) -> list[str]:
    """Verify every cluster with ≥ min_reviews reviews appears in the report."""
    errors: list[str] = []
    report_lower = report_markdown.lower()

    for idx, cl in enumerate(clusters):
        review_ids = cl.get("review_ids", [])
        label = cl.get("label", "")

        if len(review_ids) >= min_reviews and label:
            # Check if the theme label (or a close variant) appears in the report
            if label.lower() not in report_lower:
                errors.append(
                    f"Theme \"{label}\" ({len(review_ids)} reviews) "
                    f"not found in report."
                )

    return errors


def _check_formatting(report_markdown: str) -> list[str]:
    """Verify required Markdown sections are present."""
    errors: list[str] = []
    report_lower = report_markdown.lower()

    for section in _REQUIRED_SECTIONS:
        # Look for the section as a heading (## Section Name) case-insensitively
        pattern = rf"#+\s*{re.escape(section.lower())}"
        if not re.search(pattern, report_lower):
            errors.append(f"Missing required section: \"{section}\"")

    return errors


def _check_idempotency(
    product: str, week_start: str, week_end: str
) -> str | None:
    """Check if a report for this composite key already exists.

    Returns the existing report path if found, None otherwise.
    """
    composite = f"{product}_{week_start}_{week_end}"
    report_hash = hashlib.sha256(composite.encode()).hexdigest()[:12]
    report_path = os.path.join(REPORTS_DIR, f"report_{report_hash}.md")

    if os.path.exists(report_path):
        return report_path
    return None


def _save_report(
    report_markdown: str, product: str, week_start: str, week_end: str
) -> str:
    """Save the validated report to disk."""
    os.makedirs(REPORTS_DIR, exist_ok=True)

    composite = f"{product}_{week_start}_{week_end}"
    report_hash = hashlib.sha256(composite.encode()).hexdigest()[:12]
    report_path = os.path.join(REPORTS_DIR, f"report_{report_hash}.md")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_markdown)

    logger.info("Report saved to %s", report_path)
    return report_path


# ── Main node ────────────────────────────────────────────────


def validate(state: PipelineState) -> dict:
    """LangGraph node: validate the generated report.

    **Input** (from state):
        - ``report_markdown``  — generated Markdown report
        - ``cleaned_reviews``  — source reviews for quote matching
        - ``clusters``         — labelled clusters for coverage check
        - ``product``          — product name
        - ``week_start``       — ISO date
        - ``week_end``         — ISO date
        - ``retry_count``      — current retry counter

    **Output**::

        {
            "validation_passed": bool,
            "validation_errors": list[str],
            "retry_count": int
        }
    """
    report_markdown = state.get("report_markdown", "")
    cleaned_reviews = state.get("cleaned_reviews", [])
    clusters = state.get("clusters", [])
    product = state.get("product", "Groww")
    week_start = state.get("week_start", "")
    week_end = state.get("week_end", "")
    retry_count = state.get("retry_count", 0)

    all_errors: list[str] = []

    # ── Guard: empty report ──────────────────────────────────
    if not report_markdown.strip():
        logger.error("Report is empty — validation failed.")
        return {
            "validation_passed": False,
            "validation_errors": ["Report is empty."],
            "retry_count": retry_count + 1,
        }

    # ── 1. Idempotency check ────────────────────────────────
    existing = _check_idempotency(product, week_start, week_end)
    if existing:
        logger.info("Existing report found at %s — skipping validation.", existing)
        return {
            "validation_passed": True,
            "validation_errors": [],
            "retry_count": retry_count,
        }

    # ── 2. Quote authenticity ────────────────────────────────
    quotes = _extract_blockquotes(report_markdown)
    quote_errors, matched, total = _check_quote_authenticity(
        quotes, cleaned_reviews
    )
    all_errors.extend(quote_errors)

    quote_pass_rate = matched / max(total, 1)
    logger.info(
        "Quote validation: %d/%d matched (%.0f%%).",
        matched, total, quote_pass_rate * 100,
    )

    # If > 50% of quotes fail, flag for retry
    quote_validation_failed = total > 0 and quote_pass_rate < 0.50

    # ── 3. Theme coverage ────────────────────────────────────
    coverage_errors = _check_theme_coverage(report_markdown, clusters)
    all_errors.extend(coverage_errors)

    # ── 4. Formatting ────────────────────────────────────────
    format_errors = _check_formatting(report_markdown)
    all_errors.extend(format_errors)

    # ── Decision ─────────────────────────────────────────────
    # ── 5. Word count ────────────────────────────────────────
    word_count = len(report_markdown.split())
    if word_count > MAX_REPORT_WORDS:
        word_count_error = (
            f"Report exceeds {MAX_REPORT_WORDS}-word limit "
            f"({word_count} words)."
        )
        all_errors.append(word_count_error)
        logger.warning(word_count_error)

    validation_passed = not quote_validation_failed and len(format_errors) == 0

    if validation_passed:
        # Save the validated report
        _save_report(report_markdown, product, week_start, week_end)
        logger.info("Validation passed with %d warnings.", len(all_errors))
    else:
        logger.warning(
            "Validation FAILED (%d errors). retry_count → %d",
            len(all_errors),
            retry_count + 1,
        )

    return {
        "validation_passed": validation_passed,
        "validation_errors": all_errors,
        "retry_count": retry_count + 1 if not validation_passed else retry_count,
    }
