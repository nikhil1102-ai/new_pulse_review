"""
Phase 1d — Scrub PII from reviews.

Redacts personally identifiable information from both ``cleaned_text`` and
``original_text``:
  1. Phone numbers  → [PHONE]
  2. Email addresses → [EMAIL]
  3. Account/Order IDs → [ID]
  4. Names (in-text heuristic) → [NAME]

Applied **after** deduplication to avoid MinHash mismatches from redaction tokens.
"""

import json
import logging
import os
import re

from src.config import CLEAN_DATA_DIR
from src.state import PipelineState

logger = logging.getLogger(__name__)

# ── PII regex patterns (applied in order) ────────────────────

_PII_PATTERNS: list[tuple[str, re.Pattern, str]] = [
    (
        "email",
        re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"),
        "[EMAIL]",
    ),
    (
        "id",
        re.compile(r"[A-Z]{2,}\d{6,}"),
        "[ID]",
    ),
    (
        "name",
        re.compile(
            r"(?i)(?:my name is|i am|i'm)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)"
        ),
        "[NAME]",
    ),
    (
        "phone",
        re.compile(r"\+?\d[\d\s\-]{7,}"),
        "[PHONE]",
    ),
]


def _scrub_text(text: str) -> tuple[str, dict[str, int]]:
    """
    Apply all PII patterns to a text string.

    Returns the scrubbed text and a dict of redaction counts per PII type.
    """
    counts: dict[str, int] = {}
    for pii_type, pattern, replacement in _PII_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            counts[pii_type] = len(matches)
            text = pattern.sub(replacement, text)
    return text, counts


def pii_scrub(state: PipelineState) -> dict:
    """
    LangGraph node: redact PII from review text.

    Input:  state["cleaned_reviews"]  — deduplicated from deduplicate node
    Output: {"cleaned_reviews": list[dict]}  — PII-scrubbed reviews

    Side-effect: Persists final corpus to data/clean/clean_reviews_{week_start}.json

    PII patterns are applied **in order** (phone → email → IDs → names)
    to both ``cleaned_text`` and ``original_text``.
    """
    cleaned_reviews = state.get("cleaned_reviews", [])
    week_start = state.get("week_start", "unknown")

    total_reviews = len(cleaned_reviews)
    total_redactions: dict[str, int] = {}
    reviews_with_pii = 0

    scrubbed_reviews: list[dict] = []

    for review in cleaned_reviews:
        review_copy = dict(review)  # shallow copy to avoid mutating state
        review_had_pii = False

        # Scrub cleaned_text
        scrubbed_cleaned, counts_cleaned = _scrub_text(
            review_copy.get("cleaned_text", "")
        )
        review_copy["cleaned_text"] = scrubbed_cleaned

        # Scrub original_text
        scrubbed_original, counts_original = _scrub_text(
            review_copy.get("original_text", "")
        )
        review_copy["original_text"] = scrubbed_original

        # Aggregate counts (take max of cleaned/original per type, since
        # the same PII appears in both)
        for pii_type in set(list(counts_cleaned.keys()) + list(counts_original.keys())):
            count = max(
                counts_cleaned.get(pii_type, 0),
                counts_original.get(pii_type, 0),
            )
            if count > 0:
                review_had_pii = True
                total_redactions[pii_type] = total_redactions.get(pii_type, 0) + count

        if review_had_pii:
            reviews_with_pii += 1

        scrubbed_reviews.append(review_copy)

    # ── Persist final corpus to disk ─────────────────────────
    os.makedirs(CLEAN_DATA_DIR, exist_ok=True)
    output_file = os.path.join(CLEAN_DATA_DIR, f"clean_reviews_{week_start}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(scrubbed_reviews, f, ensure_ascii=False, indent=2)

    # ── Logging ──────────────────────────────────────────────
    logger.info(
        "PII scrubbing complete: %d reviews processed, %d had PII",
        total_reviews,
        reviews_with_pii,
    )
    for pii_type, count in sorted(total_redactions.items()):
        logger.info("  %s redactions: %d", pii_type, count)
    logger.info("Clean corpus saved to %s", output_file)

    return {"cleaned_reviews": scrubbed_reviews}
