"""
Phase 1b — Clean raw reviews.

Applies five sequential transformations:
  1. Unicode normalisation (NFKC)
  2. Strip HTML / URLs
  3. Language filter (English only via langdetect)
  4. Minimum length gate (default 10 chars)
  5. Lowercasing (preserve original_text for quoting)
"""

import logging
import re
import unicodedata

from langdetect import detect, LangDetectException

from src.config import MIN_REVIEW_LENGTH
from src.state import PipelineState

logger = logging.getLogger(__name__)

# ── Regex patterns (compiled once) ───────────────────────────
_RE_HTML_TAGS = re.compile(r"<[^>]+>")
_RE_HTTP_URLS = re.compile(r"https?://\S+")
_RE_WWW_URLS = re.compile(r"www\.\S+")
_RE_NON_PRINTABLE = re.compile(r"[^\x20-\x7E\n\t]")


def _normalise_unicode(text: str) -> str:
    """NFKC normalisation + strip non-printable characters."""
    text = unicodedata.normalize("NFKC", text)
    text = _RE_NON_PRINTABLE.sub("", text)
    return text.strip()


def _strip_html_urls(text: str) -> str:
    """Remove HTML tags, HTTP(S) URLs, and www-prefixed URLs."""
    text = _RE_HTML_TAGS.sub("", text)
    text = _RE_HTTP_URLS.sub("", text)
    text = _RE_WWW_URLS.sub("", text)
    return text.strip()


def _is_english(text: str) -> bool:
    """
    Detect whether the text is English using langdetect.

    For short bilingual text (>50 chars), err on the side of inclusion.
    Returns False on detection failure (empty / ambiguous text).
    """
    if not text.strip():
        return False
    try:
        lang = detect(text)
        return lang == "en"
    except Exception:
        logger.warning("Language detection failed for text: %.50s…", text)
        return False


def _clean_single_review(raw_review: dict) -> dict | None:
    """
    Apply the cleaning pipeline to a single raw review.

    Returns a cleaned review dict, or None if the review should be discarded.
    """
    text = raw_review.get("text", "")
    if not text:
        return None

    # Step 1: Unicode normalisation
    text = _normalise_unicode(text)

    # Step 2: Strip HTML / URLs
    text = _strip_html_urls(text)

    # Step 3: Language filter
    if not _is_english(text):
        return None

    # Step 4: Minimum length gate
    if len(text.strip()) < MIN_REVIEW_LENGTH:
        return None

    # Step 5: Lowercasing — preserve original before lowering
    original_text = text
    cleaned_text = text.lower()

    return {
        "review_id": raw_review.get("review_id", ""),
        "original_text": original_text,
        "cleaned_text": cleaned_text,
        "rating": raw_review.get("star_rating", 0),
        "date": raw_review.get("date", ""),
        "author_name": raw_review.get("author_name", ""),
        "thumbs_up_count": raw_review.get("thumbs_up_count", 0),
        "app_version": raw_review.get("app_version", ""),
    }


def clean(state: PipelineState) -> dict:
    """
    LangGraph node: clean raw reviews.

    Input:  state["raw_reviews"]  — list[dict] from fetch_reviews
    Output: {"cleaned_reviews": list[dict]}  — partially cleaned reviews

    Applies: Unicode normalisation → HTML/URL stripping → language filter →
             minimum length gate → lowercasing.
    """
    raw_reviews = state.get("raw_reviews", [])
    total_input = len(raw_reviews)

    cleaned: list[dict] = []
    discarded_non_english = 0
    discarded_short = 0
    discarded_empty = 0

    for raw_review in raw_reviews:
        text = raw_review.get("text", "")

        # Track empty before any processing
        if not text or not text.strip():
            discarded_empty += 1
            continue

        # Apply Unicode normalisation + HTML/URL stripping first
        processed_text = _normalise_unicode(text)
        processed_text = _strip_html_urls(processed_text)

        # Check if text became empty after stripping
        if not processed_text.strip():
            discarded_empty += 1
            continue

        # Language filter
        if not _is_english(processed_text):
            discarded_non_english += 1
            continue

        # Minimum length gate
        if len(processed_text.strip()) < MIN_REVIEW_LENGTH:
            discarded_short += 1
            continue

        # Lowercasing — preserve original
        original_text = processed_text
        cleaned_text = processed_text.lower()

        cleaned.append({
            "review_id": raw_review.get("review_id", ""),
            "original_text": original_text,
            "cleaned_text": cleaned_text,
            "rating": raw_review.get("star_rating", 0),
            "date": raw_review.get("date", ""),
            "author_name": raw_review.get("author_name", ""),
            "thumbs_up_count": raw_review.get("thumbs_up_count", 0),
            "app_version": raw_review.get("app_version", ""),
        })

    # ── Logging ──────────────────────────────────────────────
    logger.info(
        "Cleaning complete: %d input → %d output "
        "(discarded: %d non-English, %d too short, %d empty)",
        total_input,
        len(cleaned),
        discarded_non_english,
        discarded_short,
        discarded_empty,
    )

    return {"cleaned_reviews": cleaned}
