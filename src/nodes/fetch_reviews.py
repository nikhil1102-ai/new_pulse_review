"""
Phase 1a — Fetch public Google Play reviews for Groww.

Uses the ``google-play-scraper`` library to scrape publicly available reviews.
No API credentials or Play Console access required.

The official Google Play Developer API (v3) can be swapped in later by
replacing only this module.
"""

import json
import logging
import os
import time
from datetime import datetime

from google_play_scraper import Sort, reviews

from src.config import (
    GOOGLE_PLAY_COUNTRY,
    GOOGLE_PLAY_LANGUAGE,
    GOOGLE_PLAY_PACKAGE_NAME,
    RAW_DATA_DIR,
)
from src.state import PipelineState

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────
BATCH_SIZE = 200          # reviews per scraper call
SLEEP_BETWEEN_BATCHES = 1  # seconds; avoid rate-limiting
MAX_RETRIES = 3


def _map_review(raw: dict) -> dict:
    """Map google-play-scraper fields to the pipeline's canonical schema."""
    review_date = raw.get("at")
    if isinstance(review_date, datetime):
        date_str = review_date.strftime("%Y-%m-%d")
    else:
        date_str = str(review_date) if review_date else ""

    return {
        "review_id": raw.get("reviewId", ""),
        "author_name": raw.get("userName", ""),
        "text": raw.get("content", ""),
        "star_rating": raw.get("score", 0),
        "date": date_str,
        "thumbs_up_count": raw.get("thumbsUpCount", 0),
        "app_version": raw.get("appVersion", ""),
    }


def _fetch_from_play_store(week_start: str, week_end: str) -> list[dict]:
    """
    Fetch reviews from Google Play Store using google-play-scraper.

    Paginates using ``continuation_token`` and filters by date window.
    """
    start_date = datetime.strptime(week_start, "%Y-%m-%d")
    all_reviews: list[dict] = []
    continuation_token = None
    seen_ids: set[str] = set()

    logger.info(
        "Fetching reviews for %s (window: %s → %s)",
        GOOGLE_PLAY_PACKAGE_NAME,
        week_start,
        week_end,
    )

    while True:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                batch, continuation_token = reviews(
                    GOOGLE_PLAY_PACKAGE_NAME,
                    lang=GOOGLE_PLAY_LANGUAGE,
                    country=GOOGLE_PLAY_COUNTRY,
                    sort=Sort.NEWEST,
                    count=BATCH_SIZE,
                    continuation_token=continuation_token,
                )
                break
            except Exception as exc:
                logger.warning(
                    "Fetch attempt %d/%d failed: %s", attempt, MAX_RETRIES, exc
                )
                if attempt == MAX_RETRIES:
                    logger.error("Max retries reached. Stopping fetch.")
                    return all_reviews
                time.sleep(2 ** attempt)

        if not batch:
            logger.info("No more reviews returned. Stopping.")
            break

        # Filter by date window and deduplicate within this fetch
        for raw_review in batch:
            review_date = raw_review.get("at")
            if isinstance(review_date, datetime) and review_date < start_date:
                # Reviews are sorted newest-first; once we pass the window, stop
                logger.info(
                    "Reached review dated %s — outside window. Stopping.",
                    review_date.strftime("%Y-%m-%d"),
                )
                return all_reviews

            rid = raw_review.get("reviewId", "")
            if rid and rid not in seen_ids:
                seen_ids.add(rid)
                all_reviews.append(_map_review(raw_review))

        logger.info("Fetched %d reviews so far …", len(all_reviews))

        if continuation_token is None:
            break

        time.sleep(SLEEP_BETWEEN_BATCHES)

    return all_reviews


def fetch_reviews(state: PipelineState) -> dict:
    """
    LangGraph node: fetch raw reviews from the Google Play Store.

    Input:  state["product"], state["week_start"], state["week_end"]
    Output: {"raw_reviews": list[dict]}

    Side-effect: Persists raw reviews to data/raw/reviews_{week_start}.json
    """
    week_start = state["week_start"]
    week_end = state["week_end"]

    # ── Idempotency: reuse cached raw file if it exists ──────
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    raw_file = os.path.join(RAW_DATA_DIR, f"reviews_{week_start}.json")

    if os.path.exists(raw_file):
        logger.info("Raw reviews already cached at %s — loading from disk.", raw_file)
        with open(raw_file, "r", encoding="utf-8") as f:
            raw_reviews = json.load(f)
        logger.info("Loaded %d cached reviews.", len(raw_reviews))
        return {"raw_reviews": raw_reviews}

    # ── Fetch from Play Store ────────────────────────────────
    raw_reviews = _fetch_from_play_store(week_start, week_end)

    # ── Persist to disk ──────────────────────────────────────
    with open(raw_file, "w", encoding="utf-8") as f:
        json.dump(raw_reviews, f, ensure_ascii=False, indent=2)
    logger.info(
        "Fetched %d reviews. Saved to %s", len(raw_reviews), raw_file
    )

    return {"raw_reviews": raw_reviews}
