"""
Tests for Phase 1 — fetch_reviews node.

Covers: field mapping, date filtering, idempotent caching, pagination,
and retry logic.
"""

import json
import os
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.nodes.fetch_reviews import _map_review, fetch_reviews


# ── Fixtures ─────────────────────────────────────────────────

def _make_raw_review(
    review_id="abc123",
    user_name="TestUser",
    content="Great app for trading!",
    score=5,
    at=None,
    thumbs_up=10,
    app_version="5.6.1",
):
    """Create a raw review dict in google-play-scraper format."""
    return {
        "reviewId": review_id,
        "userName": user_name,
        "content": content,
        "score": score,
        "at": at or datetime(2026, 8, 15, 12, 0, 0),
        "thumbsUpCount": thumbs_up,
        "appVersion": app_version,
    }


# ── _map_review tests ───────────────────────────────────────

class TestMapReview:
    """Tests for the field mapping function."""

    def test_maps_all_fields(self):
        raw = _make_raw_review()
        mapped = _map_review(raw)

        assert mapped["review_id"] == "abc123"
        assert mapped["author_name"] == "TestUser"
        assert mapped["text"] == "Great app for trading!"
        assert mapped["star_rating"] == 5
        assert mapped["date"] == "2026-08-15"
        assert mapped["thumbs_up_count"] == 10
        assert mapped["app_version"] == "5.6.1"

    def test_handles_missing_fields(self):
        raw = {}
        mapped = _map_review(raw)

        assert mapped["review_id"] == ""
        assert mapped["author_name"] == ""
        assert mapped["text"] == ""
        assert mapped["star_rating"] == 0
        assert mapped["date"] == ""
        assert mapped["thumbs_up_count"] == 0
        assert mapped["app_version"] == ""

    def test_handles_string_date(self):
        raw = _make_raw_review(at="2026-08-15")
        mapped = _map_review(raw)
        assert mapped["date"] == "2026-08-15"

    def test_handles_none_date(self):
        raw = _make_raw_review(at=None)
        # at=None triggers default in _make_raw_review, so override directly
        raw["at"] = None
        mapped = _map_review(raw)
        assert mapped["date"] == ""


# ── fetch_reviews node tests ─────────────────────────────────

class TestFetchReviews:
    """Tests for the LangGraph fetch_reviews node."""

    @pytest.fixture
    def state(self):
        return {
            "product": "Groww",
            "week_start": "2026-07-13",
            "week_end": "2026-09-07",
        }

    @pytest.fixture
    def mock_reviews_batch(self):
        """A batch of 3 reviews within the date window."""
        return [
            _make_raw_review(
                review_id=f"rev_{i}",
                content=f"Review number {i}",
                score=i % 5 + 1,
                at=datetime(2026, 8, 15 + i),
            )
            for i in range(3)
        ]

    @patch("src.nodes.fetch_reviews.reviews")
    def test_fetches_and_maps_reviews(self, mock_scraper, state, mock_reviews_batch, tmp_path):
        """Verify reviews are fetched, mapped, and persisted."""
        mock_scraper.return_value = (mock_reviews_batch, None)  # no continuation

        with patch("src.nodes.fetch_reviews.RAW_DATA_DIR", str(tmp_path)):
            result = fetch_reviews(state)

        assert "raw_reviews" in result
        assert len(result["raw_reviews"]) == 3
        assert result["raw_reviews"][0]["review_id"] == "rev_0"
        assert result["raw_reviews"][0]["text"] == "Review number 0"

        # Verify file was persisted
        raw_file = tmp_path / "reviews_2026-07-13.json"
        assert raw_file.exists()
        saved = json.loads(raw_file.read_text(encoding="utf-8"))
        assert len(saved) == 3

    @patch("src.nodes.fetch_reviews.reviews")
    def test_idempotent_cache_hit(self, mock_scraper, state, tmp_path):
        """If raw file already exists, load from disk and skip API call."""
        cached_reviews = [
            {"review_id": "cached_1", "author_name": "User", "text": "Cached",
             "star_rating": 4, "date": "2026-08-01", "thumbs_up_count": 0,
             "app_version": "5.0"}
        ]
        raw_file = tmp_path / "reviews_2026-07-13.json"
        raw_file.write_text(json.dumps(cached_reviews), encoding="utf-8")

        with patch("src.nodes.fetch_reviews.RAW_DATA_DIR", str(tmp_path)):
            result = fetch_reviews(state)

        # Should NOT call the scraper
        mock_scraper.assert_not_called()
        assert len(result["raw_reviews"]) == 1
        assert result["raw_reviews"][0]["review_id"] == "cached_1"

    @patch("src.nodes.fetch_reviews.reviews")
    def test_date_filtering(self, mock_scraper, state, tmp_path):
        """Reviews outside the date window are excluded."""
        in_window = _make_raw_review(
            review_id="in_window",
            at=datetime(2026, 8, 1),
        )
        out_of_window = _make_raw_review(
            review_id="out_of_window",
            at=datetime(2026, 6, 1),  # before week_start
        )
        mock_scraper.return_value = ([in_window, out_of_window], None)

        with patch("src.nodes.fetch_reviews.RAW_DATA_DIR", str(tmp_path)):
            result = fetch_reviews(state)

        # Only the in-window review should be present
        assert len(result["raw_reviews"]) == 1
        assert result["raw_reviews"][0]["review_id"] == "in_window"

    @patch("src.nodes.fetch_reviews.reviews")
    def test_deduplicates_within_fetch(self, mock_scraper, state, tmp_path):
        """Duplicate reviewIds within a batch are deduplicated."""
        dup1 = _make_raw_review(review_id="dup", content="First")
        dup2 = _make_raw_review(review_id="dup", content="Second")
        mock_scraper.return_value = ([dup1, dup2], None)

        with patch("src.nodes.fetch_reviews.RAW_DATA_DIR", str(tmp_path)):
            result = fetch_reviews(state)

        assert len(result["raw_reviews"]) == 1
        assert result["raw_reviews"][0]["text"] == "First"

    @patch("src.nodes.fetch_reviews.reviews")
    def test_pagination(self, mock_scraper, state, tmp_path):
        """Handles multiple pages via continuation_token."""
        batch1 = [_make_raw_review(review_id="page1_rev")]
        batch2 = [_make_raw_review(review_id="page2_rev")]

        mock_scraper.side_effect = [
            (batch1, "token_page2"),  # first call returns continuation
            (batch2, None),           # second call ends pagination
        ]

        with patch("src.nodes.fetch_reviews.RAW_DATA_DIR", str(tmp_path)):
            with patch("src.nodes.fetch_reviews.SLEEP_BETWEEN_BATCHES", 0):
                result = fetch_reviews(state)

        assert len(result["raw_reviews"]) == 2
        ids = {r["review_id"] for r in result["raw_reviews"]}
        assert ids == {"page1_rev", "page2_rev"}

    @patch("src.nodes.fetch_reviews.reviews")
    def test_retry_on_failure(self, mock_scraper, state, tmp_path):
        """Retries on transient errors and succeeds on subsequent attempt."""
        batch = [_make_raw_review(review_id="retry_rev")]
        mock_scraper.side_effect = [
            Exception("Network error"),    # first attempt fails
            (batch, None),                 # second attempt succeeds
        ]

        with patch("src.nodes.fetch_reviews.RAW_DATA_DIR", str(tmp_path)):
            with patch("src.nodes.fetch_reviews.time.sleep"):  # skip actual sleep
                result = fetch_reviews(state)

        assert len(result["raw_reviews"]) == 1
        assert result["raw_reviews"][0]["review_id"] == "retry_rev"

    @patch("src.nodes.fetch_reviews.reviews")
    def test_empty_batch_stops(self, mock_scraper, state, tmp_path):
        """Empty batch from scraper stops the fetch loop."""
        mock_scraper.return_value = ([], None)

        with patch("src.nodes.fetch_reviews.RAW_DATA_DIR", str(tmp_path)):
            result = fetch_reviews(state)

        assert result["raw_reviews"] == []
