"""
Unit tests for Phase 6 — Validation.

Tests cover:
  - Quote extraction from Markdown blockquotes
  - Quote authenticity checking via fuzzy matching
  - Theme coverage verification
  - Markdown formatting checks
  - Idempotency (existing report detection)
  - Full validate node integration
"""

import json
import os
import pytest
from unittest.mock import patch


def _make_review(review_id, text, rating=4):
    """Helper to create a review dict."""
    return {
        "review_id": review_id,
        "original_text": text,
        "cleaned_text": text.lower(),
        "rating": rating,
        "date": "2026-08-15",
        "author_name": "TestUser",
        "thumbs_up_count": 0,
        "app_version": "5.0.0",
    }


def _make_cluster(label, review_ids):
    """Helper to create a cluster dict."""
    return {
        "label": label,
        "review_ids": review_ids,
        "centroid_quotes": [],
    }


GOOD_REPORT = """\
## Summary of Top Themes

Groww users report crashes and slow withdrawals.

## Supporting User Quotes

> The app crashes every morning when the market opens
> My withdrawal has been stuck for 3 days

## Key Observation

App stability at market open is the most critical recurring issue.

## Action Ideas

1. Fix server load at market open
2. Speed up withdrawal processing
3. Improve support response time
"""

BAD_REPORT_MISSING_SECTIONS = """\
## Summary of Top Themes

Some summary here.

## Supporting User Quotes

> Some quote here
"""

BAD_REPORT_FAKE_QUOTES = """\
## Summary of Top Themes

Summary here.

## Supporting User Quotes

> This is a completely invented quote that does not exist in the data
> Another fabricated quote that nobody wrote

## Action Ideas

1. Do something
"""


class TestExtractBlockquotes:
    """Tests for ``_extract_blockquotes``."""

    def test_extracts_standard_blockquotes(self):
        from src.nodes.validate import _extract_blockquotes

        md = "> Quote one\n> Quote two\nNot a quote"
        quotes = _extract_blockquotes(md)

        assert len(quotes) == 2
        assert "Quote one" in quotes
        assert "Quote two" in quotes

    def test_skips_empty_blockquotes(self):
        from src.nodes.validate import _extract_blockquotes

        md = "> \n> Real quote"
        quotes = _extract_blockquotes(md)

        assert len(quotes) == 1
        assert "Real quote" in quotes

    def test_skips_admonitions(self):
        from src.nodes.validate import _extract_blockquotes

        md = "> [!NOTE]\n> This is a note\n> A real quote"
        quotes = _extract_blockquotes(md)

        # Should skip the [!NOTE] line but keep the others
        assert "[!NOTE]" not in [q for q in quotes]


class TestQuoteAuthenticity:
    """Tests for ``_check_quote_authenticity``."""

    def test_matching_quotes_pass(self):
        from src.nodes.validate import _check_quote_authenticity

        quotes = ["The app crashes every morning"]
        reviews = [_make_review("r1", "The app crashes every morning when the market opens")]

        errors, matched, total = _check_quote_authenticity(quotes, reviews)

        assert matched == 1
        assert total == 1
        assert len(errors) == 0

    def test_non_matching_quotes_fail(self):
        from src.nodes.validate import _check_quote_authenticity

        quotes = ["Completely invented nonsense quote"]
        reviews = [_make_review("r1", "App crashes at market open")]

        errors, matched, total = _check_quote_authenticity(quotes, reviews)

        assert matched == 0
        assert total == 1
        assert len(errors) == 1

    def test_empty_quotes_return_zero(self):
        from src.nodes.validate import _check_quote_authenticity

        errors, matched, total = _check_quote_authenticity([], [])

        assert matched == 0
        assert total == 0
        assert len(errors) == 0

    def test_partial_match_passes(self):
        from src.nodes.validate import _check_quote_authenticity

        # Partial match should pass with partial_ratio
        quotes = ["crashes every morning"]
        reviews = [_make_review("r1", "The app crashes every morning when the market opens")]

        errors, matched, total = _check_quote_authenticity(quotes, reviews)

        assert matched == 1


class TestThemeCoverage:
    """Tests for ``_check_theme_coverage``."""

    def test_covered_theme_passes(self):
        from src.nodes.validate import _check_theme_coverage

        report = "## Top Themes\n| App crashes | 100 |"
        clusters = [_make_cluster("App crashes", [f"r{i}" for i in range(35)])]

        errors = _check_theme_coverage(report, clusters)

        assert len(errors) == 0

    def test_missing_theme_fails(self):
        from src.nodes.validate import _check_theme_coverage

        report = "## Top Themes\n| Some other theme | 100 |"
        clusters = [_make_cluster("App crashes", [f"r{i}" for i in range(35)])]

        errors = _check_theme_coverage(report, clusters)

        assert len(errors) == 1
        assert "App crashes" in errors[0]

    def test_small_cluster_not_required(self):
        from src.nodes.validate import _check_theme_coverage

        report = "## Top Themes\nNothing about the theme"
        clusters = [_make_cluster("Small theme", ["r1", "r2"])]

        errors = _check_theme_coverage(report, clusters)

        # Cluster has < 30 reviews, so it's not required
        assert len(errors) == 0


class TestFormatChecks:
    """Tests for ``_check_formatting``."""

    def test_good_report_passes(self):
        from src.nodes.validate import _check_formatting

        errors = _check_formatting(GOOD_REPORT)

        assert len(errors) == 0

    def test_missing_sections_detected(self):
        from src.nodes.validate import _check_formatting

        errors = _check_formatting(BAD_REPORT_MISSING_SECTIONS)

        # Missing: Key Observation, Action Ideas
        assert len(errors) >= 2
        section_names = " ".join(errors)
        assert "Key Observation" in section_names
        assert "Action Ideas" in section_names


class TestValidateNode:
    """Tests for the full ``validate`` node."""

    def test_good_report_passes_validation(self, tmp_path):
        from src.nodes.validate import validate

        reviews = [
            _make_review("r1", "The app crashes every morning when the market opens"),
            _make_review("r2", "My withdrawal has been stuck for 3 days"),
        ]

        state = {
            "report_markdown": GOOD_REPORT,
            "cleaned_reviews": reviews,
            "clusters": [_make_cluster("App crashes", [f"r{i}" for i in range(5)])],
            "product": "TestProduct",
            "week_start": "2026-08-01",
            "week_end": "2026-08-07",
            "retry_count": 0,
        }

        with patch("src.nodes.validate.REPORTS_DIR", str(tmp_path)):
            result = validate(state)

        assert result["validation_passed"] is True

    def test_empty_report_fails(self):
        from src.nodes.validate import validate

        state = {
            "report_markdown": "",
            "cleaned_reviews": [],
            "clusters": [],
            "product": "Groww",
            "week_start": "2026-08-01",
            "week_end": "2026-08-07",
            "retry_count": 0,
        }

        result = validate(state)

        assert result["validation_passed"] is False
        assert "Report is empty" in result["validation_errors"][0]
        assert result["retry_count"] == 1

    def test_fake_quotes_fail_validation(self, tmp_path):
        from src.nodes.validate import validate

        reviews = [
            _make_review("r1", "App crashes at market open"),
        ]

        state = {
            "report_markdown": BAD_REPORT_FAKE_QUOTES,
            "cleaned_reviews": reviews,
            "clusters": [],
            "product": "TestProduct",
            "week_start": "2026-08-01",
            "week_end": "2026-08-07",
            "retry_count": 0,
        }

        with patch("src.nodes.validate.REPORTS_DIR", str(tmp_path)):
            result = validate(state)

        assert result["validation_passed"] is False
        assert result["retry_count"] == 1

    def test_retry_count_increments_on_failure(self, tmp_path):
        from src.nodes.validate import validate

        state = {
            "report_markdown": "",
            "cleaned_reviews": [],
            "clusters": [],
            "product": "Groww",
            "week_start": "2026-08-01",
            "week_end": "2026-08-07",
            "retry_count": 1,
        }

        result = validate(state)

        assert result["retry_count"] == 2
