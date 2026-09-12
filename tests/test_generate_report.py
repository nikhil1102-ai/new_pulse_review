"""
Unit tests for Phase 5 — Report Generation.

Tests cover:
  - Prompt construction (themes block, system/user messages)
  - OpenAI invocation and response handling
  - Error handling on API failure
  - Empty cluster handling
"""

import pytest
from unittest.mock import patch, MagicMock


def _make_cluster(label, review_ids, quotes=None, rank=1, is_top_3=True):
    """Helper to create a cluster dict."""
    return {
        "label": label,
        "review_ids": review_ids,
        "centroid_quotes": quotes or [],
        "rank": rank,
        "is_top_3": is_top_3,
    }


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


SAMPLE_REPORT = """\
## Summary of Top Themes

Groww users report app crashes during market hours and slow withdrawals.

## Supporting User Quotes

> The app crashes every morning when the market opens
> App freezes during trading hours, very frustrating
> Withdrawal takes too long, money stuck for 3 days

## Key Observation

App stability at market open is the most critical recurring issue.

## Action Ideas

1. Investigate server load during market open (9:15 AM IST)
2. Optimize withdrawal processing pipeline
3. Add real-time status updates for pending withdrawals
"""


class TestBuildThemesBlock:
    """Tests for ``_build_themes_block`` helper."""

    def test_formats_clusters_with_quotes(self):
        from src.nodes.generate_report import _build_themes_block

        clusters = [
            _make_cluster("App crashes", ["r1", "r2"], ["Quote A", "Quote B"], rank=1),
        ]
        result = _build_themes_block(clusters)

        assert "Theme 1: App crashes" in result
        assert "Review count: 2" in result
        assert '"Quote A"' in result
        assert '"Quote B"' in result

    def test_handles_no_quotes(self):
        from src.nodes.generate_report import _build_themes_block

        clusters = [
            _make_cluster("Slow app", ["r1"], [], rank=1),
        ]
        result = _build_themes_block(clusters)

        assert "(none available)" in result

    def test_multiple_clusters(self):
        from src.nodes.generate_report import _build_themes_block

        clusters = [
            _make_cluster("Theme A", ["r1"], rank=1),
            _make_cluster("Theme B", ["r2", "r3"], rank=2),
        ]
        result = _build_themes_block(clusters)

        assert "Theme 1: Theme A" in result
        assert "Theme 2: Theme B" in result
        assert "Review count: 1" in result
        assert "Review count: 2" in result


class TestGenerateReportNode:
    """Tests for ``src.nodes.generate_report.generate_report``."""

    @patch("src.nodes.generate_report.ChatOpenAI")
    def test_returns_report_markdown(self, MockChatOpenAI):
        """Should return the LLM response as report_markdown."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = SAMPLE_REPORT
        mock_llm.invoke.return_value = mock_response
        MockChatOpenAI.return_value = mock_llm

        from src.nodes.generate_report import generate_report

        state = {
            "clusters": [
                _make_cluster("App crashes", ["r1", "r2"], ["Quote A"]),
            ],
            "cleaned_reviews": [
                _make_review("r1", "App crashes at open"),
                _make_review("r2", "Crashes every morning"),
            ],
            "product": "Groww",
            "week_start": "2026-08-01",
            "week_end": "2026-08-07",
        }

        result = generate_report(state)

        assert "report_markdown" in result
        assert len(result["report_markdown"]) > 0
        assert "Summary of Top Themes" in result["report_markdown"]

    @patch("src.nodes.generate_report.ChatOpenAI")
    def test_prompt_contains_product_and_dates(self, MockChatOpenAI):
        """The user prompt must include product name and date range."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Report content"
        mock_llm.invoke.return_value = mock_response
        MockChatOpenAI.return_value = mock_llm

        from src.nodes.generate_report import generate_report

        state = {
            "clusters": [_make_cluster("Theme", ["r1"])],
            "cleaned_reviews": [_make_review("r1", "Review text")],
            "product": "Groww",
            "week_start": "2026-08-01",
            "week_end": "2026-08-07",
        }

        generate_report(state)

        call_args = mock_llm.invoke.call_args[0][0]
        user_msg = call_args[1].content

        assert "Groww" in user_msg
        assert "2026-08-01" in user_msg
        assert "2026-08-07" in user_msg

    @patch("src.nodes.generate_report.ChatOpenAI")
    def test_prompt_contains_theme_data(self, MockChatOpenAI):
        """The user prompt must include theme labels and quotes."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Report"
        mock_llm.invoke.return_value = mock_response
        MockChatOpenAI.return_value = mock_llm

        from src.nodes.generate_report import generate_report

        state = {
            "clusters": [
                _make_cluster("App crashes", ["r1"], ["The app crashes at 9:15"]),
            ],
            "cleaned_reviews": [_make_review("r1", "The app crashes at 9:15")],
            "product": "Groww",
            "week_start": "2026-08-01",
            "week_end": "2026-08-07",
        }

        generate_report(state)

        call_args = mock_llm.invoke.call_args[0][0]
        user_msg = call_args[1].content

        assert "App crashes" in user_msg
        assert "The app crashes at 9:15" in user_msg

    @patch("src.nodes.generate_report.ChatOpenAI")
    def test_system_prompt_forbids_invented_quotes(self, MockChatOpenAI):
        """System prompt must instruct the LLM not to invent quotes."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Report"
        mock_llm.invoke.return_value = mock_response
        MockChatOpenAI.return_value = mock_llm

        from src.nodes.generate_report import generate_report

        state = {
            "clusters": [_make_cluster("Theme", ["r1"])],
            "cleaned_reviews": [_make_review("r1", "Text")],
            "product": "Groww",
            "week_start": "2026-08-01",
            "week_end": "2026-08-07",
        }

        generate_report(state)

        call_args = mock_llm.invoke.call_args[0][0]
        system_msg = call_args[0].content

        assert "Do NOT invent quotes" in system_msg

    def test_empty_clusters_returns_empty_report(self):
        """No clusters → empty report, no LLM call."""
        from src.nodes.generate_report import generate_report

        state = {
            "clusters": [],
            "cleaned_reviews": [],
            "product": "Groww",
            "week_start": "2026-08-01",
            "week_end": "2026-08-07",
        }

        result = generate_report(state)

        assert result["report_markdown"] == ""

    @patch("src.nodes.generate_report.ChatOpenAI")
    def test_api_failure_raises_runtime_error(self, MockChatOpenAI):
        """OpenAI failure should raise a RuntimeError."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("API timeout")
        MockChatOpenAI.return_value = mock_llm

        from src.nodes.generate_report import generate_report

        state = {
            "clusters": [_make_cluster("Theme", ["r1"])],
            "cleaned_reviews": [_make_review("r1", "Text")],
            "product": "Groww",
            "week_start": "2026-08-01",
            "week_end": "2026-08-07",
        }

        with pytest.raises(RuntimeError, match="OpenAI report generation failed"):
            generate_report(state)
