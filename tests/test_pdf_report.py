"""
Unit tests for Phase 6b — PDF rendering and the ``generate_pdf`` node.

Covers src/charts.py, src/pdf_report.py and src/nodes/generate_pdf.py.
Charts and the PDF are rendered for real (both libraries are pure-Python and
fast enough), then asserted on structurally.
"""

import os
from unittest.mock import patch

import pytest

from src.analytics import compute_insights
from src.charts import (
    rating_distribution_chart,
    render_all,
    theme_share_chart,
    weekly_trend_chart,
)
from src.nodes.generate_pdf import _pdf_filename, generate_pdf
from src.pdf_report import _markdown_block, _rich, _truncate, build_pdf

PNG_MAGIC = b"\x89PNG"
PDF_MAGIC = b"%PDF-"


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────


def _make_review(review_id: str, rating: int = 4, date: str = "2026-08-15") -> dict:
    return {
        "review_id": review_id,
        "original_text": f"Review text for {review_id}",
        "cleaned_text": f"review text for {review_id}",
        "rating": rating,
        "date": date,
        "author_name": "User",
        "thumbs_up_count": 3,
        "app_version": "18.15.1",
    }


@pytest.fixture
def state() -> dict:
    reviews = [
        _make_review(f"r{i}", rating=(i % 5) + 1,
                     date=f"2026-08-{10 + (i % 20):02d}")
        for i in range(40)
    ]
    clusters = [
        {"label": "Crashes on login", "review_ids": [r["review_id"] for r in reviews[:20]],
         "centroid_quotes": ["It crashes at market open"], "rank": 1, "is_top_3": True},
        {"label": "Hidden charges", "review_ids": [r["review_id"] for r in reviews[20:]],
         "centroid_quotes": ["Why was I charged?"], "rank": 2, "is_top_3": True},
    ]
    return {
        "product": "Groww",
        "week_start": "2026-08-10",
        "week_end": "2026-08-30",
        "raw_reviews": [{}] * 50,
        "cleaned_reviews": reviews,
        "chunks": [{}] * 45,
        "clusters": clusters,
        "noise_reviews": [],
        "fee_related_reviews": reviews[20:],
        "fee_pain_point": "Unexplained brokerage deductions",
        "fee_explainer": "- Brokerage is zero on delivery trades.\n\nLast checked: 2026-09-13",
        "report_markdown": "## Summary of Top Themes\nUsers report crashes.",
        "validation_passed": True,
        "validation_errors": [],
        "retry_count": 0,
    }


@pytest.fixture
def insights(state) -> dict:
    return compute_insights(state)


# ──────────────────────────────────────────────────────────────
# Charts
# ──────────────────────────────────────────────────────────────


class TestCharts:
    def test_rating_chart_returns_png(self, insights):
        png = rating_distribution_chart(insights["ratings"])
        assert png and png.startswith(PNG_MAGIC)

    def test_trend_chart_returns_png(self, insights):
        png = weekly_trend_chart(insights["weekly_trend"])
        assert png and png.startswith(PNG_MAGIC)

    def test_theme_chart_returns_png(self, insights):
        png = theme_share_chart(insights["themes"])
        assert png and png.startswith(PNG_MAGIC)

    def test_charts_return_none_on_empty_input(self):
        assert rating_distribution_chart({"buckets": []}) is None
        assert theme_share_chart([]) is None

    def test_trend_needs_at_least_two_weeks(self):
        assert weekly_trend_chart([{"week": "2026-08-10", "count": 5,
                                    "avg_rating": 4.0}]) is None

    def test_render_all_returns_every_chart(self, insights):
        charts = render_all(insights)
        assert set(charts) == {"ratings", "trend", "themes"}
        assert all(v.startswith(PNG_MAGIC) for v in charts.values())

    def test_render_all_survives_a_failing_renderer(self, insights):
        """One broken chart must not lose the others."""
        with patch("src.charts.weekly_trend_chart", side_effect=RuntimeError("boom")):
            charts = render_all(insights)
        assert "trend" not in charts
        assert "ratings" in charts and "themes" in charts


# ──────────────────────────────────────────────────────────────
# Text helpers
# ──────────────────────────────────────────────────────────────


class TestTextHelpers:
    def test_escapes_markup_before_applying_emphasis(self):
        """Review text must never inject ReportLab markup."""
        out = _rich("<b>not bold</b> & <font color='red'>x</font>")
        assert "&lt;b&gt;" in out and "&amp;" in out
        assert "<b>not bold</b>" not in out

    def test_converts_bold_italic_and_links(self):
        assert "<b>hi</b>" in _rich("**hi**")
        assert "<i>hi</i>" in _rich("*hi*")
        assert '<link href="https://x.test"' in _rich("[label](https://x.test)")

    def test_handles_none_and_empty(self):
        assert _rich(None) == "" and _rich("") == ""

    def test_truncate_on_word_boundary(self):
        assert _truncate("one two three four", 9).endswith("...")
        assert _truncate("short", 100) == "short"

    def test_truncate_collapses_whitespace(self):
        assert _truncate("a   b\n\nc", 100) == "a b c"

    def test_markdown_block_handles_every_construct(self):
        flowables = _markdown_block(
            "## Heading\n"
            "Plain paragraph\n"
            "- bullet item\n"
            "1. numbered item\n"
            "> a quote\n"
            "---\n",
            __import__("src.pdf_report", fromlist=["_styles"])._styles(),
        )
        assert len(flowables) == 6

    def test_markdown_block_empty_input(self):
        from src.pdf_report import _styles
        assert _markdown_block("", _styles()) == []


# ──────────────────────────────────────────────────────────────
# PDF document
# ──────────────────────────────────────────────────────────────


class TestBuildPdf:
    def test_produces_a_valid_pdf(self, insights, state):
        pdf = build_pdf(insights, state["report_markdown"], state["fee_explainer"],
                        render_all(insights))
        assert pdf.startswith(PDF_MAGIC)
        assert len(pdf) > 5000

    def test_builds_without_charts(self, insights, state):
        """Chart failures must not prevent the document from rendering."""
        pdf = build_pdf(insights, state["report_markdown"], state["fee_explainer"], {})
        assert pdf.startswith(PDF_MAGIC)

    def test_builds_with_no_fee_explainer(self, insights):
        pdf = build_pdf(insights, "## Summary\ntext", "", {})
        assert pdf.startswith(PDF_MAGIC)

    def test_builds_from_empty_insights(self):
        pdf = build_pdf(compute_insights({}), "", "", {})
        assert pdf.startswith(PDF_MAGIC)

    def test_review_text_cannot_break_the_renderer(self, state):
        """Adversarial review text is data, never markup."""
        state["cleaned_reviews"][0]["original_text"] = (
            '<font color="bogus">&</font> <unclosed> 100% <b>'
        )
        state["cleaned_reviews"][0]["rating"] = 1
        state["clusters"][0]["centroid_quotes"] = ['<para>evil</para> & "quotes"']
        pdf = build_pdf(compute_insights(state), "", "", {})
        assert pdf.startswith(PDF_MAGIC)


# ──────────────────────────────────────────────────────────────
# Node
# ──────────────────────────────────────────────────────────────


class TestGeneratePdfNode:
    def test_writes_pdf_and_returns_path(self, state, tmp_path):
        with patch("src.nodes.generate_pdf.REPORTS_DIR", str(tmp_path)):
            result = generate_pdf(state)

        assert result["pdf_path"] is not None
        assert os.path.exists(result["pdf_path"])
        with open(result["pdf_path"], "rb") as f:
            assert f.read(5) == PDF_MAGIC
        assert "insights" in result

    def test_respects_pdf_enabled_flag(self, state, tmp_path):
        with patch("src.nodes.generate_pdf.PDF_ENABLED", False):
            with patch("src.nodes.generate_pdf.REPORTS_DIR", str(tmp_path)):
                result = generate_pdf(state)

        assert result["pdf_path"] is None
        assert result["insights"]  # analytics still computed
        assert os.listdir(tmp_path) == []

    def test_render_failure_does_not_break_pipeline(self, state, tmp_path):
        with patch("src.pdf_report.build_pdf", side_effect=RuntimeError("boom")):
            with patch("src.nodes.generate_pdf.REPORTS_DIR", str(tmp_path)):
                result = generate_pdf(state)

        assert result["pdf_path"] is None
        assert "insights" in result

    def test_filename_is_slugified(self):
        assert _pdf_filename("Groww", "2026-08-10") == (
            "groww_weekly_pulse_2026-08-10.pdf"
        )
        assert _pdf_filename("My App/v2", "2026-08-10").startswith("my_app_v2")

    def test_filename_falls_back_when_product_has_no_safe_chars(self):
        assert _pdf_filename("///", "2026-08-10").startswith("report_")
