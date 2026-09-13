"""
Unit tests for Phase 6b — analytics derived for the detailed PDF report.

Covers src/analytics.py: the ingestion funnel, rating distribution, weekly
trend, per-theme breakdown, version breakdown, top quotes and fee stats.
"""

import pytest

from src.analytics import (
    MIN_VERSION_SAMPLE,
    compute_fee_stats,
    compute_funnel,
    compute_insights,
    compute_rating_distribution,
    compute_theme_breakdown,
    compute_top_quotes,
    compute_version_breakdown,
    compute_weekly_trend,
)


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────


def _make_review(
    review_id: str,
    rating: int = 4,
    date: str = "2026-08-15",
    thumbs: int = 0,
    version: str | None = "18.15.1",
    text: str = "Some review text",
) -> dict:
    """Create a minimal cleaned-review dict."""
    return {
        "review_id": review_id,
        "original_text": text,
        "cleaned_text": text.lower(),
        "rating": rating,
        "date": date,
        "author_name": "User",
        "thumbs_up_count": thumbs,
        "app_version": version,
    }


@pytest.fixture
def reviews() -> list[dict]:
    """Twelve reviews spanning three weeks, two versions and all ratings."""
    return [
        _make_review("r1", 1, "2026-08-10", 50, "18.15.1"),
        _make_review("r2", 1, "2026-08-11", 30, "18.15.1"),
        _make_review("r3", 2, "2026-08-12", 10, "18.15.1"),
        _make_review("r4", 5, "2026-08-13", 0, "18.15.1"),
        _make_review("r5", 5, "2026-08-17", 1, "18.16.0"),
        _make_review("r6", 4, "2026-08-18", 2, "18.16.0"),
        _make_review("r7", 3, "2026-08-19", 0, "18.16.0"),
        _make_review("r8", 5, "2026-08-24", 0, None),
        _make_review("r9", 5, "2026-08-25", 0, None),
        _make_review("r10", 4, "2026-08-26", 5, "18.15.1"),
        _make_review("r11", 5, "2026-08-27", 0, "18.15.1"),
        _make_review("r12", 5, "2026-08-28", 0, "18.15.1"),
    ]


@pytest.fixture
def clusters() -> list[dict]:
    """Two clusters: a small angry one and a larger happy one."""
    return [
        {
            "label": "Crashes on login",
            "review_ids": ["r1", "r2", "r3"],
            "centroid_quotes": ["it crashes"],
            "rank": 1,
            "is_top_3": True,
        },
        {
            "label": "Great experience",
            "review_ids": ["r4", "r5", "r6", "r11", "r12"],
            "centroid_quotes": ["love it"],
            "rank": 2,
            "is_top_3": True,
        },
    ]


# ──────────────────────────────────────────────────────────────
# Funnel
# ──────────────────────────────────────────────────────────────


class TestFunnel:
    def test_stages_and_percentages(self, reviews, clusters):
        state = {
            "raw_reviews": [{}] * 20,
            "cleaned_reviews": reviews,
            "chunks": [{}] * 14,
            "clusters": clusters,
            "noise_reviews": reviews[:2],
        }
        funnel = compute_funnel(state)

        assert [f["count"] for f in funnel] == [20, 12, 14, 8, 2]
        assert funnel[0]["pct_of_raw"] == 100.0
        assert funnel[1]["pct_of_raw"] == 60.0

    def test_empty_state_does_not_divide_by_zero(self):
        funnel = compute_funnel({})
        assert all(f["count"] == 0 and f["pct_of_raw"] == 0.0 for f in funnel)


# ──────────────────────────────────────────────────────────────
# Ratings
# ──────────────────────────────────────────────────────────────


class TestRatingDistribution:
    def test_counts_average_and_negative_share(self, reviews):
        dist = compute_rating_distribution(reviews)

        assert dist["total"] == 12
        assert dist["negative_count"] == 3  # two 1-star, one 2-star
        assert dist["negative_pct"] == 25.0
        assert dist["average"] == pytest.approx(3.75, abs=0.01)

    def test_buckets_are_ordered_five_to_one(self, reviews):
        dist = compute_rating_distribution(reviews)
        assert [b["stars"] for b in dist["buckets"]] == [5, 4, 3, 2, 1]

    def test_ignores_out_of_range_and_malformed_ratings(self):
        dist = compute_rating_distribution([
            _make_review("a", 5),
            {"review_id": "b", "rating": 0},
            {"review_id": "c", "rating": 9},
            {"review_id": "d", "rating": "bad"},
            {"review_id": "e"},
        ])
        assert dist["total"] == 1

    def test_empty_input(self):
        dist = compute_rating_distribution([])
        assert dist["total"] == 0 and dist["average"] == 0.0


# ──────────────────────────────────────────────────────────────
# Weekly trend
# ──────────────────────────────────────────────────────────────


class TestWeeklyTrend:
    def test_groups_by_monday_anchored_week(self, reviews):
        trend = compute_weekly_trend(reviews)

        assert [t["week"] for t in trend] == [
            "2026-08-10", "2026-08-17", "2026-08-24",
        ]
        assert [t["count"] for t in trend] == [4, 3, 5]

    def test_sorted_oldest_first(self, reviews):
        trend = compute_weekly_trend(list(reversed(reviews)))
        assert [t["week"] for t in trend] == sorted(t["week"] for t in trend)

    def test_skips_unparseable_dates(self):
        trend = compute_weekly_trend([
            _make_review("a", 5, "2026-08-10"),
            _make_review("b", 5, "not-a-date"),
            _make_review("c", 5, ""),
        ])
        assert len(trend) == 1 and trend[0]["count"] == 1


# ──────────────────────────────────────────────────────────────
# Themes
# ──────────────────────────────────────────────────────────────


class TestThemeBreakdown:
    def test_volume_share_and_severity(self, reviews, clusters):
        themes = compute_theme_breakdown(clusters, reviews)

        happy = next(t for t in themes if t["label"] == "Great experience")
        angry = next(t for t in themes if t["label"] == "Crashes on login")

        assert angry["count"] == 3
        assert happy["count"] == 5
        assert angry["avg_rating"] < happy["avg_rating"]
        assert angry["negative_pct"] == 100.0
        assert happy["negative_pct"] == 0.0
        assert angry["share_pct"] + happy["share_pct"] == pytest.approx(100.0)

    def test_sorted_by_volume_descending(self, reviews, clusters):
        themes = compute_theme_breakdown(clusters, reviews)
        counts = [t["count"] for t in themes]
        assert counts == sorted(counts, reverse=True)

    def test_aggregates_thumbs_up(self, reviews, clusters):
        themes = compute_theme_breakdown(clusters, reviews)
        angry = next(t for t in themes if t["label"] == "Crashes on login")
        assert angry["thumbs_up"] == 90  # 50 + 30 + 10

    def test_unlabelled_cluster_gets_placeholder(self, reviews):
        themes = compute_theme_breakdown(
            [{"label": None, "review_ids": ["r1"]}], reviews
        )
        assert themes[0]["label"].startswith("Theme")

    def test_tolerates_review_ids_missing_from_corpus(self, reviews):
        themes = compute_theme_breakdown(
            [{"label": "Ghost", "review_ids": ["nope1", "nope2"]}], reviews
        )
        assert themes[0]["count"] == 2
        assert themes[0]["avg_rating"] == 0.0


# ──────────────────────────────────────────────────────────────
# Versions
# ──────────────────────────────────────────────────────────────


class TestVersionBreakdown:
    def test_applies_minimum_sample_floor(self, reviews):
        # None of the fixture versions clear the floor
        assert compute_version_breakdown(reviews) == []

    def test_includes_versions_above_floor(self):
        many = [
            _make_review(f"r{i}", 5, version="18.15.1")
            for i in range(MIN_VERSION_SAMPLE)
        ]
        rows = compute_version_breakdown(many)
        assert len(rows) == 1
        assert rows[0]["version"] == "18.15.1"
        assert rows[0]["count"] == MIN_VERSION_SAMPLE

    def test_excludes_reviews_with_no_version(self):
        many = [
            _make_review(f"r{i}", 5, version=None)
            for i in range(MIN_VERSION_SAMPLE * 2)
        ]
        assert compute_version_breakdown(many) == []


# ──────────────────────────────────────────────────────────────
# Top quotes
# ──────────────────────────────────────────────────────────────


class TestTopQuotes:
    def test_only_negative_reviews_sorted_by_upvotes(self, reviews):
        quotes = compute_top_quotes(reviews)

        assert all(q["rating"] <= 2 for q in quotes)
        assert [q["thumbs_up"] for q in quotes] == [50, 30, 10]

    def test_respects_limit(self, reviews):
        assert len(compute_top_quotes(reviews, limit=2)) == 2

    def test_empty_when_no_negative_reviews(self):
        assert compute_top_quotes([_make_review("a", 5)]) == []


# ──────────────────────────────────────────────────────────────
# Fees
# ──────────────────────────────────────────────────────────────


class TestFeeStats:
    def test_counts_and_share(self, reviews):
        stats = compute_fee_stats({
            "cleaned_reviews": reviews,
            "fee_related_reviews": reviews[:3],
            "fee_pain_point": "Hidden charges",
            "fee_explainer": "- bullet",
        })

        assert stats["count"] == 3
        assert stats["pct_of_corpus"] == 25.0
        assert stats["has_explainer"] is True

    def test_no_fee_reviews(self):
        stats = compute_fee_stats({"cleaned_reviews": [], "fee_related_reviews": []})
        assert stats["count"] == 0 and stats["pct_of_corpus"] == 0.0


# ──────────────────────────────────────────────────────────────
# Aggregator
# ──────────────────────────────────────────────────────────────


class TestComputeInsights:
    def test_bundle_contains_every_section(self, reviews, clusters):
        insights = compute_insights({
            "product": "Groww",
            "week_start": "2026-08-10",
            "week_end": "2026-08-28",
            "raw_reviews": [{}] * 20,
            "cleaned_reviews": reviews,
            "chunks": [{}] * 14,
            "clusters": clusters,
            "noise_reviews": [],
            "fee_related_reviews": reviews[:3],
            "fee_pain_point": "Hidden charges",
            "fee_explainer": "- bullet",
            "validation_passed": True,
            "validation_errors": [],
            "retry_count": 0,
        })

        for key in (
            "product", "week_start", "week_end", "funnel", "ratings",
            "weekly_trend", "themes", "versions", "top_quotes", "fees",
            "validation",
        ):
            assert key in insights, f"Missing insights key: {key}"

    def test_empty_state_does_not_raise(self):
        insights = compute_insights({})
        assert insights["themes"] == []
        assert insights["ratings"]["total"] == 0
