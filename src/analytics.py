"""
Derived analytics for the detailed PDF report.

Pure functions over ``PipelineState`` — no rendering or I/O dependencies, so
this module is cheap to import and straightforward to unit-test.

The weekly email carries a ≤250-word summary of the top 3 themes. The PDF is
meant to go deeper, so everything computed here is intentionally *additional*
signal that the email does not show: the ingestion funnel, rating mix, severity
per theme, week-over-week movement, and app-version correlation.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from datetime import date, datetime

from src.state import PipelineState

logger = logging.getLogger(__name__)

# Reviews with no app_version reported (~8% of Google Play rows)
UNKNOWN_VERSION = "(not reported)"

# Only surface versions with at least this many reviews — below it the
# average rating is noise rather than signal.
MIN_VERSION_SAMPLE = 25


# ── Helpers ──────────────────────────────────────────────────


def _parse_date(value: str) -> date | None:
    """Parse an ISO ``YYYY-MM-DD`` date, tolerating junk and datetimes."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value)[:10]).date()
    except (ValueError, TypeError):
        return None


def _mean(values: list[float]) -> float:
    """Arithmetic mean, or 0.0 for an empty list."""
    return round(sum(values) / len(values), 2) if values else 0.0


def _pct(part: int, whole: int) -> float:
    """*part* as a percentage of *whole*, guarding division by zero."""
    return round(100.0 * part / whole, 1) if whole else 0.0


def _week_label(d: date) -> str:
    """Monday-anchored week label, e.g. ``2026-08-10``."""
    return date.fromordinal(d.toordinal() - d.weekday()).isoformat()


# ── Individual metrics ───────────────────────────────────────


def compute_funnel(state: PipelineState) -> list[dict]:
    """Reviews retained at each ingestion stage.

    Shows reviewers how much of the raw scrape actually reached the analysis,
    which is the main credibility question a stakeholder asks first.
    """
    raw = len(state.get("raw_reviews", []))
    cleaned = len(state.get("cleaned_reviews", []))
    chunks = len(state.get("chunks", []))
    clusters = state.get("clusters", [])
    clustered = sum(len(c.get("review_ids", [])) for c in clusters)
    noise = len(state.get("noise_reviews", []))

    stages = [
        ("Fetched from Google Play", raw),
        ("Survived cleaning, dedup & PII scrub", cleaned),
        ("Embedding chunks produced", chunks),
        ("Assigned to a theme", clustered),
        ("Outliers (no theme)", noise),
    ]
    return [
        {"stage": name, "count": count, "pct_of_raw": _pct(count, raw)}
        for name, count in stages
    ]


def compute_rating_distribution(reviews: list[dict]) -> dict:
    """Star-rating histogram plus the overall average and negative share."""
    counts: Counter = Counter()
    for r in reviews:
        try:
            stars = int(r.get("rating", 0))
        except (TypeError, ValueError):
            continue
        if 1 <= stars <= 5:
            counts[stars] += 1

    total = sum(counts.values())
    ratings = [s for s, n in counts.items() for _ in range(n)]
    negative = counts[1] + counts[2]

    return {
        "total": total,
        "average": _mean(ratings),
        "negative_count": negative,
        "negative_pct": _pct(negative, total),
        "buckets": [
            {"stars": s, "count": counts[s], "pct": _pct(counts[s], total)}
            for s in (5, 4, 3, 2, 1)
        ],
    }


def compute_weekly_trend(reviews: list[dict]) -> list[dict]:
    """Review volume and average rating per calendar week, oldest first."""
    by_week: dict[str, list[int]] = defaultdict(list)

    for r in reviews:
        d = _parse_date(r.get("date", ""))
        if d is None:
            continue
        try:
            by_week[_week_label(d)].append(int(r.get("rating", 0)))
        except (TypeError, ValueError):
            continue

    return [
        {
            "week": week,
            "count": len(ratings),
            "avg_rating": _mean([x for x in ratings if 1 <= x <= 5]),
        }
        for week, ratings in sorted(by_week.items())
    ]


def compute_theme_breakdown(
    clusters: list[dict], reviews: list[dict]
) -> list[dict]:
    """Per-theme volume, share, severity and engagement.

    ``avg_rating`` is the severity signal the email never shows: two themes of
    equal size are not equally urgent if one averages 1.8 stars and the other
    4.3.
    """
    lookup = {r.get("review_id"): r for r in reviews}
    total_clustered = sum(len(c.get("review_ids", [])) for c in clusters)

    breakdown: list[dict] = []
    for idx, cluster in enumerate(clusters):
        review_ids = cluster.get("review_ids", [])
        members = [lookup[rid] for rid in review_ids if rid in lookup]

        ratings: list[int] = []
        thumbs = 0
        for m in members:
            try:
                stars = int(m.get("rating", 0))
            except (TypeError, ValueError):
                stars = 0
            if 1 <= stars <= 5:
                ratings.append(stars)
            thumbs += int(m.get("thumbs_up_count", 0) or 0)

        negative = sum(1 for s in ratings if s <= 2)

        breakdown.append(
            {
                "rank": cluster.get("rank", idx + 1),
                "label": cluster.get("label") or f"Unnamed theme {idx + 1}",
                "count": len(review_ids),
                "share_pct": _pct(len(review_ids), total_clustered),
                "avg_rating": _mean(ratings),
                "negative_pct": _pct(negative, len(ratings)),
                "thumbs_up": thumbs,
                "is_top_3": cluster.get("is_top_3", idx < 3),
                "quotes": cluster.get("centroid_quotes", []),
            }
        )

    breakdown.sort(key=lambda t: t["count"], reverse=True)
    return breakdown


def compute_version_breakdown(reviews: list[dict]) -> list[dict]:
    """Average rating by app version, for versions above the sample floor.

    Useful for spotting a release that measurably moved sentiment.
    """
    by_version: dict[str, list[int]] = defaultdict(list)

    for r in reviews:
        version = r.get("app_version") or UNKNOWN_VERSION
        try:
            stars = int(r.get("rating", 0))
        except (TypeError, ValueError):
            continue
        if 1 <= stars <= 5:
            by_version[str(version)].append(stars)

    rows = [
        {
            "version": version,
            "count": len(ratings),
            "avg_rating": _mean(ratings),
            "negative_pct": _pct(sum(1 for s in ratings if s <= 2), len(ratings)),
        }
        for version, ratings in by_version.items()
        if len(ratings) >= MIN_VERSION_SAMPLE and version != UNKNOWN_VERSION
    ]
    rows.sort(key=lambda v: v["count"], reverse=True)
    return rows


def compute_top_quotes(reviews: list[dict], limit: int = 5) -> list[dict]:
    """Most-upvoted negative reviews — complaints that resonated with others."""
    negatives = []
    for r in reviews:
        try:
            stars = int(r.get("rating", 0))
        except (TypeError, ValueError):
            continue
        if 1 <= stars <= 2:
            negatives.append(r)

    negatives.sort(key=lambda r: int(r.get("thumbs_up_count", 0) or 0), reverse=True)

    return [
        {
            "text": r.get("original_text", r.get("cleaned_text", "")),
            "rating": r.get("rating", 0),
            "thumbs_up": int(r.get("thumbs_up_count", 0) or 0),
            "date": r.get("date", ""),
        }
        for r in negatives[:limit]
        if r.get("original_text") or r.get("cleaned_text")
    ]


def compute_fee_stats(state: PipelineState) -> dict:
    """Volume and severity of fee/charge-related reviews."""
    fee_reviews = state.get("fee_related_reviews", [])
    cleaned = state.get("cleaned_reviews", [])

    ratings: list[int] = []
    for r in fee_reviews:
        try:
            stars = int(r.get("rating", 0))
        except (TypeError, ValueError):
            continue
        if 1 <= stars <= 5:
            ratings.append(stars)

    return {
        "count": len(fee_reviews),
        "pct_of_corpus": _pct(len(fee_reviews), len(cleaned)),
        "avg_rating": _mean(ratings),
        "pain_point": state.get("fee_pain_point", ""),
        "has_explainer": bool(state.get("fee_explainer")),
    }


# ── Aggregator ───────────────────────────────────────────────


def compute_insights(state: PipelineState) -> dict:
    """Build the full analytics bundle consumed by the PDF renderer.

    Every value is derived from state already in the graph — this adds no API
    calls and no extra cost.
    """
    cleaned = state.get("cleaned_reviews", [])
    clusters = state.get("clusters", [])

    insights = {
        "product": state.get("product", "Groww"),
        "week_start": state.get("week_start", ""),
        "week_end": state.get("week_end", ""),
        "funnel": compute_funnel(state),
        "ratings": compute_rating_distribution(cleaned),
        "weekly_trend": compute_weekly_trend(cleaned),
        "themes": compute_theme_breakdown(clusters, cleaned),
        "versions": compute_version_breakdown(cleaned),
        "top_quotes": compute_top_quotes(cleaned),
        "fees": compute_fee_stats(state),
        "validation": {
            "passed": state.get("validation_passed", False),
            "errors": state.get("validation_errors", []),
            "retry_count": state.get("retry_count", 0),
        },
    }

    logger.info(
        "Insights computed: %d themes, %d reviews, avg rating %.2f",
        len(insights["themes"]),
        insights["ratings"]["total"],
        insights["ratings"]["average"],
    )
    return insights
