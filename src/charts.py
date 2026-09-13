"""
Chart rendering for the detailed PDF report.

Each function takes a slice of the analytics bundle from :mod:`src.analytics`
and returns PNG bytes, ready to embed via ``reportlab.platypus.Image``.

``matplotlib`` is configured for the headless Agg backend at import time so
this works inside the Docker container with no display attached. Every
renderer returns ``None`` rather than raising when there is nothing to plot —
a missing chart degrades the PDF, it should never fail the pipeline.
"""

from __future__ import annotations

import io
import logging

import matplotlib

# Must be set before pyplot is imported — no display in Docker/Railway.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

logger = logging.getLogger(__name__)

# matplotlib logs category/font details at INFO, which would swamp the
# pipeline's own INFO output on every chart.
logging.getLogger("matplotlib").setLevel(logging.WARNING)

# ── Palette (matched to the HTML email styling) ──────────────
INK = "#1a1a2e"
ACCENT = "#1a73e8"
MUTED = "#8a8a9e"
GRID = "#e8e8e8"
# 1-2 stars red, 3 amber, 4-5 green — severity reads at a glance
RATING_COLORS = {5: "#2e7d32", 4: "#66bb6a", 3: "#fbc02d", 2: "#ef6c00", 1: "#c62828"}

DPI = 150


def _finish(fig) -> bytes:
    """Serialise *fig* to PNG bytes and release it."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def _style_axes(ax) -> None:
    """Apply the shared minimal axis styling."""
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.set_axisbelow(True)


def rating_distribution_chart(ratings: dict) -> bytes | None:
    """Horizontal bar chart of the star-rating mix."""
    buckets = [b for b in ratings.get("buckets", []) if b.get("count")]
    if not buckets:
        logger.warning("No rating data — skipping rating chart.")
        return None

    labels = [f"{b['stars']}★" for b in buckets]
    counts = [b["count"] for b in buckets]
    colors = [RATING_COLORS.get(b["stars"], ACCENT) for b in buckets]

    fig, ax = plt.subplots(figsize=(5.4, 2.6))
    bars = ax.barh(labels, counts, color=colors, height=0.68)
    ax.invert_yaxis()
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_xlabel("Reviews", color=MUTED, fontsize=8)
    _style_axes(ax)

    span = max(counts) or 1
    for bar, bucket in zip(bars, buckets):
        ax.text(
            bar.get_width() + span * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{bucket['count']:,} ({bucket['pct']}%)",
            va="center",
            fontsize=8,
            color=INK,
        )
    ax.set_xlim(0, span * 1.22)

    return _finish(fig)


def weekly_trend_chart(trend: list[dict]) -> bytes | None:
    """Review volume (bars) against average rating (line) per week."""
    if len(trend) < 2:
        logger.warning("Need >=2 weeks for a trend chart — skipping.")
        return None

    weeks = [t["week"][5:] for t in trend]  # MM-DD keeps labels short
    counts = [t["count"] for t in trend]
    averages = [t["avg_rating"] for t in trend]

    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    ax.bar(weeks, counts, color=ACCENT, alpha=0.28, width=0.6, label="Review volume")
    ax.set_ylabel("Reviews", color=MUTED, fontsize=8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    _style_axes(ax)

    ax2 = ax.twinx()
    ax2.plot(
        weeks, averages, color=INK, marker="o", markersize=4.5,
        linewidth=1.8, label="Avg rating",
    )
    ax2.set_ylabel("Avg rating", color=MUTED, fontsize=8)
    ax2.set_ylim(1, 5)
    ax2.tick_params(colors=MUTED, labelsize=8, length=0)
    for spine in ("top", "left", "right"):
        ax2.spines[spine].set_visible(False)

    for x, y in zip(weeks, averages):
        ax2.annotate(
            f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 8),
            ha="center", fontsize=7.5, color=INK,
        )

    handles = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.18),
        ncol=2, frameon=False, fontsize=8, labelcolor=MUTED,
    )

    return _finish(fig)


def theme_share_chart(themes: list[dict], limit: int = 8) -> bytes | None:
    """Horizontal bars of review volume per theme, coloured by severity.

    Bar colour encodes average rating, so a small-but-angry theme is visually
    distinct from a large-but-neutral one.
    """
    rows = [t for t in themes if t.get("count")][:limit]
    if not rows:
        logger.warning("No themes — skipping theme share chart.")
        return None

    labels = [
        (t["label"][:42] + "…") if len(t["label"]) > 42 else t["label"] for t in rows
    ]
    counts = [t["count"] for t in rows]

    def _severity_color(avg: float) -> str:
        if avg <= 2.0:
            return RATING_COLORS[1]
        if avg <= 3.0:
            return RATING_COLORS[2]
        if avg <= 3.8:
            return RATING_COLORS[3]
        return RATING_COLORS[4]

    colors = [_severity_color(t.get("avg_rating", 0)) for t in rows]

    fig, ax = plt.subplots(figsize=(7.2, 0.46 * len(rows) + 1.1))
    bars = ax.barh(labels, counts, color=colors, height=0.62)
    ax.invert_yaxis()
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_xlabel("Reviews in theme", color=MUTED, fontsize=8)
    _style_axes(ax)
    ax.tick_params(axis="y", labelsize=8.5)

    span = max(counts) or 1
    for bar, theme in zip(bars, rows):
        ax.text(
            bar.get_width() + span * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{theme['count']:,}  ({theme['share_pct']}%)  {theme['avg_rating']}★",
            va="center",
            fontsize=7.5,
            color=INK,
        )
    ax.set_xlim(0, span * 1.32)

    return _finish(fig)


def render_all(insights: dict) -> dict:
    """Render every chart, tolerating individual failures.

    Returns a dict of chart-name → PNG bytes, omitting any that could not be
    produced.
    """
    renderers = {
        "ratings": lambda: rating_distribution_chart(insights.get("ratings", {})),
        "trend": lambda: weekly_trend_chart(insights.get("weekly_trend", [])),
        "themes": lambda: theme_share_chart(insights.get("themes", [])),
    }

    charts: dict[str, bytes] = {}
    for name, render in renderers.items():
        try:
            png = render()
            if png:
                charts[name] = png
        except Exception as exc:
            logger.error("Chart '%s' failed to render: %s", name, exc)

    logger.info("Rendered %d/%d charts.", len(charts), len(renderers))
    return charts
