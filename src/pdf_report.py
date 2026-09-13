"""
Detailed PDF report builder (ReportLab).

The emailed summary is deliberately short (≤250 words, top 3 themes only).
This document is the deep version: the full theme table with severity and
engagement, the ingestion funnel, rating mix, week-over-week movement,
app-version correlation, the fee analysis in full, and a methodology and
validation appendix.

Only the base-14 fonts are used, so the output is portable and needs no font
files in the Docker image. Note that ReportLab encodes those fonts as
WinAnsi/Latin-1, so glyphs outside that range (star symbols, emoji) are not
safe here — charts carry the symbols instead.
"""

from __future__ import annotations

import io
import logging
import re
from datetime import datetime, timezone
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

logger = logging.getLogger(__name__)

# ── Palette (matched to src/charts.py and the HTML email) ────
INK = colors.HexColor("#1a1a2e")
ACCENT = colors.HexColor("#1a73e8")
MUTED = colors.HexColor("#6b6b80")
RULE = colors.HexColor("#e8e8e8")
BAND = colors.HexColor("#f4f7fb")
GOOD = colors.HexColor("#2e7d32")
WARN = colors.HexColor("#ef6c00")
BAD = colors.HexColor("#c62828")

PAGE_W, PAGE_H = A4
MARGIN = 18 * mm
CONTENT_W = PAGE_W - 2 * MARGIN


# ── Styles ───────────────────────────────────────────────────


def _styles() -> dict:
    """Build the paragraph styles used throughout the document."""
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title", parent=base["Title"], fontName="Helvetica-Bold",
            fontSize=23, leading=27, textColor=INK, alignment=TA_CENTER,
            spaceAfter=2,
        ),
        "subtitle": ParagraphStyle(
            "subtitle", parent=base["Normal"], fontSize=10.5, leading=14,
            textColor=MUTED, alignment=TA_CENTER, spaceAfter=14,
        ),
        "h1": ParagraphStyle(
            "h1", parent=base["Heading1"], fontName="Helvetica-Bold",
            fontSize=14.5, leading=18, textColor=INK,
            spaceBefore=16, spaceAfter=7,
        ),
        "h2": ParagraphStyle(
            "h2", parent=base["Heading2"], fontName="Helvetica-Bold",
            fontSize=11.5, leading=15, textColor=INK,
            spaceBefore=11, spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "body", parent=base["Normal"], fontSize=9.7, leading=14.2,
            textColor=colors.HexColor("#333340"), spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "small", parent=base["Normal"], fontSize=8.2, leading=11.5,
            textColor=MUTED, spaceAfter=4,
        ),
        "quote": ParagraphStyle(
            "quote", parent=base["Normal"], fontName="Helvetica-Oblique",
            fontSize=9.4, leading=13.5, textColor=colors.HexColor("#44445a"),
            leftIndent=9, spaceBefore=3, spaceAfter=3,
        ),
        "bullet": ParagraphStyle(
            "bullet", parent=base["Normal"], fontSize=9.7, leading=14,
            textColor=colors.HexColor("#333340"), leftIndent=12,
            bulletIndent=3, spaceAfter=3.5,
        ),
        "kpi_num": ParagraphStyle(
            "kpi_num", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=17, leading=20, textColor=INK, alignment=TA_CENTER,
        ),
        "kpi_lbl": ParagraphStyle(
            "kpi_lbl", parent=base["Normal"], fontSize=7.4, leading=9.5,
            textColor=MUTED, alignment=TA_CENTER,
        ),
        "th": ParagraphStyle(
            "th", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=8.4, leading=11, textColor=colors.white,
        ),
        "td": ParagraphStyle(
            "td", parent=base["Normal"], fontSize=8.4, leading=11.4,
            textColor=colors.HexColor("#333340"),
        ),
    }


# ── Text helpers ─────────────────────────────────────────────


def _rich(text: str) -> str:
    """Escape text for ReportLab, then re-apply inline Markdown emphasis.

    Handles ``**bold**``, ``*italic*`` and ``[label](url)``. Escaping happens
    first so review text can never inject ReportLab markup.
    """
    safe = escape(str(text or ""))
    safe = re.sub(r"\[([^\]]+)\]\((https?://[^\s)]+)\)",
                  r'<link href="\2" color="#1a73e8">\1</link>', safe)
    safe = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", safe)
    safe = re.sub(r"(?<!\*)\*([^*\n]+?)\*(?!\*)", r"<i>\1</i>", safe)
    return safe


def _truncate(text: str, limit: int) -> str:
    """Trim *text* to *limit* characters on a word boundary."""
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "..."


def _severity(avg_rating: float) -> colors.Color:
    """Colour for an average-rating cell."""
    if avg_rating <= 2.0:
        return BAD
    if avg_rating <= 3.4:
        return WARN
    return GOOD


# ── Reusable components ──────────────────────────────────────


def _kpi_row(insights: dict, st: dict) -> Table:
    """The headline metric strip under the cover title."""
    ratings = insights.get("ratings", {})
    fees = insights.get("fees", {})
    cells = [
        (f"{ratings.get('total', 0):,}", "Reviews analysed"),
        (f"{ratings.get('average', 0):.2f}", "Average rating"),
        (f"{ratings.get('negative_pct', 0):.0f}%", "1-2 star share"),
        (str(len(insights.get("themes", []))), "Themes found"),
        (f"{fees.get('pct_of_corpus', 0):.0f}%", "Fee-related"),
    ]
    data = [
        [Paragraph(v, st["kpi_num"]) for v, _ in cells],
        [Paragraph(l, st["kpi_lbl"]) for _, l in cells],
    ]
    table = Table(data, colWidths=[CONTENT_W / len(cells)] * len(cells))
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), BAND),
        ("BOX", (0, 0), (-1, -1), 0.6, RULE),
        ("INNERGRID", (0, 0), (-1, -1), 0.6, colors.white),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, 0), 9),
        ("BOTTOMPADDING", (0, 1), (-1, 1), 9),
    ]))
    return table


def _data_table(header: list[str], rows: list[list], widths: list[float],
                st: dict, align_right: list[int] | None = None) -> Table:
    """A styled table with a dark header band and zebra striping."""
    align_right = align_right or []
    data = [[Paragraph(h, st["th"]) for h in header]]
    for row in rows:
        data.append([
            cell if isinstance(cell, Paragraph) else Paragraph(str(cell), st["td"])
            for cell in row
        ])

    table = Table(data, colWidths=widths, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), INK),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.4, RULE),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]
    for idx in range(1, len(data)):
        if idx % 2 == 0:
            style.append(("BACKGROUND", (0, idx), (-1, idx), BAND))
    for col in align_right:
        style.append(("ALIGN", (col, 0), (col, -1), "RIGHT"))

    table.setStyle(TableStyle(style))
    return table


def _chart(png: bytes | None, width: float = CONTENT_W) -> list:
    """Wrap PNG bytes in a width-constrained, aspect-preserving Image."""
    if not png:
        return []
    try:
        img = Image(io.BytesIO(png))
        img.drawWidth = width
        img.drawHeight = width * (img.imageHeight / float(img.imageWidth))
        return [img, Spacer(1, 7)]
    except Exception as exc:
        logger.error("Failed to embed chart: %s", exc)
        return []


def _markdown_block(markdown: str, st: dict) -> list:
    """Render a Markdown subset into flowables.

    Supports ``##``/``###`` headings, ``-``/``*`` bullets, ``N.`` numbered
    items, ``>`` blockquotes and ``---`` rules. Good enough for the LLM output,
    which is prompt-constrained to exactly these constructs.
    """
    flow: list = []
    for raw_line in str(markdown or "").split("\n"):
        line = raw_line.strip()

        if not line:
            continue
        if line in ("---", "***", "___"):
            flow.append(HRFlowable(width="100%", thickness=0.6, color=RULE,
                                   spaceBefore=6, spaceAfter=6))
        elif line.startswith("###"):
            flow.append(Paragraph(_rich(line.lstrip("#").strip()), st["h2"]))
        elif line.startswith("##"):
            flow.append(Paragraph(_rich(line.lstrip("#").strip()), st["h2"]))
        elif line.startswith(">"):
            flow.append(Paragraph(_rich(line.lstrip(">").strip()), st["quote"]))
        elif re.match(r"^[-*]\s+", line):
            flow.append(Paragraph(_rich(re.sub(r"^[-*]\s+", "", line)),
                                  st["bullet"], bulletText="•"))
        elif re.match(r"^\d+[.)]\s+", line):
            marker = re.match(r"^(\d+)[.)]\s+", line).group(1)
            flow.append(Paragraph(_rich(re.sub(r"^\d+[.)]\s+", "", line)),
                                  st["bullet"], bulletText=f"{marker}."))
        else:
            flow.append(Paragraph(_rich(line), st["body"]))
    return flow


# ── Page furniture ───────────────────────────────────────────


def _page_decoration(canvas, doc) -> None:
    """Draw the footer rule, source note and page number on every page."""
    canvas.saveState()
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.5)
    canvas.line(MARGIN, 13 * mm, PAGE_W - MARGIN, 13 * mm)

    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(MARGIN, 9 * mm,
                      f"{doc.product} - Weekly Review Pulse - Google Play reviews")
    canvas.drawRightString(PAGE_W - MARGIN, 9 * mm, f"Page {canvas.getPageNumber()}")
    canvas.restoreState()


# ── Sections ─────────────────────────────────────────────────


def _cover(insights: dict, st: dict) -> list:
    """Title block, KPI strip and the at-a-glance takeaway."""
    product = insights.get("product", "Groww")
    generated = datetime.now(timezone.utc).strftime("%d %b %Y, %H:%M UTC")

    flow = [
        Spacer(1, 4),
        Paragraph(f"{escape(product)} - Weekly Review Pulse", st["title"]),
        Paragraph(
            f"Google Play reviews - {insights.get('week_start', '')} to "
            f"{insights.get('week_end', '')}<br/>Generated {generated}",
            st["subtitle"],
        ),
        _kpi_row(insights, st),
        Spacer(1, 13),
    ]

    themes = insights.get("themes", [])
    ratings = insights.get("ratings", {})
    if themes:
        worst = min(themes, key=lambda t: t.get("avg_rating", 5))
        flow += [
            Paragraph("At a glance", st["h1"]),
            Paragraph(
                f"The largest theme this period is <b>{escape(themes[0]['label'])}</b>, "
                f"covering {themes[0]['count']:,} reviews "
                f"({themes[0]['share_pct']}% of all themed reviews). "
                f"The most negative theme is <b>{escape(worst['label'])}</b> at "
                f"{worst['avg_rating']} stars average, with "
                f"{worst['negative_pct']}% of its reviews rated 1-2 stars. "
                f"Across the whole corpus the average rating is "
                f"{ratings.get('average', 0)} and "
                f"{ratings.get('negative_count', 0):,} reviews "
                f"({ratings.get('negative_pct', 0)}%) are 1-2 stars.",
                st["body"],
            ),
        ]
    return flow


def _executive_summary(report_markdown: str, st: dict) -> list:
    """The same ≤250-word note that goes out by email, for standalone context."""
    if not report_markdown:
        return []
    return [
        Paragraph("Executive summary", st["h1"]),
        Paragraph(
            "This is the summary circulated by email. The sections that follow "
            "expand on it with the supporting data.", st["small"],
        ),
        Spacer(1, 3),
        *_markdown_block(report_markdown, st),
    ]


def _landscape(insights: dict, charts: dict, st: dict) -> list:
    """Ingestion funnel, rating mix and the weekly trend."""
    flow = [PageBreak(), Paragraph("Review landscape", st["h1"])]

    funnel = insights.get("funnel", [])
    if funnel:
        flow += [
            Paragraph("How the corpus was filtered", st["h2"]),
            _data_table(
                ["Stage", "Reviews", "% of fetched"],
                [[f["stage"], f"{f['count']:,}", f"{f['pct_of_raw']}%"]
                 for f in funnel],
                [CONTENT_W * 0.58, CONTENT_W * 0.21, CONTENT_W * 0.21],
                st, align_right=[1, 2],
            ),
            Spacer(1, 5),
            Paragraph(
                "Chunk count can exceed review count: long reviews are split "
                "before embedding, then reassembled to their parent review "
                "after clustering.", st["small"],
            ),
        ]

    if charts.get("ratings"):
        flow += [Paragraph("Rating distribution", st["h2"]),
                 *_chart(charts["ratings"], CONTENT_W * 0.74)]

    if charts.get("trend"):
        flow += [Paragraph("Volume and sentiment by week", st["h2"]),
                 *_chart(charts["trend"])]

    return flow


def _themes_section(insights: dict, charts: dict, st: dict) -> list:
    """Theme comparison chart, full table, and per-theme detail cards."""
    themes = insights.get("themes", [])
    if not themes:
        return []

    flow = [PageBreak(), Paragraph("Theme analysis", st["h1"]),
            Paragraph(
                "Bar colour encodes severity: a smaller theme with a low average "
                "rating can matter more than a larger, neutral one.", st["small"]),
            Spacer(1, 3)]

    flow += _chart(charts.get("themes"))

    rows = []
    for theme in themes:
        avg = theme.get("avg_rating", 0)
        rows.append([
            str(theme.get("rank", "")),
            _truncate(theme.get("label", ""), 58),
            f"{theme.get('count', 0):,}",
            f"{theme.get('share_pct', 0)}%",
            Paragraph(
                f'<font color="#{_severity(avg).hexval()[2:]}"><b>{avg}</b></font>',
                st["td"],
            ),
            f"{theme.get('negative_pct', 0)}%",
            f"{theme.get('thumbs_up', 0):,}",
        ])

    flow += [
        Paragraph("All themes ranked by volume", st["h2"]),
        _data_table(
            ["#", "Theme", "Reviews", "Share", "Avg", "1-2 star", "Upvotes"],
            rows,
            [CONTENT_W * w for w in (0.05, 0.40, 0.12, 0.10, 0.09, 0.12, 0.12)],
            st, align_right=[2, 3, 4, 5, 6],
        ),
        Spacer(1, 4),
        Paragraph(
            "Upvotes are the total thumbs-up other users gave reviews in the "
            "theme - a proxy for how widely the experience is shared.", st["small"]),
    ]

    top = [t for t in themes if t.get("is_top_3")][:3] or themes[:3]
    if top:
        flow += [Paragraph("Top themes in detail", st["h1"])]
    for theme in top:
        card = [
            Paragraph(
                f"{theme.get('rank', '')}. {escape(theme.get('label', ''))}",
                st["h2"],
            ),
            Paragraph(
                f"{theme.get('count', 0):,} reviews - {theme.get('share_pct', 0)}% "
                f"of themed volume - {theme.get('avg_rating', 0)} average rating - "
                f"{theme.get('negative_pct', 0)}% rated 1-2 stars",
                st["small"],
            ),
        ]
        for quote in theme.get("quotes", [])[:3]:
            card.append(Paragraph(f'"{_rich(_truncate(quote, 300))}"', st["quote"]))
        card.append(Spacer(1, 7))
        flow.append(KeepTogether(card))

    return flow


def _fee_section(insights: dict, fee_explainer: str, st: dict) -> list:
    """Fee pain point, its volume, and the customer-ready explainer."""
    fees = insights.get("fees", {})
    if not fees.get("pain_point"):
        return []

    flow = [
        PageBreak(),
        Paragraph("Fee and charge analysis", st["h1"]),
        Paragraph(
            f"<b>Identified confusion:</b> {_rich(fees['pain_point'])}", st["body"]),
        _data_table(
            ["Metric", "Value"],
            [
                ["Reviews mentioning fees or charges", f"{fees.get('count', 0):,}"],
                ["Share of analysed corpus", f"{fees.get('pct_of_corpus', 0)}%"],
                ["Average rating of those reviews", f"{fees.get('avg_rating', 0)}"],
            ],
            [CONTENT_W * 0.66, CONTENT_W * 0.34], st, align_right=[1],
        ),
        Spacer(1, 9),
    ]

    if fee_explainer:
        flow += [
            Paragraph("Customer-ready explainer", st["h1"]),
            Paragraph(
                "Reusable support snippet - neutral, facts-only, with official "
                "sources. Safe to paste into a ticket reply or help-centre entry.",
                st["small"]),
            Spacer(1, 3),
            *_markdown_block(fee_explainer, st),
        ]
    return flow


def _voice_section(insights: dict, st: dict) -> list:
    """The most-upvoted negative reviews verbatim."""
    quotes = insights.get("top_quotes", [])
    if not quotes:
        return []

    flow = [
        Paragraph("Most-upvoted complaints", st["h1"]),
        Paragraph(
            "Low-rated reviews other users endorsed most - these carry "
            "disproportionate weight in what prospective users read first.",
            st["small"]),
        Spacer(1, 3),
    ]
    for q in quotes:
        flow.append(KeepTogether([
            Paragraph(f'"{_rich(_truncate(q.get("text", ""), 420))}"', st["quote"]),
            Paragraph(
                f"{q.get('rating', 0)} star{'' if q.get('rating') == 1 else 's'} - "
                f"{q.get('thumbs_up', 0):,} upvotes - "
                f"{escape(str(q.get('date', '')))}", st["small"]),
            Spacer(1, 5),
        ]))
    return flow


def _versions_section(insights: dict, st: dict) -> list:
    """Average rating by app version."""
    versions = insights.get("versions", [])
    if not versions:
        return []
    return [
        Paragraph("Rating by app version", st["h1"]),
        Paragraph(
            "Versions with at least 25 reviews, most-reviewed first. A version "
            "whose average sits well below its neighbours is worth correlating "
            "with its release notes.", st["small"]),
        Spacer(1, 3),
        _data_table(
            ["Version", "Reviews", "Avg rating", "1-2 star share"],
            [[v["version"], f"{v['count']:,}", f"{v['avg_rating']}",
              f"{v['negative_pct']}%"] for v in versions[:12]],
            [CONTENT_W * 0.28, CONTENT_W * 0.22, CONTENT_W * 0.25, CONTENT_W * 0.25],
            st, align_right=[1, 2, 3],
        ),
    ]


def _appendix(insights: dict, st: dict) -> list:
    """Methodology and the validation/audit trail."""
    validation = insights.get("validation", {})
    status = "PASSED" if validation.get("passed") else "PASSED WITH WARNINGS"
    errors = validation.get("errors", [])

    flow = [
        PageBreak(),
        Paragraph("Methodology", st["h1"]),
        *_markdown_block(
            "- Reviews are scraped from the public Google Play listing for the "
            "configured package, over the report window.\n"
            "- Text is Unicode-normalised, stripped of HTML and URLs, filtered to "
            "English, and short entries are dropped.\n"
            "- Duplicates are removed exactly by review ID and near-duplicates via "
            "MinHash LSH at Jaccard 0.90, keeping the earliest of each group.\n"
            "- Phone numbers, emails, account IDs and names are redacted before any "
            "text reaches the language model.\n"
            "- Reviews are embedded with jina-embeddings-v3 (1024 dimensions), "
            "reduced with UMAP and clustered with HDBSCAN, falling back to K-Means "
            "if too few clusters emerge.\n"
            "- Themes are labelled by the language model from a sample of each "
            "cluster; every quoted line is fuzzy-matched back to its source review "
            "before the report is released.",
            st,
        ),
        Paragraph("Validation and audit", st["h1"]),
        _data_table(
            ["Check", "Result"],
            [
                ["Quote authenticity and structure", status],
                ["Report regeneration attempts", str(validation.get("retry_count", 0))],
                ["Warnings recorded", str(len(errors))],
            ],
            [CONTENT_W * 0.6, CONTENT_W * 0.4], st,
        ),
    ]

    if errors:
        flow += [Spacer(1, 6), Paragraph("Recorded warnings", st["h2"])]
        flow += [Paragraph(_rich(f"- {e}"), st["small"]) for e in errors[:12]]

    flow += [
        Spacer(1, 9),
        Paragraph(
            "Quotes are reproduced from public Google Play reviews after PII "
            "redaction. Review text is treated strictly as data and never as "
            "instructions to the language model.", st["small"]),
    ]
    return flow


# ── Entry point ──────────────────────────────────────────────


def build_pdf(
    insights: dict,
    report_markdown: str = "",
    fee_explainer: str = "",
    charts: dict | None = None,
) -> bytes:
    """Render the full report and return the PDF as bytes.

    Args:
        insights:        Bundle from :func:`src.analytics.compute_insights`.
        report_markdown: The ≤250-word emailed summary, included for context.
        fee_explainer:   Markdown fee explainer, if one was generated.
        charts:          Name → PNG bytes from :func:`src.charts.render_all`.

    Returns:
        The complete PDF document as bytes.
    """
    charts = charts or {}
    st = _styles()
    buf = io.BytesIO()

    product = insights.get("product", "Groww")
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=22 * mm,
        title=f"{product} - Weekly Review Pulse - {insights.get('week_start', '')}",
        author="AI Review Pulse Pipeline",
        subject="Weekly Google Play review analysis",
    )
    doc.product = product  # read back by the page decorator

    story: list = []
    story += _cover(insights, st)
    story += _executive_summary(report_markdown, st)
    story += _landscape(insights, charts, st)
    story += _themes_section(insights, charts, st)
    story += _fee_section(insights, fee_explainer, st)
    story += _voice_section(insights, st)
    story += _versions_section(insights, st)
    story += _appendix(insights, st)

    doc.build(story, onFirstPage=_page_decoration, onLaterPages=_page_decoration)

    pdf = buf.getvalue()
    logger.info("PDF built: %d pages worth, %.1f KB", doc.page, len(pdf) / 1024)
    return pdf
