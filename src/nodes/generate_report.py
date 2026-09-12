"""
Phase 5 — Generate weekly pulse report via OpenAI LLM.

Constructs a structured prompt from cluster data (theme labels,
review counts, representative quotes) and calls OpenAI to produce
a ≤250-word structured weekly note (Step 2 of the requirements).

Report sections:
1. Summary of top themes
2. Supporting user quotes (3 real quotes)
3. Key observation (what's going wrong / trending)
4. 3 action ideas for the product team
"""

import logging

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from src.config import (
    MAX_REPORT_WORDS,
    OPENAI_API_KEY,
    OPENAI_MAX_TOKENS,
    OPENAI_MODEL,
    OPENAI_TEMPERATURE,
)
from src.state import PipelineState

logger = logging.getLogger(__name__)

# ── Prompt templates ─────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are a product analyst. Generate a concise weekly review pulse note.\n"
    "STRICT RULES:\n"
    "- The ENTIRE note must be ≤250 words. This is a hard limit.\n"
    "- Do NOT invent quotes — only use quotes provided in the data below.\n"
    "- Use exactly 3 real user quotes total (not per theme).\n"
    "- Output well-structured Markdown with the exact section headings "
    "specified in the user message.\n"
    "- Be concise and direct. Every word must earn its place."
)

_USER_PROMPT_TEMPLATE = """\
Product: {product}
Review period: {week_start} to {week_end}
Total reviews processed: {total_reviews}

Top 3 Themes (by review volume):
{top_themes_block}

All Themes:
{themes_block}

{fee_context}

Generate a ≤250-word structured weekly note with these EXACT sections:

## Summary of Top Themes
A brief overview of the top 3 themes (2-3 sentences).

## Supporting User Quotes
Exactly 3 real user quotes (use blockquotes). Pick the most impactful ones from the themes above.

## Key Observation
What's going wrong or trending — 1-2 sentences identifying the most critical pattern or emerging issue. {fee_observation_hint}

## Action Ideas
Exactly 3 actionable suggestions for the product team (numbered list).

REMEMBER: Total word count must be ≤250 words.
"""


def _build_themes_block(clusters: list[dict], top_only: bool = False) -> str:
    """Format cluster data into a numbered text block for the prompt."""
    lines: list[str] = []
    for idx, cl in enumerate(clusters):
        if top_only and not cl.get("is_top_3", False):
            continue

        label = cl.get("label", f"Cluster {idx + 1}")
        review_ids = cl.get("review_ids", [])
        quotes = cl.get("centroid_quotes", [])

        rank = cl.get("rank", idx + 1)
        lines.append(f"Theme {rank}: {label}")
        lines.append(f"  Review count: {len(review_ids)}")

        if quotes:
            lines.append("  Sample quotes:")
            for qi, q in enumerate(quotes):
                lines.append(f"    {qi + 1}. \"{q}\"")
        else:
            lines.append("  Sample quotes: (none available)")

        lines.append("")  # blank line separator

    return "\n".join(lines)


def generate_report(state: PipelineState) -> dict:
    """LangGraph node: generate a structured ≤250-word weekly pulse note.

    **Input** (from state):
        - ``clusters``         — labelled clusters with review_ids and quotes
        - ``cleaned_reviews``  — full review records (for total count)
        - ``product``          — product name (e.g. "Groww")
        - ``week_start``       — ISO date string
        - ``week_end``         — ISO date string
        - ``fee_pain_point``   — identified fee confusion (optional)

    **Output**::

        {"report_markdown": str}
    """
    clusters = state.get("clusters", [])
    cleaned_reviews = state.get("cleaned_reviews", [])
    product = state.get("product", "Groww")
    week_start = state.get("week_start", "unknown")
    week_end = state.get("week_end", "unknown")
    fee_pain_point = state.get("fee_pain_point", "")

    if not clusters:
        logger.warning("No clusters available — generating empty report.")
        return {"report_markdown": ""}

    # ── Build the prompt ─────────────────────────────────────
    top_themes_block = _build_themes_block(clusters, top_only=True)
    themes_block = _build_themes_block(clusters, top_only=False)

    # Add fee context if available
    fee_context = ""
    fee_observation_hint = ""
    if fee_pain_point and not fee_pain_point.startswith("No specific"):
        fee_context = (
            f"Identified fee/charge confusion:\n"
            f"  \"{fee_pain_point}\"\n"
        )
        fee_observation_hint = (
            "If relevant, mention the fee/charge confusion identified above."
        )

    user_msg = _USER_PROMPT_TEMPLATE.format(
        product=product,
        week_start=week_start,
        week_end=week_end,
        total_reviews=len(cleaned_reviews),
        top_themes_block=top_themes_block,
        themes_block=themes_block,
        fee_context=fee_context,
        fee_observation_hint=fee_observation_hint,
    )

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=user_msg),
    ]

    # ── Call OpenAI ───────────────────────────────────────────
    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=OPENAI_TEMPERATURE,
        api_key=OPENAI_API_KEY,
        max_tokens=OPENAI_MAX_TOKENS,
    )

    logger.info(
        "Generating ≤%d-word report for %s (%s to %s) — %d themes, %d reviews …",
        MAX_REPORT_WORDS,
        product,
        week_start,
        week_end,
        len(clusters),
        len(cleaned_reviews),
    )

    try:
        response = llm.invoke(messages)
        report_markdown = response.content.strip()
    except Exception as exc:
        logger.error("Report generation failed: %s", exc)
        raise RuntimeError(f"OpenAI report generation failed: {exc}") from exc

    # Log word count
    word_count = len(report_markdown.split())
    logger.info(
        "Report generated: %d characters, %d words (limit: %d).",
        len(report_markdown),
        word_count,
        MAX_REPORT_WORDS,
    )

    if word_count > MAX_REPORT_WORDS:
        logger.warning(
            "Report exceeds %d-word limit (%d words). "
            "Validation will flag this for retry.",
            MAX_REPORT_WORDS,
            word_count,
        )

    return {"report_markdown": report_markdown}
