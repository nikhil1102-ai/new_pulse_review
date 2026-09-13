"""
Phase 3b — Identify recurring fee/charge confusion from clustered reviews.

Scans all clusters and their reviews for fee/charge-related content using
keyword matching, then uses OpenAI to identify the single most recurring
fee/charge pain point that users are confused about.

This output feeds directly into the Fee Explainer (Step 3).
"""

import logging

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from src.config import (
    OPENAI_API_BASE,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENAI_TEMPERATURE,
)
from src.state import PipelineState

logger = logging.getLogger(__name__)

# ── Fee-related keywords ────────────────────────────────────

_FEE_KEYWORDS = [
    "fee", "fees", "charge", "charges", "charged", "charging",
    "brokerage", "commission", "deduction", "deducted", "deduct",
    "hidden", "extra", "cost", "costs", "pricing", "price",
    "transaction", "tax", "taxes", "gst", "stt", "stamp duty",
    "dp charges", "amc", "maintenance", "penalty", "fine",
    "subscription", "premium", "plan", "free", "paid",
    "money deducted", "amount deducted", "unexpected charge",
    "why was i charged", "overcharged", "overcharge",
]

# ── Prompt templates ────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are a product analyst specialising in fintech apps. "
    "You will be given user reviews that mention fees, charges, or pricing. "
    "Your task is to identify the SINGLE most recurring confusion or pain "
    "point related to a fee or charge. "
    "Be specific — name the exact fee/charge users are confused about. "
    "Reply with a single sentence (≤30 words) describing the confusion."
)

_USER_PROMPT_TEMPLATE = """\
Product: {product}

Here are {count} user reviews that mention fees or charges:

{reviews}

What is the single most recurring fee/charge confusion or pain point?
"""


def _find_fee_related_reviews(
    clusters: list[dict],
    cleaned_reviews_lookup: dict[str, dict],
) -> list[dict]:
    """Find reviews across all clusters that mention fees/charges.

    Returns a list of review dicts that contain fee-related keywords.
    """
    fee_reviews: list[dict] = []
    seen_ids: set[str] = set()

    for cl in clusters:
        for rid in cl.get("review_ids", []):
            if rid in seen_ids:
                continue
            review = cleaned_reviews_lookup.get(rid, {})
            text = review.get("cleaned_text", "").lower()
            if any(kw in text for kw in _FEE_KEYWORDS):
                fee_reviews.append(review)
                seen_ids.add(rid)

    return fee_reviews


# ── Main node ────────────────────────────────────────────────


def identify_fee_issue(state: PipelineState) -> dict:
    """LangGraph node: identify the recurring fee/charge pain point.

    **Input** (from state):
        - ``clusters``         — labelled clusters with review_ids
        - ``cleaned_reviews``  — full review records

    **Output**::

        {
            "fee_pain_point": str,          # the identified confusion
            "fee_related_reviews": list[dict]  # reviews mentioning fees
        }
    """
    clusters = state.get("clusters", [])
    cleaned_reviews = state.get("cleaned_reviews", [])
    product = state.get("product", "Groww")

    if not clusters or not cleaned_reviews:
        logger.warning("No clusters or reviews — skipping fee detection.")
        return {"fee_pain_point": "", "fee_related_reviews": []}

    # Build lookup
    cleaned_reviews_lookup = {r["review_id"]: r for r in cleaned_reviews}

    # ── Find fee-related reviews ─────────────────────────────
    fee_reviews = _find_fee_related_reviews(clusters, cleaned_reviews_lookup)

    logger.info(
        "Found %d fee-related reviews out of %d total.",
        len(fee_reviews),
        len(cleaned_reviews),
    )

    if not fee_reviews:
        logger.info("No fee-related reviews found.")
        return {
            "fee_pain_point": "No specific fee/charge confusion identified in reviews.",
            "fee_related_reviews": [],
        }

    # ── Use OpenAI to identify the key pain point ────────────
    # Sample up to 30 fee-related reviews for the prompt
    sample_reviews = fee_reviews[:30]
    numbered = "\n".join(
        f"{i + 1}. \"{r.get('original_text', r.get('cleaned_text', ''))}\""
        for i, r in enumerate(sample_reviews)
    )

    user_msg = _USER_PROMPT_TEMPLATE.format(
        product=product,
        count=len(sample_reviews),
        reviews=numbered,
    )

    llm_kwargs = dict(
        model=OPENAI_MODEL,
        temperature=OPENAI_TEMPERATURE,
        api_key=OPENAI_API_KEY,
        max_tokens=128,
    )
    if OPENAI_API_BASE:
        llm_kwargs["base_url"] = OPENAI_API_BASE
    llm = ChatOpenAI(**llm_kwargs)

    try:
        response = llm.invoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ])
        fee_pain_point = response.content.strip()
    except Exception as exc:
        logger.error("Fee issue identification failed: %s", exc)
        fee_pain_point = "Unable to identify fee confusion — LLM call failed."

    logger.info("Identified fee pain point: %s", fee_pain_point)

    return {
        "fee_pain_point": fee_pain_point,
        "fee_related_reviews": fee_reviews,
    }
