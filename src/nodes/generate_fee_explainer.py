"""
Phase 4b — Generate a Fee Explainer derived from the identified fee confusion.

Using the fee/charge pain point identified in ``identify_fee_issue``,
generates a structured explanation with:
- ≤6 bullet points explaining the fee clearly
- Neutral, facts-only tone
- 2 official source links
- "Last checked: <date>" footer

This is Step 3 of the requirements — the output connects directly to
what users are confused about in reviews.
"""

import logging
from datetime import datetime, timezone

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

# ── Prompt templates ────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a financial information writer. Your task is to explain a specific \
fee or charge that users of a fintech app are confused about.

Rules:
1. Write EXACTLY 4-6 bullet points explaining the fee clearly.
2. Use a neutral, facts-only tone — no marketing language, no opinions.
3. Each bullet should be 1-2 sentences max.
4. Include exactly 2 official source links (Groww support pages, SEBI, \
   NSE/BSE, or other regulatory sources). Use real, plausible URLs.
5. End with: "Last checked: <today's date>"
6. Do NOT add any extra sections, headers, or commentary.

Format your response as:
- Bullet point 1
- Bullet point 2
- ...
- Bullet point N

Sources:
1. [Title](URL)
2. [Title](URL)

Last checked: YYYY-MM-DD
"""

_USER_PROMPT_TEMPLATE = """\
Product: {product}

User confusion identified from app reviews:
"{fee_pain_point}"

Here are {count} real user reviews expressing this confusion:
{sample_reviews}

Explain this fee/charge clearly so that a confused user would understand \
exactly what it is, why it exists, and how it is calculated.
"""


def generate_fee_explainer(state: PipelineState) -> dict:
    """LangGraph node: generate a fee explainer from the identified confusion.

    **Input** (from state):
        - ``fee_pain_point``        — the identified fee confusion (from Step 1)
        - ``fee_related_reviews``   — reviews mentioning fees
        - ``product``               — product name

    **Output**::

        {"fee_explainer": str}
    """
    fee_pain_point = state.get("fee_pain_point", "")
    fee_related_reviews = state.get("fee_related_reviews", [])
    product = state.get("product", "Groww")

    if not fee_pain_point or fee_pain_point.startswith("No specific"):
        logger.info("No fee confusion identified — skipping Fee Explainer.")
        return {"fee_explainer": ""}

    # ── Build prompt with sample reviews ─────────────────────
    sample = fee_related_reviews[:10]
    sample_text = "\n".join(
        f"{i + 1}. \"{r.get('original_text', r.get('cleaned_text', ''))}\""
        for i, r in enumerate(sample)
    )

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    user_msg = _USER_PROMPT_TEMPLATE.format(
        product=product,
        fee_pain_point=fee_pain_point,
        count=len(sample),
        sample_reviews=sample_text,
    )

    # Append today's date hint for the LLM
    user_msg += f"\n\nToday's date for 'Last checked': {today}"

    llm_kwargs = dict(
        model=OPENAI_MODEL,
        temperature=OPENAI_TEMPERATURE,
        api_key=OPENAI_API_KEY,
        max_tokens=512,
    )
    if OPENAI_API_BASE:
        llm_kwargs["base_url"] = OPENAI_API_BASE
    llm = ChatOpenAI(**llm_kwargs)

    logger.info("Generating Fee Explainer for: %s", fee_pain_point)

    try:
        response = llm.invoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ])
        fee_explainer = response.content.strip()
    except Exception as exc:
        logger.error("Fee Explainer generation failed: %s", exc)
        fee_explainer = ""

    # ── Ensure "Last checked" is present ─────────────────────
    if fee_explainer and "Last checked" not in fee_explainer:
        fee_explainer += f"\n\nLast checked: {today}"

    logger.info("Fee Explainer generated: %d characters.", len(fee_explainer))

    return {"fee_explainer": fee_explainer}
