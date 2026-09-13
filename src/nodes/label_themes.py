"""
Phase 3 — Label cluster themes via OpenAI LLM.

For each cluster produced by the ``cluster`` node, samples 15–20 reviews
and asks OpenAI (gpt-oss-120b) to summarise the common theme in ≤ 10 words.
"""

import logging

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from src.config import (
    CLUSTER_SAMPLE_SIZE,
    OPENAI_API_BASE,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENAI_TEMPERATURE,
)
from src.state import PipelineState

logger = logging.getLogger(__name__)

# ── Prompt templates ─────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are a product analyst. You will be given a set of user reviews for "
    "a mobile application. Your task is to identify and summarise the single "
    "common theme shared by these reviews in 10 words or fewer. "
    "Reply ONLY with the theme label — no explanation, no punctuation."
)

_USER_PROMPT_TEMPLATE = (
    "Here are {count} user reviews. Summarise their common theme in ≤ 10 words:\n\n"
    "{reviews}"
)


def _sample_reviews(
    review_ids: list[str],
    cleaned_reviews_lookup: dict[str, dict],
    n: int = CLUSTER_SAMPLE_SIZE,
) -> list[str]:
    """Sample up to *n* review texts from a cluster for the labelling prompt.

    Uses ``original_text`` (preserves casing) when available, falling
    back to ``cleaned_text``.
    """
    sampled: list[str] = []
    for rid in review_ids[:n]:
        review = cleaned_reviews_lookup.get(rid, {})
        text = review.get("original_text", review.get("cleaned_text", ""))
        if text:
            sampled.append(text)
    return sampled


def _label_single_cluster(
    llm: ChatOpenAI,
    review_texts: list[str],
) -> str:
    """Call OpenAI to produce a ≤ 10-word theme label for a list of reviews."""
    numbered = "\n".join(
        f"{i + 1}. {text}" for i, text in enumerate(review_texts)
    )
    user_msg = _USER_PROMPT_TEMPLATE.format(
        count=len(review_texts),
        reviews=numbered,
    )

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=user_msg),
    ]

    response = llm.invoke(messages)
    label = response.content.strip().strip('"').strip("'")
    return label


# ── Main node ────────────────────────────────────────────────


def label_themes(state: PipelineState) -> dict:
    """LangGraph node: generate a short theme label for each cluster.

    **Input** (from state):
        - ``clusters``          — list of cluster dicts (label=None)
        - ``cleaned_reviews``   — full review records (for text sampling)

    **Output**::

        {"clusters": [...]}   # same list, with ``label`` populated
    """
    clusters = state.get("clusters", [])
    cleaned_reviews = state.get("cleaned_reviews", [])

    if not clusters:
        logger.warning("No clusters to label.")
        return {"clusters": []}

    # Build review lookup
    cleaned_reviews_lookup = {r["review_id"]: r for r in cleaned_reviews}

    # Initialise OpenAI LLM
    llm_kwargs = dict(
        model=OPENAI_MODEL,
        temperature=OPENAI_TEMPERATURE,
        api_key=OPENAI_API_KEY,
        max_tokens=64,
    )
    if OPENAI_API_BASE:
        llm_kwargs["base_url"] = OPENAI_API_BASE
    llm = ChatOpenAI(**llm_kwargs)

    labelled_clusters: list[dict] = []

    for idx, cl in enumerate(clusters):
        review_ids = cl.get("review_ids", [])
        review_texts = _sample_reviews(review_ids, cleaned_reviews_lookup)

        if not review_texts:
            logger.warning("Cluster %d has no sampleable reviews — skipping.", idx)
            cl["label"] = f"Cluster {idx + 1}"
            labelled_clusters.append(cl)
            continue

        try:
            label = _label_single_cluster(llm, review_texts)
        except Exception as exc:
            logger.error("Failed to label cluster %d: %s", idx, exc)
            label = f"Cluster {idx + 1}"

        cl["label"] = label
        labelled_clusters.append(cl)

        logger.info(
            "Cluster %d: \"%s\" (%d reviews, %d sampled)",
            idx,
            label,
            len(review_ids),
            len(review_texts),
        )

    logger.info("Theme labelling complete: %d clusters labelled.", len(labelled_clusters))

    return {"clusters": labelled_clusters}
