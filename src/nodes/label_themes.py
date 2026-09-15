"""
Phase 3 — Label cluster themes via the LLM.

For each cluster produced by the ``cluster`` node, samples reviews and asks the
model to summarise the common theme in a few words.

Two details matter here:

*Token budget.* The configured model (``openai/gpt-oss-120b``) is a reasoning
model: it spends tokens thinking before it emits any visible content. Too small
a ceiling and the reasoning consumes the entire budget, leaving ``content``
empty and every theme falling back to a placeholder. Hence
``OPENAI_LABEL_MAX_TOKENS`` is generous relative to the handful of words we
actually want back.

*Fallbacks.* When labelling genuinely fails, the cluster is named from its own
most distinctive words rather than "Theme 3", so the report stays readable
instead of merely well-formed.
"""

import logging
import re
from collections import Counter

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from src.config import (
    CLUSTER_SAMPLE_SIZE,
    OPENAI_API_BASE,
    OPENAI_API_KEY,
    OPENAI_LABEL_MAX_TOKENS,
    OPENAI_MODEL,
    OPENAI_TEMPERATURE,
    THEME_LABEL_MAX_WORDS,
)
from src.state import PipelineState

logger = logging.getLogger(__name__)

# ── Prompt templates ─────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are a product analyst. You will be given a set of user reviews for "
    "a mobile application. Identify the single common theme and name it as a "
    f"short category of {THEME_LABEL_MAX_WORDS} words or fewer.\n"
    "Rules:\n"
    "- Reply with ONLY the category name. No explanation, no preamble, "
    "no quotation marks, no trailing full stop.\n"
    "- Name the specific problem or topic, not a generic bucket. "
    "Write 'Fund withdrawal delays', not 'Negative feedback' or 'Theme 1'.\n"
    "- Use sentence case, e.g. 'App crashes during market hours'."
)

_USER_PROMPT_TEMPLATE = (
    "Here are {count} user reviews. Name their common theme in "
    f"{THEME_LABEL_MAX_WORDS} words or fewer:\n\n"
    "{reviews}"
)

# A retry nudge for when the model returns nothing usable.
_RETRY_SUFFIX = (
    "\n\nRespond with the category name only — a short noun phrase, "
    "nothing else."
)

# Labels that are structurally valid but say nothing. Rejecting these is what
# stops "Theme 1" reaching the report via the model itself.
_USELESS_LABELS = {
    "theme", "themes", "cluster", "clusters", "general", "other", "others",
    "misc", "miscellaneous", "feedback", "user feedback", "reviews",
    "review", "negative feedback", "positive feedback", "mixed feedback",
    "n/a", "none", "unknown", "various",
}

# Common words carry no signal in a keyword-derived label.
_STOPWORDS = {
    "the", "and", "for", "are", "but", "not", "you", "your", "with", "this",
    "that", "have", "has", "had", "was", "were", "been", "its", "it's", "from",
    "they", "them", "then", "than", "can", "cant", "will", "would", "could",
    "should", "there", "their", "what", "when", "why", "how", "all", "any",
    "get", "got", "very", "just", "only", "also", "too", "out", "about",
    "after", "before", "even", "more", "most", "much", "some", "such", "into",
    "over", "under", "again", "once", "does", "did", "doing", "done", "app",
    "application", "please", "thanks", "thank", "good", "bad", "best", "worst",
    "like", "dont", "don't", "i'm", "im", "not", "one", "two", "use", "using",
    "used", "make", "made", "want", "need", "know", "time", "every", "always",
    "never", "still", "now", "back", "give", "given", "take", "taken", "see",
    "keep", "keeps", "really", "lot", "many", "going", "way", "thing", "things",
    "review", "reviews", "user", "users", "star", "stars",
}

_WORD_RE = re.compile(r"[a-z']{3,}")


# ── Helpers ──────────────────────────────────────────────────


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


def _clean_label(raw: str) -> str:
    """Normalise a model response into a usable category name.

    Strips the wrappers models habitually add — quotation marks, Markdown
    emphasis, bullet markers, a "Theme:" prefix, a trailing full stop — and
    caps the result at the configured word count. Returns an empty string when
    nothing usable survives.
    """
    if not raw:
        return ""

    label = str(raw).strip()

    # Reasoning models sometimes emit several lines; the name is the last
    # non-empty one, after any thinking.
    lines = [ln.strip() for ln in label.splitlines() if ln.strip()]
    if lines:
        label = lines[-1]

    label = re.sub(r"^[-*•\d.)\s]+", "", label)          # bullets, numbering
    label = re.sub(r"^(theme|category|label)\s*[:\-]\s*", "", label, flags=re.I)
    label = label.replace("**", "").replace("`", "").replace("#", "")
    label = label.strip().strip('"').strip("'").strip()
    label = re.sub(r"\s+", " ", label).rstrip(".").strip()

    if not label:
        return ""

    words = label.split()
    if len(words) > THEME_LABEL_MAX_WORDS:
        label = " ".join(words[:THEME_LABEL_MAX_WORDS])

    # A label that is only a placeholder is no better than no label at all.
    if label.lower() in _USELESS_LABELS:
        return ""
    if re.fullmatch(r"(theme|cluster)\s*\d+", label, flags=re.I):
        return ""

    return label


def _keyword_label(review_texts: list[str]) -> str:
    """Derive a label from a cluster's own most frequent meaningful words.

    The fallback of last resort. Far from elegant, but "Charges, deducted,
    refund" tells a reader what the cluster is about, whereas "Theme 3" does
    not.
    """
    counts: Counter = Counter()
    for text in review_texts:
        for word in _WORD_RE.findall(str(text).lower()):
            if word not in _STOPWORDS:
                counts[word] += 1

    top = [word for word, _ in counts.most_common(3)]
    if not top:
        return ""

    return ", ".join(w.capitalize() if i == 0 else w for i, w in enumerate(top))


def _label_single_cluster(llm: ChatOpenAI, review_texts: list[str]) -> str:
    """Ask the model for a theme label, retrying once if nothing usable comes back.

    Returns an empty string if both attempts fail, leaving the caller to fall
    back to a keyword label.
    """
    numbered = "\n".join(f"{i + 1}. {text}" for i, text in enumerate(review_texts))
    user_msg = _USER_PROMPT_TEMPLATE.format(
        count=len(review_texts), reviews=numbered
    )

    for attempt in (1, 2):
        message = user_msg if attempt == 1 else user_msg + _RETRY_SUFFIX
        response = llm.invoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=message),
        ])

        label = _clean_label(getattr(response, "content", ""))
        if label:
            return label

        logger.warning(
            "Labelling attempt %d returned nothing usable (content=%r).",
            attempt,
            getattr(response, "content", None),
        )

    return ""


# ── Main node ────────────────────────────────────────────────


def label_themes(state: PipelineState) -> dict:
    """LangGraph node: generate a short theme label for each cluster.

    **Input** (from state):
        - ``clusters``          — list of cluster dicts (label=None)
        - ``cleaned_reviews``   — full review records (for text sampling)

    **Output**::

        {"clusters": [...]}   # same list, with ``label`` populated

    Every cluster comes back with a non-empty, meaningful label: from the
    model where possible, otherwise from the cluster's own keywords.
    """
    clusters = state.get("clusters", [])
    cleaned_reviews = state.get("cleaned_reviews", [])

    if not clusters:
        logger.warning("No clusters to label.")
        return {"clusters": []}

    cleaned_reviews_lookup = {r["review_id"]: r for r in cleaned_reviews}

    llm_kwargs = dict(
        model=OPENAI_MODEL,
        temperature=OPENAI_TEMPERATURE,
        api_key=OPENAI_API_KEY,
        # Generous on purpose: a reasoning model spends tokens thinking before
        # it writes anything, and a tight cap yields an empty response.
        max_tokens=OPENAI_LABEL_MAX_TOKENS,
    )
    if OPENAI_API_BASE:
        llm_kwargs["base_url"] = OPENAI_API_BASE
    llm = ChatOpenAI(**llm_kwargs)

    labelled_clusters: list[dict] = []
    fallback_count = 0

    for idx, cl in enumerate(clusters):
        review_ids = cl.get("review_ids", [])
        review_texts = _sample_reviews(review_ids, cleaned_reviews_lookup)

        label = ""

        if review_texts:
            try:
                label = _label_single_cluster(llm, review_texts)
            except Exception as exc:
                logger.error("Failed to label cluster %d: %s", idx, exc)
        else:
            logger.warning("Cluster %d has no sampleable reviews.", idx)

        if not label:
            label = _keyword_label(review_texts)
            if label:
                fallback_count += 1
                logger.warning(
                    "Cluster %d labelled from keywords: %r", idx, label
                )

        if not label:
            # Nothing to work with at all — a positional name is all that is left.
            label = f"Unnamed theme {idx + 1}"
            fallback_count += 1
            logger.error("Cluster %d could not be labelled at all.", idx)

        cl["label"] = label
        labelled_clusters.append(cl)

        logger.info(
            'Cluster %d: "%s" (%d reviews, %d sampled)',
            idx,
            label,
            len(review_ids),
            len(review_texts),
        )

    if fallback_count:
        logger.warning(
            "%d of %d themes used a fallback label. Check the model response "
            "and OPENAI_LABEL_MAX_TOKENS (currently %d).",
            fallback_count,
            len(labelled_clusters),
            OPENAI_LABEL_MAX_TOKENS,
        )

    logger.info(
        "Theme labelling complete: %d clusters labelled.", len(labelled_clusters)
    )

    return {"clusters": labelled_clusters}
