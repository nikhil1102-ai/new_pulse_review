"""
Phase 2a — Chunk reviews for embedding.

Implements a three-tier chunking strategy based on token count:
  - ≤ 512 tokens  → whole-review (single chunk)
  - 513–1024      → sentence-split (nltk.sent_tokenize)
  - > 1024        → sliding window (512 tokens, 128 overlap)

In practice, Groww reviews are short (median ~9 tokens, max ~495),
so nearly all reviews fall into the whole-review bucket.
"""

import logging

import tiktoken

from src.state import PipelineState

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────
MAX_WHOLE_REVIEW_TOKENS = 512
MAX_SENTENCE_SPLIT_TOKENS = 1024
SLIDING_WINDOW_SIZE = 512
SLIDING_WINDOW_OVERLAP = 128

# Use cl100k_base as a proxy tokeniser (compatible with JINA token counts)
_ENCODING = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    """Count tokens using tiktoken."""
    return len(_ENCODING.encode(text))


def _chunk_whole_review(review: dict, review_id: str) -> list[dict]:
    """Single chunk = entire review text."""
    return [
        {
            "chunk_id": f"{review_id}_chunk_0",
            "review_id": review_id,
            "chunk_index": 0,
            "chunk_text": review.get("cleaned_text", ""),
            "total_chunks": 1,
        }
    ]


def _chunk_sentence_split(review: dict, review_id: str) -> list[dict]:
    """Split review into individual sentences."""
    # Lazy import to avoid nltk download issues at module level
    import nltk
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)

    from nltk.tokenize import sent_tokenize

    text = review.get("cleaned_text", "")
    sentences = sent_tokenize(text)

    if not sentences:
        return _chunk_whole_review(review, review_id)

    chunks = []
    for i, sentence in enumerate(sentences):
        chunks.append(
            {
                "chunk_id": f"{review_id}_chunk_{i}",
                "review_id": review_id,
                "chunk_index": i,
                "chunk_text": sentence,
                "total_chunks": len(sentences),
            }
        )
    return chunks


def _chunk_sliding_window(review: dict, review_id: str) -> list[dict]:
    """Slide a 512-token window with 128-token overlap across the text."""
    text = review.get("cleaned_text", "")
    tokens = _ENCODING.encode(text)

    chunks = []
    start = 0
    chunk_index = 0

    while start < len(tokens):
        end = min(start + SLIDING_WINDOW_SIZE, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = _ENCODING.decode(chunk_tokens)

        chunks.append(
            {
                "chunk_id": f"{review_id}_chunk_{chunk_index}",
                "review_id": review_id,
                "chunk_index": chunk_index,
                "chunk_text": chunk_text,
                "total_chunks": -1,  # placeholder, set below
            }
        )

        chunk_index += 1
        start += SLIDING_WINDOW_SIZE - SLIDING_WINDOW_OVERLAP

        if end == len(tokens):
            break

    # Set total_chunks now that we know the count
    for c in chunks:
        c["total_chunks"] = len(chunks)

    return chunks


def chunk(state: PipelineState) -> dict:
    """
    LangGraph node: chunk reviews into embeddable units.

    Input:  state["cleaned_reviews"]  — PII-scrubbed reviews from Phase 2
    Output: {"chunks": list[dict]}

    Chunking decision:
      ≤ 512 tokens  → whole-review (single chunk)   [~100% of Groww data]
      513–1024      → sentence-split
      > 1024        → sliding window (512 tokens, 128 overlap)
    """
    cleaned_reviews = state.get("cleaned_reviews", [])

    all_chunks: list[dict] = []
    bucket_counts = {"whole": 0, "sentence": 0, "sliding": 0}

    for review in cleaned_reviews:
        review_id = review.get("review_id", "")
        text = review.get("cleaned_text", "")
        token_count = _count_tokens(text)

        if token_count <= MAX_WHOLE_REVIEW_TOKENS:
            chunks = _chunk_whole_review(review, review_id)
            bucket_counts["whole"] += 1
        elif token_count <= MAX_SENTENCE_SPLIT_TOKENS:
            chunks = _chunk_sentence_split(review, review_id)
            bucket_counts["sentence"] += 1
        else:
            chunks = _chunk_sliding_window(review, review_id)
            bucket_counts["sliding"] += 1

        all_chunks.extend(chunks)

    logger.info(
        "Chunking complete: %d reviews → %d chunks "
        "(whole: %d, sentence-split: %d, sliding-window: %d)",
        len(cleaned_reviews),
        len(all_chunks),
        bucket_counts["whole"],
        bucket_counts["sentence"],
        bucket_counts["sliding"],
    )

    return {"chunks": all_chunks}
