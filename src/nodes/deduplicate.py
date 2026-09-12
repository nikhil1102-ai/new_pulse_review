"""
Phase 1c — Deduplicate reviews.

Two-tier deduplication strategy:
  1. Exact-match on ``review_id``
  2. Near-duplicate detection via MinHash LSH (Jaccard ≥ 0.90)

For each near-duplicate group, the earliest review (by date) is kept.
"""

import logging

from datasketch import MinHash, MinHashLSH

from src.config import NEAR_DUPLICATE_THRESHOLD
from src.state import PipelineState

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────
NUM_PERM = 128        # MinHash permutations
SHINGLE_K = 3         # word-level shingle size


def _word_shingles(text: str, k: int = SHINGLE_K) -> list[str]:
    """
    Generate word-level k-shingles from text.

    Example (k=3): "the app crashes often" → ["the app crashes", "app crashes often"]
    """
    words = text.split()
    if len(words) < k:
        # For very short texts, use single-word shingles
        return words if words else [""]
    return [" ".join(words[i : i + k]) for i in range(len(words) - k + 1)]


def _build_minhash(text: str) -> MinHash:
    """Build a MinHash signature from word-level shingles."""
    m = MinHash(num_perm=NUM_PERM)
    for shingle in _word_shingles(text):
        m.update(shingle.encode("utf-8"))
    return m


def deduplicate(state: PipelineState) -> dict:
    """
    LangGraph node: remove exact and near-duplicate reviews.

    Input:  state["cleaned_reviews"]  — partially cleaned from clean node
    Output: {"cleaned_reviews": list[dict]}  — deduplicated reviews

    Strategy:
      1. Exact dedup by review_id (set membership)
      2. Near-duplicate via MinHash LSH (Jaccard ≥ NEAR_DUPLICATE_THRESHOLD)
         → for each duplicate group, keep the earliest review (by date)
    """
    cleaned_reviews = state.get("cleaned_reviews", [])
    total_input = len(cleaned_reviews)

    # ── Step 1: Exact dedup by review_id ─────────────────────
    seen_ids: set[str] = set()
    exact_deduped: list[dict] = []
    exact_dupes_removed = 0

    for review in cleaned_reviews:
        rid = review.get("review_id", "")
        if rid in seen_ids:
            exact_dupes_removed += 1
            continue
        seen_ids.add(rid)
        exact_deduped.append(review)

    # ── Step 2: Near-duplicate via MinHash LSH ───────────────
    lsh = MinHashLSH(threshold=NEAR_DUPLICATE_THRESHOLD, num_perm=NUM_PERM)
    minhashes: dict[str, MinHash] = {}

    # Build MinHash for each review and insert into LSH
    for review in exact_deduped:
        rid = review["review_id"]
        text = review.get("cleaned_text", "")
        mh = _build_minhash(text)
        minhashes[rid] = mh
        try:
            lsh.insert(rid, mh)
        except ValueError:
            # Duplicate key in LSH (shouldn't happen after exact dedup)
            pass

    # Find near-duplicate groups
    # Track which reviews have already been assigned to a group
    assigned: set[str] = set()
    near_dupe_groups: list[list[str]] = []

    for review in exact_deduped:
        rid = review["review_id"]
        if rid in assigned:
            continue

        # Query LSH for similar reviews
        candidates = lsh.query(minhashes[rid])
        if len(candidates) > 1:
            # This is a near-duplicate group
            group = [c for c in candidates if c not in assigned]
            if len(group) > 1:
                near_dupe_groups.append(group)
                assigned.update(group)
        else:
            assigned.add(rid)

    # For each group, keep the earliest review (by date)
    reviews_by_id = {r["review_id"]: r for r in exact_deduped}
    ids_to_remove: set[str] = set()

    for group in near_dupe_groups:
        group_reviews = [reviews_by_id[rid] for rid in group if rid in reviews_by_id]
        # Sort by date ascending; keep the first (earliest)
        group_reviews.sort(key=lambda r: r.get("date", ""))
        # Remove all but the earliest
        for review in group_reviews[1:]:
            ids_to_remove.add(review["review_id"])

    near_dupes_merged = len(ids_to_remove)

    # Build final list (preserving original order)
    final_reviews = [r for r in exact_deduped if r["review_id"] not in ids_to_remove]

    # ── Logging ──────────────────────────────────────────────
    logger.info(
        "Deduplication complete: %d input → %d output "
        "(exact dupes: %d, near-dupe groups: %d, near-dupes merged: %d)",
        total_input,
        len(final_reviews),
        exact_dupes_removed,
        len(near_dupe_groups),
        near_dupes_merged,
    )

    return {"cleaned_reviews": final_reviews}
