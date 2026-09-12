"""
Phase 3 — Cluster embeddings using UMAP + HDBSCAN.

Reduces dimensionality via UMAP, clusters with HDBSCAN (falling back to
K-Means if fewer than 3 clusters are found), handles noisy/outlier reviews,
reassembles chunks back to parent ``review_id``s, and selects representative
quotes closest to each cluster centroid.
"""

import logging
from collections import defaultdict

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.config import (
    HDBSCAN_MIN_CLUSTER_SIZE,
    HDBSCAN_MIN_SAMPLES,
    HDBSCAN_RELAXED_MIN_CLUSTER_SIZE,
    KMEANS_FALLBACK_K,
    MAX_CLUSTERS,
    MIN_CLUSTERS,
    NOISE_THRESHOLD_PCT,
    REPRESENTATIVE_QUOTES_COUNT,
)
from src.state import PipelineState

logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────


def _reduce_dimensions(embedding_matrix: np.ndarray) -> np.ndarray:
    """Apply UMAP dimensionality reduction.

    Imported lazily so that ``umap`` is only required when this node runs.
    Falls back to returning the original matrix if the dataset is too small
    for the configured UMAP parameters.
    """
    import umap

    from src.config import UMAP_MIN_DIST, UMAP_N_COMPONENTS, UMAP_N_NEIGHBORS

    n_samples = embedding_matrix.shape[0]

    # UMAP requires n_samples > n_neighbors
    n_neighbors = min(UMAP_N_NEIGHBORS, n_samples - 1)
    if n_neighbors < 2:
        logger.warning(
            "Too few samples (%d) for UMAP — skipping dimensionality reduction.",
            n_samples,
        )
        return embedding_matrix

    n_components = min(UMAP_N_COMPONENTS, n_samples - 1)

    logger.info(
        "UMAP: reducing %d dims → %d (n_neighbors=%d, min_dist=%.2f)",
        embedding_matrix.shape[1],
        n_components,
        n_neighbors,
        UMAP_MIN_DIST,
    )

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=UMAP_MIN_DIST,
        random_state=42,
        metric="cosine",
    )
    return reducer.fit_transform(embedding_matrix)


def _run_hdbscan(
    reduced: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
) -> np.ndarray:
    """Run HDBSCAN and return cluster labels (noise = -1)."""
    import hdbscan

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(reduced)
    return labels


def _run_kmeans(reduced: np.ndarray, k: int) -> np.ndarray:
    """Fallback: K-Means clustering."""
    from sklearn.cluster import KMeans

    logger.info("K-Means fallback: k=%d", k)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    return kmeans.fit_predict(reduced)


def _cluster_embeddings(reduced: np.ndarray) -> np.ndarray:
    """Orchestrate HDBSCAN → optional re-cluster → K-Means fallback.

    Returns an array of integer labels (one per sample). Label ``-1``
    denotes noise/outlier points.
    """
    labels = _run_hdbscan(reduced, HDBSCAN_MIN_CLUSTER_SIZE, HDBSCAN_MIN_SAMPLES)
    n_clusters = len(set(labels) - {-1})
    n_noise = int(np.sum(labels == -1))
    total = len(labels)

    logger.info(
        "HDBSCAN (min_cluster_size=%d): %d clusters, %d noise (%.1f%%)",
        HDBSCAN_MIN_CLUSTER_SIZE,
        n_clusters,
        n_noise,
        100 * n_noise / max(total, 1),
    )

    # ── Noise re-clustering with relaxed params ──────────────
    if n_clusters >= MIN_CLUSTERS and n_noise / max(total, 1) > NOISE_THRESHOLD_PCT:
        logger.info(
            "Noise exceeds %.0f%% — re-clustering with relaxed min_cluster_size=%d",
            NOISE_THRESHOLD_PCT * 100,
            HDBSCAN_RELAXED_MIN_CLUSTER_SIZE,
        )
        labels = _run_hdbscan(
            reduced, HDBSCAN_RELAXED_MIN_CLUSTER_SIZE, HDBSCAN_MIN_SAMPLES
        )
        n_clusters = len(set(labels) - {-1})
        n_noise = int(np.sum(labels == -1))
        logger.info(
            "Re-cluster result: %d clusters, %d noise (%.1f%%)",
            n_clusters,
            n_noise,
            100 * n_noise / max(total, 1),
        )

    # ── K-Means fallback if still too few clusters ───────────
    if n_clusters < MIN_CLUSTERS:
        logger.warning(
            "HDBSCAN produced %d clusters (< %d minimum). Falling back to K-Means.",
            n_clusters,
            MIN_CLUSTERS,
        )
        labels = _run_kmeans(reduced, KMEANS_FALLBACK_K)
        # K-Means does not produce noise points
        n_clusters = len(set(labels))
        logger.info("K-Means result: %d clusters, 0 noise.", n_clusters)

    return labels


def _reassemble_chunks_to_reviews(
    labels: np.ndarray,
    metadata: list[dict],
) -> dict[int, list[str]]:
    """Map cluster labels back to unique ``review_id`` values.

    Each chunk inherits a cluster label from HDBSCAN/K-Means. When a review
    was split into multiple chunks, the review is assigned to the cluster
    that contains the **majority** of its chunks (majority-vote).

    Returns ``{cluster_label: [review_id, ...]}``.
    """
    # Step 1: collect per-review cluster votes
    review_votes: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for idx, meta in enumerate(metadata):
        review_id = meta["review_id"]
        label = int(labels[idx])
        review_votes[review_id][label] += 1

    # Step 2: assign each review to its majority cluster
    cluster_to_reviews: dict[int, list[str]] = defaultdict(list)
    for review_id, votes in review_votes.items():
        majority_label = max(votes, key=votes.get)  # type: ignore[arg-type]
        cluster_to_reviews[majority_label].append(review_id)

    return dict(cluster_to_reviews)


def _select_representative_quotes(
    cluster_review_ids: list[str],
    embedding_matrix: np.ndarray,
    metadata: list[dict],
    cleaned_reviews_lookup: dict[str, dict],
    n_quotes: int = REPRESENTATIVE_QUOTES_COUNT,
) -> list[str]:
    """Pick the *n_quotes* reviews whose embeddings are closest to the
    cluster centroid (cosine similarity).

    Uses ``original_text`` for the quote text so that casing and
    punctuation are preserved.
    """
    # Build index of chunk_id → position in embedding matrix
    id_to_idx: dict[str, int] = {}
    for idx, meta in enumerate(metadata):
        id_to_idx[meta["chunk_id"]] = idx

    # Collect embeddings for reviews in this cluster (use first chunk per review)
    review_indices: list[tuple[str, int]] = []
    for meta_idx, meta in enumerate(metadata):
        if (
            meta["review_id"] in cluster_review_ids
            and meta.get("chunk_index", 0) == 0
            and meta["review_id"] not in {r for r, _ in review_indices}
        ):
            review_indices.append((meta["review_id"], meta_idx))

    if not review_indices:
        return []

    # Compute cluster centroid from all chunk embeddings belonging to this cluster
    cluster_chunk_indices = [
        idx
        for idx, meta in enumerate(metadata)
        if meta["review_id"] in set(cluster_review_ids)
    ]
    cluster_embeddings = embedding_matrix[cluster_chunk_indices]
    centroid = cluster_embeddings.mean(axis=0, keepdims=True)

    # Rank review-level embeddings by cosine similarity to centroid
    review_emb_indices = [idx for _, idx in review_indices]
    review_embeddings = embedding_matrix[review_emb_indices]
    similarities = cosine_similarity(review_embeddings, centroid).flatten()

    top_k = min(n_quotes, len(similarities))
    top_indices = np.argsort(similarities)[::-1][:top_k]

    quotes: list[str] = []
    for ti in top_indices:
        review_id = review_indices[ti][0]
        review = cleaned_reviews_lookup.get(review_id, {})
        quote = review.get("original_text", review.get("cleaned_text", ""))
        if quote:
            quotes.append(quote)

    return quotes


# ── Main node ────────────────────────────────────────────────


def cluster(state: PipelineState) -> dict:
    """LangGraph node: cluster review embeddings into themes.

    **Input**  (from state):
        - ``embeddings``       — flat list of embedding vectors
        - ``embedding_ids``    — aligned chunk IDs
        - ``chunks``           — chunk metadata dicts
        - ``cleaned_reviews``  — full review records (for quote text)

    **Output**::

        {
            "clusters": [
                {
                    "label": None,
                    "review_ids": [...],
                    "centroid_quotes": [...]
                },
                ...
            ],
            "noise_reviews": [...]
        }
    """
    embeddings_raw = state.get("embeddings", [])
    embedding_ids = state.get("embedding_ids", [])
    chunks = state.get("chunks", [])
    cleaned_reviews = state.get("cleaned_reviews", [])

    if not embeddings_raw:
        logger.warning("No embeddings available — skipping clustering.")
        return {"clusters": [], "noise_reviews": []}

    # ── Build numpy matrix ───────────────────────────────────
    embedding_matrix = np.array(embeddings_raw, dtype=np.float32)
    logger.info("Clustering %d embeddings (dim=%d).", *embedding_matrix.shape)

    # ── Build metadata aligned with embedding matrix ─────────
    # The embedding_ids from cache_vectors are sorted; we need metadata
    # in the same order.
    chunk_lookup = {c["chunk_id"]: c for c in chunks}
    metadata = [chunk_lookup.get(cid, {"chunk_id": cid, "review_id": cid.rsplit("_chunk_", 1)[0], "chunk_index": 0}) for cid in embedding_ids]

    # ── Dimensionality reduction ─────────────────────────────
    reduced = _reduce_dimensions(embedding_matrix)

    # ── Clustering ───────────────────────────────────────────
    labels = _cluster_embeddings(reduced)

    # ── Reassemble chunks → reviews ──────────────────────────
    cluster_to_reviews = _reassemble_chunks_to_reviews(labels, metadata)

    # ── Build review lookup for quote selection ──────────────
    cleaned_reviews_lookup = {r["review_id"]: r for r in cleaned_reviews}

    # ── Build output clusters ────────────────────────────────
    result_clusters: list[dict] = []
    noise_reviews: list[dict] = []

    for cluster_label in sorted(cluster_to_reviews.keys()):
        review_ids = cluster_to_reviews[cluster_label]

        if cluster_label == -1:
            # Noise / outlier reviews
            noise_reviews = [
                cleaned_reviews_lookup[rid]
                for rid in review_ids
                if rid in cleaned_reviews_lookup
            ]
            logger.info("Noise bucket: %d reviews.", len(noise_reviews))
            continue

        # Select representative quotes
        centroid_quotes = _select_representative_quotes(
            review_ids,
            embedding_matrix,
            metadata,
            cleaned_reviews_lookup,
        )

        result_clusters.append(
            {
                "label": None,  # populated by label_themes node
                "review_ids": review_ids,
                "centroid_quotes": centroid_quotes,
            }
        )

    logger.info(
        "Clustering complete: %d themes, %d noise reviews.",
        len(result_clusters),
        len(noise_reviews),
    )

    # ── Cap to MAX_CLUSTERS (≤5) by merging smallest ─────────
    if len(result_clusters) > MAX_CLUSTERS:
        logger.info(
            "Merging %d clusters down to %d (MAX_CLUSTERS).",
            len(result_clusters),
            MAX_CLUSTERS,
        )
        # Sort ascending by review count — merge smallest into "Other Themes"
        result_clusters.sort(key=lambda c: len(c.get("review_ids", [])), reverse=True)
        top_clusters = result_clusters[:MAX_CLUSTERS - 1]
        merged_ids: list[str] = []
        merged_quotes: list[str] = []
        for excess in result_clusters[MAX_CLUSTERS - 1:]:
            merged_ids.extend(excess.get("review_ids", []))
            merged_quotes.extend(excess.get("centroid_quotes", [])[:1])
        top_clusters.append({
            "label": None,
            "review_ids": merged_ids,
            "centroid_quotes": merged_quotes[:REPRESENTATIVE_QUOTES_COUNT],
        })
        result_clusters = top_clusters
        logger.info("After merge: %d clusters.", len(result_clusters))

    # ── Sort by review count (descending) and mark top 3 ─────
    result_clusters.sort(
        key=lambda c: len(c.get("review_ids", [])), reverse=True
    )
    for idx, cl in enumerate(result_clusters):
        cl["is_top_3"] = idx < 3
        cl["rank"] = idx + 1

    logger.info(
        "Top 3 themes: %s",
        [cl.get('label', f'Cluster {cl["rank"]}') for cl in result_clusters[:3]],
    )

    return {"clusters": result_clusters, "noise_reviews": noise_reviews}
