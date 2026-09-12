"""
Phase 2d — Cache embedding vectors to disk.

Persists embeddings as NumPy ``.npy`` files and metadata as JSON,
so re-runs for the same week do not re-call the JINA API.
On each run, merges new embeddings with any previously cached ones.
"""

import json
import logging
import os

import numpy as np

from src.config import EMBEDDINGS_DIR, JINA_DIMENSIONS
from src.state import PipelineState

logger = logging.getLogger(__name__)


def cache_vectors(state: PipelineState) -> dict:
    """
    LangGraph node: persist embeddings to disk for idempotent re-runs.

    Input:  state["embeddings"], state["embedding_ids"], state["chunks"]
    Output: {} (side-effect: writes .npy and metadata JSON)

    On re-run, merges newly computed embeddings with any cached ones
    to produce a complete embedding matrix for downstream clustering.
    """
    week_start = state.get("week_start", "unknown")
    new_embeddings = state.get("embeddings", [])
    new_ids = state.get("embedding_ids", [])
    all_chunks = state.get("chunks", [])

    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    npy_file = os.path.join(EMBEDDINGS_DIR, f"embeddings_{week_start}.npy")
    meta_file = os.path.join(EMBEDDINGS_DIR, f"metadata_{week_start}.json")

    # ── Load cached embeddings if they exist ─────────────────
    cached_embeddings: list[list[float]] = []
    cached_meta: list[dict] = []

    if os.path.exists(npy_file) and os.path.exists(meta_file):
        cached_matrix = np.load(npy_file)
        cached_embeddings = cached_matrix.tolist()
        with open(meta_file, "r", encoding="utf-8") as f:
            cached_meta = json.load(f)
        logger.info("Loaded %d cached embeddings.", len(cached_meta))

    # ── Build lookup for new embeddings ──────────────────────
    new_embed_map: dict[str, list[float]] = {}
    for chunk_id, embedding in zip(new_ids, new_embeddings):
        new_embed_map[chunk_id] = embedding

    # ── Merge: cached + new ──────────────────────────────────
    # Build a complete set by chunk_id
    merged_map: dict[str, list[float]] = {}
    merged_meta_map: dict[str, dict] = {}

    # Add cached first
    for meta, emb in zip(cached_meta, cached_embeddings):
        cid = meta["chunk_id"]
        merged_map[cid] = emb
        merged_meta_map[cid] = meta

    # Add/overwrite with new
    chunk_lookup = {c["chunk_id"]: c for c in all_chunks}
    for chunk_id, embedding in new_embed_map.items():
        merged_map[chunk_id] = embedding
        chunk = chunk_lookup.get(chunk_id, {})
        merged_meta_map[chunk_id] = {
            "chunk_id": chunk_id,
            "review_id": chunk.get("review_id", ""),
            "chunk_index": chunk.get("chunk_index", 0),
            "chunk_text": chunk.get("chunk_text", ""),
            "total_chunks": chunk.get("total_chunks", 1),
        }

    # ── Build aligned arrays ─────────────────────────────────
    final_ids = sorted(merged_map.keys())
    final_embeddings = [merged_map[cid] for cid in final_ids]
    final_meta = [merged_meta_map[cid] for cid in final_ids]

    # ── Save to disk ─────────────────────────────────────────
    embedding_matrix = np.array(final_embeddings, dtype=np.float32)
    np.save(npy_file, embedding_matrix)

    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(final_meta, f, ensure_ascii=False, indent=2)

    logger.info(
        "Cached %d embeddings → %s (shape: %s)",
        len(final_embeddings),
        npy_file,
        embedding_matrix.shape,
    )

    # Update state with the complete set for downstream
    return {
        "embeddings": final_embeddings,
        "embedding_ids": final_ids,
    }
