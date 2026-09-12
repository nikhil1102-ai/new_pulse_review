"""
Phase 2b — Prepare batches for JINA embedding.

Groups chunks into batches of JINA_BATCH_SIZE (256), skipping any chunk
whose embedding is already cached on disk.
"""

import json
import logging
import os

from src.config import EMBEDDINGS_DIR, JINA_BATCH_SIZE
from src.state import PipelineState

logger = logging.getLogger(__name__)


def batch_prepare(state: PipelineState) -> dict:
    """
    LangGraph node: group chunks into batches for the JINA API.

    Input:  state["chunks"]  — from chunk node
    Output: state is updated with internal batch info (chunks passed through)

    Skips chunks whose chunk_id already has a cached embedding.
    """
    chunks = state.get("chunks", [])
    week_start = state.get("week_start", "unknown")

    # ── Check for cached embeddings ──────────────────────────
    metadata_file = os.path.join(EMBEDDINGS_DIR, f"metadata_{week_start}.json")
    cached_ids: set[str] = set()

    if os.path.exists(metadata_file):
        with open(metadata_file, "r", encoding="utf-8") as f:
            cached_meta = json.load(f)
        cached_ids = {item["chunk_id"] for item in cached_meta}
        logger.info("Found %d cached embeddings.", len(cached_ids))

    # ── Filter out already-cached chunks ─────────────────────
    new_chunks = [c for c in chunks if c["chunk_id"] not in cached_ids]
    skipped = len(chunks) - len(new_chunks)

    if skipped > 0:
        logger.info("Skipping %d already-cached chunks.", skipped)

    # ── Build batches ────────────────────────────────────────
    batches: list[list[dict]] = []
    for i in range(0, len(new_chunks), JINA_BATCH_SIZE):
        batches.append(new_chunks[i : i + JINA_BATCH_SIZE])

    logger.info(
        "Batch preparation: %d chunks → %d new → %d batches (size %d)",
        len(chunks),
        len(new_chunks),
        len(batches),
        JINA_BATCH_SIZE,
    )

    # Pass chunks and batch info through state
    # (all chunks are needed for metadata; batches are for the embed node)
    return {
        "chunks": chunks,  # preserve all chunks (including cached)
        "_batches": batches,  # internal: only new chunks, batched
        "_cached_ids": list(cached_ids),
    }
