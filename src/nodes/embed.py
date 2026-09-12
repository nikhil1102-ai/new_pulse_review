"""
Phase 2c — Call JINA Embeddings API.

Sends batches of chunk texts to the JINA API and collects 1024-dim vectors.
Implements exponential backoff retry (max 3 retries).
"""

import logging
import time

import requests

from src.config import (
    JINA_API_KEY,
    JINA_API_URL,
    JINA_DIMENSIONS,
    JINA_MAX_RETRIES,
    JINA_MODEL,
)
from src.state import PipelineState

logger = logging.getLogger(__name__)


def _call_jina_api(texts: list[str]) -> list[list[float]]:
    """
    Call the JINA Embeddings API for a batch of texts.

    Returns a list of embedding vectors aligned 1:1 with the input texts.
    """
    headers = {
        "Authorization": f"Bearer {JINA_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": JINA_MODEL,
        "input": texts,
        "dimensions": JINA_DIMENSIONS,
    }

    for attempt in range(1, JINA_MAX_RETRIES + 1):
        try:
            response = requests.post(
                JINA_API_URL, json=payload, headers=headers, timeout=60
            )
            response.raise_for_status()
            data = response.json()

            # Extract embedding vectors, sorted by index
            embeddings_data = sorted(data["data"], key=lambda x: x["index"])
            return [item["embedding"] for item in embeddings_data]

        except Exception as exc:
            logger.warning(
                "JINA API attempt %d/%d failed: %s",
                attempt,
                JINA_MAX_RETRIES,
                exc,
            )
            if attempt == JINA_MAX_RETRIES:
                raise RuntimeError(
                    f"JINA API failed after {JINA_MAX_RETRIES} retries: {exc}"
                ) from exc
            time.sleep(2**attempt)

    return []  # unreachable, but keeps type checker happy


def embed(state: PipelineState) -> dict:
    """
    LangGraph node: generate embeddings for review chunks via JINA API.

    Input:  state["_batches"] — batched chunks from batch_prepare
            state["_cached_ids"] — already-embedded chunk IDs
            state["chunks"] — all chunks (for metadata alignment)
    Output: {"embeddings": list[list[float]], "embedding_ids": list[str]}
    """
    batches = state.get("_batches", [])
    cached_ids = set(state.get("_cached_ids", []))

    all_embeddings: list[list[float]] = []
    all_ids: list[str] = []

    if not batches:
        logger.info("No new chunks to embed (all cached or empty).")
        return {"embeddings": [], "embedding_ids": []}

    total_chunks = sum(len(b) for b in batches)
    logger.info("Embedding %d new chunks across %d batches …", total_chunks, len(batches))

    for batch_idx, batch in enumerate(batches):
        texts = [chunk["chunk_text"] for chunk in batch]
        ids = [chunk["chunk_id"] for chunk in batch]

        logger.info(
            "  Batch %d/%d: %d chunks", batch_idx + 1, len(batches), len(texts)
        )

        vectors = _call_jina_api(texts)

        all_embeddings.extend(vectors)
        all_ids.extend(ids)

        # Rate limiting between batches
        if batch_idx < len(batches) - 1:
            time.sleep(0.5)

    logger.info("Embedding complete: %d vectors generated.", len(all_embeddings))

    return {"embeddings": all_embeddings, "embedding_ids": all_ids}
