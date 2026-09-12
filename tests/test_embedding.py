"""
Tests for Phase 3 — chunk and embedding pipeline.

Covers: chunking strategy (whole/sentence/sliding), batch preparation,
JINA API mocking, caching, and end-to-end flow.
"""

import json
import os
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from src.nodes.chunk import (
    _count_tokens,
    _chunk_whole_review,
    _chunk_sliding_window,
    chunk,
)
from src.nodes.batch_prepare import batch_prepare
from src.nodes.embed import embed
from src.nodes.cache_vectors import cache_vectors


# ── Helpers ──────────────────────────────────────────────────

def _make_review(review_id="r1", text="this is a great app for trading"):
    return {
        "review_id": review_id,
        "original_text": text.title(),
        "cleaned_text": text,
        "rating": 5,
        "date": "2026-08-15",
        "author_name": "User",
        "thumbs_up_count": 0,
        "app_version": "5.0",
    }


# ══════════════════════════════════════════════════════════════
# Chunk tests
# ══════════════════════════════════════════════════════════════

class TestChunkNode:

    def test_whole_review_chunk(self):
        """Short reviews (≤512 tokens) produce a single chunk."""
        state = {"cleaned_reviews": [_make_review()]}
        result = chunk(state)

        assert len(result["chunks"]) == 1
        c = result["chunks"][0]
        assert c["chunk_id"] == "r1_chunk_0"
        assert c["review_id"] == "r1"
        assert c["chunk_index"] == 0
        assert c["total_chunks"] == 1
        assert c["chunk_text"] == "this is a great app for trading"

    def test_multiple_reviews(self):
        """Each review gets its own chunk."""
        state = {
            "cleaned_reviews": [
                _make_review(review_id=f"r{i}", text=f"review text number {i}")
                for i in range(5)
            ]
        }
        result = chunk(state)
        assert len(result["chunks"]) == 5
        ids = {c["review_id"] for c in result["chunks"]}
        assert ids == {"r0", "r1", "r2", "r3", "r4"}

    def test_empty_input(self):
        result = chunk({"cleaned_reviews": []})
        assert result["chunks"] == []

    def test_chunk_metadata_fields(self):
        """Each chunk carries all required metadata fields."""
        state = {"cleaned_reviews": [_make_review()]}
        result = chunk(state)
        c = result["chunks"][0]
        assert set(c.keys()) == {
            "chunk_id", "review_id", "chunk_index", "chunk_text", "total_chunks"
        }

    def test_sliding_window_creates_multiple_chunks(self):
        """Sliding window on long text creates overlapping chunks."""
        # Create a text longer than 512 tokens
        long_text = " ".join(["word"] * 600)
        chunks = _chunk_sliding_window(
            {"cleaned_text": long_text}, "long_review"
        )
        assert len(chunks) > 1
        for c in chunks:
            assert c["review_id"] == "long_review"
            assert c["total_chunks"] == len(chunks)


# ══════════════════════════════════════════════════════════════
# Batch prepare tests
# ══════════════════════════════════════════════════════════════

class TestBatchPrepare:

    def test_batches_chunks(self):
        """Chunks are grouped into batches."""
        chunks = [
            {"chunk_id": f"r{i}_chunk_0", "review_id": f"r{i}",
             "chunk_index": 0, "chunk_text": f"text {i}", "total_chunks": 1}
            for i in range(10)
        ]
        state = {"chunks": chunks, "week_start": "2026-07-16"}
        result = batch_prepare(state)

        assert "_batches" in result
        total_in_batches = sum(len(b) for b in result["_batches"])
        assert total_in_batches == 10

    def test_skips_cached_chunks(self, tmp_path):
        """Chunks with cached embeddings are skipped."""
        # Write a cached metadata file
        cached_meta = [{"chunk_id": "r0_chunk_0"}]
        meta_file = tmp_path / "metadata_2026-07-16.json"
        meta_file.write_text(json.dumps(cached_meta))

        chunks = [
            {"chunk_id": "r0_chunk_0", "chunk_text": "cached"},
            {"chunk_id": "r1_chunk_0", "chunk_text": "new"},
        ]
        state = {"chunks": chunks, "week_start": "2026-07-16"}

        with patch("src.nodes.batch_prepare.EMBEDDINGS_DIR", str(tmp_path)):
            result = batch_prepare(state)

        # Only r1 should be in batches
        batched_ids = [c["chunk_id"] for b in result["_batches"] for c in b]
        assert "r0_chunk_0" not in batched_ids
        assert "r1_chunk_0" in batched_ids

    def test_empty_input(self):
        state = {"chunks": [], "week_start": "2026-07-16"}
        result = batch_prepare(state)
        assert result["_batches"] == []


# ══════════════════════════════════════════════════════════════
# Embed tests
# ══════════════════════════════════════════════════════════════

class TestEmbed:

    @patch("src.nodes.embed._call_jina_api")
    def test_embeds_batches(self, mock_api):
        """Calls JINA API per batch and collects vectors."""
        mock_api.return_value = [[0.1] * 1024, [0.2] * 1024]

        batches = [
            [
                {"chunk_id": "r0_chunk_0", "chunk_text": "text 0"},
                {"chunk_id": "r1_chunk_0", "chunk_text": "text 1"},
            ]
        ]
        state = {"_batches": batches, "_cached_ids": []}
        result = embed(state)

        assert len(result["embeddings"]) == 2
        assert len(result["embedding_ids"]) == 2
        assert result["embedding_ids"] == ["r0_chunk_0", "r1_chunk_0"]

    def test_empty_batches(self):
        """No batches → no API calls, empty output."""
        state = {"_batches": [], "_cached_ids": []}
        result = embed(state)
        assert result["embeddings"] == []
        assert result["embedding_ids"] == []

    @patch("src.nodes.embed._call_jina_api")
    def test_multiple_batches(self, mock_api):
        """Multiple batches are processed sequentially."""
        mock_api.side_effect = [
            [[0.1] * 1024],
            [[0.2] * 1024],
        ]
        batches = [
            [{"chunk_id": "r0_chunk_0", "chunk_text": "a"}],
            [{"chunk_id": "r1_chunk_0", "chunk_text": "b"}],
        ]
        state = {"_batches": batches, "_cached_ids": []}

        with patch("src.nodes.embed.time.sleep"):
            result = embed(state)

        assert len(result["embeddings"]) == 2
        assert mock_api.call_count == 2


# ══════════════════════════════════════════════════════════════
# Cache vectors tests
# ══════════════════════════════════════════════════════════════

class TestCacheVectors:

    def test_saves_to_disk(self, tmp_path):
        """Embeddings and metadata are persisted."""
        state = {
            "week_start": "2026-07-16",
            "embeddings": [[0.1] * 1024, [0.2] * 1024],
            "embedding_ids": ["r0_chunk_0", "r1_chunk_0"],
            "chunks": [
                {"chunk_id": "r0_chunk_0", "review_id": "r0",
                 "chunk_index": 0, "chunk_text": "text 0", "total_chunks": 1},
                {"chunk_id": "r1_chunk_0", "review_id": "r1",
                 "chunk_index": 0, "chunk_text": "text 1", "total_chunks": 1},
            ],
        }
        with patch("src.nodes.cache_vectors.EMBEDDINGS_DIR", str(tmp_path)):
            result = cache_vectors(state)

        npy_file = tmp_path / "embeddings_2026-07-16.npy"
        meta_file = tmp_path / "metadata_2026-07-16.json"
        assert npy_file.exists()
        assert meta_file.exists()

        matrix = np.load(npy_file)
        assert matrix.shape == (2, 1024)

        meta = json.loads(meta_file.read_text())
        assert len(meta) == 2

    def test_merges_with_cached(self, tmp_path):
        """New embeddings are merged with previously cached ones."""
        # Pre-cache one embedding
        np.save(tmp_path / "embeddings_2026-07-16.npy",
                np.array([[0.5] * 1024], dtype=np.float32))
        meta = [{"chunk_id": "old_chunk_0", "review_id": "old",
                 "chunk_index": 0, "chunk_text": "old text", "total_chunks": 1}]
        (tmp_path / "metadata_2026-07-16.json").write_text(json.dumps(meta))

        state = {
            "week_start": "2026-07-16",
            "embeddings": [[0.9] * 1024],
            "embedding_ids": ["new_chunk_0"],
            "chunks": [
                {"chunk_id": "new_chunk_0", "review_id": "new",
                 "chunk_index": 0, "chunk_text": "new text", "total_chunks": 1},
            ],
        }
        with patch("src.nodes.cache_vectors.EMBEDDINGS_DIR", str(tmp_path)):
            result = cache_vectors(state)

        # Should have 2 embeddings total (old + new)
        assert len(result["embeddings"]) == 2
        assert len(result["embedding_ids"]) == 2

        matrix = np.load(tmp_path / "embeddings_2026-07-16.npy")
        assert matrix.shape == (2, 1024)

    def test_empty_embeddings(self, tmp_path):
        """Empty embeddings produce empty files."""
        state = {
            "week_start": "2026-07-16",
            "embeddings": [],
            "embedding_ids": [],
            "chunks": [],
        }
        with patch("src.nodes.cache_vectors.EMBEDDINGS_DIR", str(tmp_path)):
            result = cache_vectors(state)

        assert result["embeddings"] == []
