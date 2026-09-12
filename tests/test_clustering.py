"""
Unit tests for Phase 4 — Clustering & Theme Extraction.

Tests cover:
  - cluster.py  — UMAP, HDBSCAN, K-Means fallback, noise handling,
                   chunk reassembly, representative quote selection
  - label_themes.py — OpenAI prompt construction, label assignment, error handling
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────


def _make_review(review_id: str, text: str, rating: int = 4) -> dict:
    """Create a minimal cleaned-review dict."""
    return {
        "review_id": review_id,
        "original_text": text,
        "cleaned_text": text.lower(),
        "rating": rating,
        "date": "2026-08-15",
        "author_name": "User",
        "thumbs_up_count": 0,
        "app_version": "1.0",
    }


def _make_chunk(review_id: str, chunk_index: int = 0, total_chunks: int = 1) -> dict:
    """Create a minimal chunk dict."""
    return {
        "chunk_id": f"{review_id}_chunk_{chunk_index}",
        "review_id": review_id,
        "chunk_index": chunk_index,
        "chunk_text": f"text for {review_id}",
        "total_chunks": total_chunks,
    }


def _make_clusterable_state(n_reviews: int = 100, n_dims: int = 1024) -> dict:
    """Build a state dict with synthetic embeddings forming known clusters.

    Creates ``n_reviews`` reviews split into 4 Gaussian clusters so that
    HDBSCAN / K-Means should reliably find ≥ 3 clusters.
    """
    rng = np.random.RandomState(42)
    reviews = []
    chunks = []
    embeddings = []

    cluster_centres = [
        rng.randn(n_dims).astype(np.float32) * 10,
        rng.randn(n_dims).astype(np.float32) * 10 + 50,
        rng.randn(n_dims).astype(np.float32) * 10 - 50,
        rng.randn(n_dims).astype(np.float32) * 10 + 100,
    ]
    per_cluster = n_reviews // len(cluster_centres)

    for ci, centre in enumerate(cluster_centres):
        for j in range(per_cluster):
            rid = f"review_{ci}_{j}"
            reviews.append(_make_review(rid, f"Review {ci} {j}"))
            chunks.append(_make_chunk(rid))
            # Tight Gaussian around centre
            vec = centre + rng.randn(n_dims).astype(np.float32) * 0.5
            embeddings.append(vec.tolist())

    embedding_ids = [c["chunk_id"] for c in chunks]

    return {
        "product": "Groww",
        "week_start": "2026-08-01",
        "week_end": "2026-08-07",
        "embeddings": embeddings,
        "embedding_ids": embedding_ids,
        "chunks": chunks,
        "cleaned_reviews": reviews,
    }


# ──────────────────────────────────────────────────────────────
# cluster.py tests
# ──────────────────────────────────────────────────────────────


class TestClusterNode:
    """Tests for ``src.nodes.cluster.cluster``."""

    def test_empty_embeddings_returns_empty(self):
        """Cluster node should return empty lists when no embeddings exist."""
        from src.nodes.cluster import cluster

        result = cluster(
            {
                "embeddings": [],
                "embedding_ids": [],
                "chunks": [],
                "cleaned_reviews": [],
            }
        )
        assert result["clusters"] == []
        assert result["noise_reviews"] == []

    def test_output_structure(self):
        """Each cluster dict must have label, review_ids, centroid_quotes."""
        from src.nodes.cluster import cluster

        state = _make_clusterable_state(n_reviews=120)
        result = cluster(state)

        assert isinstance(result["clusters"], list)
        assert isinstance(result["noise_reviews"], list)

        for cl in result["clusters"]:
            assert "label" in cl
            assert "review_ids" in cl
            assert "centroid_quotes" in cl
            assert isinstance(cl["review_ids"], list)
            assert isinstance(cl["centroid_quotes"], list)
            assert len(cl["review_ids"]) > 0

    def test_labels_are_none_before_labelling(self):
        """Cluster node should leave label=None for the label_themes node."""
        from src.nodes.cluster import cluster

        state = _make_clusterable_state(n_reviews=120)
        result = cluster(state)

        for cl in result["clusters"]:
            assert cl["label"] is None

    def test_all_reviews_accounted_for(self):
        """Every review_id should appear in exactly one cluster or noise."""
        from src.nodes.cluster import cluster

        state = _make_clusterable_state(n_reviews=120)
        result = cluster(state)

        all_assigned = set()
        for cl in result["clusters"]:
            for rid in cl["review_ids"]:
                assert rid not in all_assigned, f"{rid} in multiple clusters"
                all_assigned.add(rid)
        for nr in result["noise_reviews"]:
            rid = nr["review_id"]
            assert rid not in all_assigned, f"{rid} in cluster and noise"
            all_assigned.add(rid)

        expected_ids = {r["review_id"] for r in state["cleaned_reviews"]}
        assert all_assigned == expected_ids

    def test_representative_quotes_are_real_text(self):
        """Centroid quotes must come from original_text of actual reviews."""
        from src.nodes.cluster import cluster

        state = _make_clusterable_state(n_reviews=120)
        result = cluster(state)

        valid_texts = {r["original_text"] for r in state["cleaned_reviews"]}
        for cl in result["clusters"]:
            for quote in cl["centroid_quotes"]:
                assert quote in valid_texts, f"Quote not from corpus: {quote}"

    @patch("src.nodes.cluster._reduce_dimensions", side_effect=lambda x: x)
    @patch("src.nodes.cluster._run_hdbscan")
    def test_kmeans_fallback_on_few_clusters(self, mock_hdbscan, mock_umap):
        """When HDBSCAN returns < 3 clusters, K-Means fallback should fire."""
        from src.nodes.cluster import cluster

        state = _make_clusterable_state(n_reviews=60)
        n = len(state["embeddings"])

        # Make HDBSCAN return only 1 cluster (all label 0)
        mock_hdbscan.return_value = np.zeros(n, dtype=int)

        result = cluster(state)

        # K-Means should produce multiple clusters
        assert len(result["clusters"]) >= 3

    @patch("src.nodes.cluster._reduce_dimensions", side_effect=lambda x: x)
    @patch("src.nodes.cluster._run_hdbscan")
    def test_noise_recluster_when_above_threshold(self, mock_hdbscan, mock_umap):
        """When noise > 20%, the node should attempt relaxed re-clustering."""
        from src.nodes.cluster import cluster

        state = _make_clusterable_state(n_reviews=100)
        n = len(state["embeddings"])

        # First call: 3 clusters but 30% noise
        labels_high_noise = np.array(
            [0] * 25 + [1] * 25 + [2] * 20 + [-1] * 30
        )
        # Second call (relaxed): less noise
        labels_relaxed = np.array(
            [0] * 30 + [1] * 30 + [2] * 25 + [-1] * 15
        )
        mock_hdbscan.side_effect = [labels_high_noise, labels_relaxed]

        result = cluster(state)

        # Should have been called twice (initial + relaxed)
        assert mock_hdbscan.call_count == 2


class TestChunkReassembly:
    """Tests for ``_reassemble_chunks_to_reviews``."""

    def test_single_chunk_reviews(self):
        """Reviews with one chunk each are assigned directly."""
        from src.nodes.cluster import _reassemble_chunks_to_reviews

        labels = np.array([0, 1, 0, 1])
        metadata = [
            {"chunk_id": "r1_chunk_0", "review_id": "r1", "chunk_index": 0},
            {"chunk_id": "r2_chunk_0", "review_id": "r2", "chunk_index": 0},
            {"chunk_id": "r3_chunk_0", "review_id": "r3", "chunk_index": 0},
            {"chunk_id": "r4_chunk_0", "review_id": "r4", "chunk_index": 0},
        ]
        result = _reassemble_chunks_to_reviews(labels, metadata)

        assert set(result[0]) == {"r1", "r3"}
        assert set(result[1]) == {"r2", "r4"}

    def test_multi_chunk_majority_vote(self):
        """A review split into 3 chunks should go to the majority cluster."""
        from src.nodes.cluster import _reassemble_chunks_to_reviews

        # Review "r1" has 3 chunks: 2 in cluster 0, 1 in cluster 1
        labels = np.array([0, 0, 1])
        metadata = [
            {"chunk_id": "r1_chunk_0", "review_id": "r1", "chunk_index": 0},
            {"chunk_id": "r1_chunk_1", "review_id": "r1", "chunk_index": 1},
            {"chunk_id": "r1_chunk_2", "review_id": "r1", "chunk_index": 2},
        ]
        result = _reassemble_chunks_to_reviews(labels, metadata)

        assert "r1" in result[0]
        assert "r1" not in result.get(1, [])


class TestRepresentativeQuotes:
    """Tests for ``_select_representative_quotes``."""

    def test_returns_up_to_n_quotes(self):
        """Should return at most REPRESENTATIVE_QUOTES_COUNT quotes."""
        from src.nodes.cluster import _select_representative_quotes

        rng = np.random.RandomState(0)
        n = 10
        dim = 32
        review_ids = [f"r{i}" for i in range(n)]
        metadata = [
            {"chunk_id": f"r{i}_chunk_0", "review_id": f"r{i}", "chunk_index": 0}
            for i in range(n)
        ]
        embeddings = rng.randn(n, dim).astype(np.float32)
        lookup = {f"r{i}": _make_review(f"r{i}", f"Text {i}") for i in range(n)}

        quotes = _select_representative_quotes(
            review_ids, embeddings, metadata, lookup, n_quotes=3
        )
        assert len(quotes) <= 3
        assert all(isinstance(q, str) for q in quotes)

    def test_empty_cluster_returns_empty(self):
        """An empty cluster should yield no quotes."""
        from src.nodes.cluster import _select_representative_quotes

        quotes = _select_representative_quotes(
            [],
            np.array([]).reshape(0, 32),
            [],
            {},
        )
        assert quotes == []


# ──────────────────────────────────────────────────────────────
# label_themes.py tests
# ──────────────────────────────────────────────────────────────


class TestLabelThemesNode:
    """Tests for ``src.nodes.label_themes.label_themes``."""

    @patch("src.nodes.label_themes.ChatOpenAI")
    def test_labels_populated(self, MockChatOpenAI):
        """Each cluster should get a label string from OpenAI."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "App crashes during market hours"
        mock_llm.invoke.return_value = mock_response
        MockChatOpenAI.return_value = mock_llm

        from src.nodes.label_themes import label_themes

        state = {
            "clusters": [
                {
                    "label": None,
                    "review_ids": ["r1", "r2", "r3"],
                    "centroid_quotes": ["quote1"],
                },
                {
                    "label": None,
                    "review_ids": ["r4", "r5"],
                    "centroid_quotes": ["quote2"],
                },
            ],
            "cleaned_reviews": [
                _make_review("r1", "App crashes when market opens"),
                _make_review("r2", "Crashes every morning at 9:15"),
                _make_review("r3", "App freezes during trading"),
                _make_review("r4", "Withdrawal takes too long"),
                _make_review("r5", "Money stuck for 3 days"),
            ],
        }

        result = label_themes(state)

        assert len(result["clusters"]) == 2
        for cl in result["clusters"]:
            assert cl["label"] is not None
            assert isinstance(cl["label"], str)
            assert len(cl["label"]) > 0

    @patch("src.nodes.label_themes.ChatOpenAI")
    def test_openai_called_per_cluster(self, MockChatOpenAI):
        """OpenAI should be invoked once per cluster."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Theme label"
        mock_llm.invoke.return_value = mock_response
        MockChatOpenAI.return_value = mock_llm

        from src.nodes.label_themes import label_themes

        state = {
            "clusters": [
                {"label": None, "review_ids": [f"r{i}"], "centroid_quotes": []}
                for i in range(5)
            ],
            "cleaned_reviews": [
                _make_review(f"r{i}", f"Review text {i}") for i in range(5)
            ],
        }

        label_themes(state)

        assert mock_llm.invoke.call_count == 5

    @patch("src.nodes.label_themes.ChatOpenAI")
    def test_openai_failure_fallback_label(self, MockChatOpenAI):
        """On OpenAI error, cluster should get a fallback 'Cluster N' label."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("API error")
        MockChatOpenAI.return_value = mock_llm

        from src.nodes.label_themes import label_themes

        state = {
            "clusters": [
                {"label": None, "review_ids": ["r1"], "centroid_quotes": []},
            ],
            "cleaned_reviews": [_make_review("r1", "Some review")],
        }

        result = label_themes(state)

        assert result["clusters"][0]["label"] == "Cluster 1"

    def test_empty_clusters_returns_empty(self):
        """No clusters → no labelling, empty list returned."""
        from src.nodes.label_themes import label_themes

        result = label_themes({"clusters": [], "cleaned_reviews": []})
        assert result["clusters"] == []

    @patch("src.nodes.label_themes.ChatOpenAI")
    def test_label_stripped_of_quotes(self, MockChatOpenAI):
        """Labels wrapped in quotes by the LLM should be stripped."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '"Slow withdrawal processing"'
        mock_llm.invoke.return_value = mock_response
        MockChatOpenAI.return_value = mock_llm

        from src.nodes.label_themes import label_themes

        state = {
            "clusters": [
                {"label": None, "review_ids": ["r1"], "centroid_quotes": []},
            ],
            "cleaned_reviews": [_make_review("r1", "Withdrawal is slow")],
        }

        result = label_themes(state)

        label = result["clusters"][0]["label"]
        assert not label.startswith('"')
        assert not label.endswith('"')
        assert label == "Slow withdrawal processing"

    @patch("src.nodes.label_themes.ChatOpenAI")
    def test_prompt_contains_review_text(self, MockChatOpenAI):
        """The prompt sent to OpenAI must include actual review texts."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "App performance issues"
        mock_llm.invoke.return_value = mock_response
        MockChatOpenAI.return_value = mock_llm

        from src.nodes.label_themes import label_themes

        state = {
            "clusters": [
                {"label": None, "review_ids": ["r1", "r2"], "centroid_quotes": []},
            ],
            "cleaned_reviews": [
                _make_review("r1", "App is very slow"),
                _make_review("r2", "Performance is terrible"),
            ],
        }

        label_themes(state)

        # Inspect the messages passed to invoke
        call_args = mock_llm.invoke.call_args
        messages = call_args[0][0]
        user_msg = messages[1].content

        assert "App is very slow" in user_msg
        assert "Performance is terrible" in user_msg
