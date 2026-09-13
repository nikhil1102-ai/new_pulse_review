"""
Central configuration for the Groww Weekly Review Pulse pipeline.

Loads environment variables via python-dotenv and exposes typed constants
used across all pipeline nodes.
"""

import os
from dotenv import load_dotenv

# Load .env from project root
load_dotenv()

# ──────────────────────────────────────────────
# API Keys
# ──────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
JINA_API_KEY: str = os.getenv("JINA_API_KEY", "")

# ──────────────────────────────────────────────
# Google Play Store (scraper — no credentials needed)
# ──────────────────────────────────────────────
GOOGLE_PLAY_PACKAGE_NAME: str = os.getenv(
    "GOOGLE_PLAY_PACKAGE_NAME", "com.nextbillion.groww"
)
GOOGLE_PLAY_COUNTRY: str = os.getenv("GOOGLE_PLAY_COUNTRY", "in")
GOOGLE_PLAY_LANGUAGE: str = os.getenv("GOOGLE_PLAY_LANGUAGE", "en")

# ──────────────────────────────────────────────
# REST Server (Google Docs & Gmail delivery via Railway FastAPI)
# ──────────────────────────────────────────────
# Base URL of the Railway FastAPI server — no /sse suffix required.
# Endpoints used: POST /append_to_doc, POST /create_email_draft
MCP_SERVER_URL: str = os.getenv("MCP_SERVER_URL", "")
MCP_SERVER_TIMEOUT: int = int(os.getenv("MCP_SERVER_TIMEOUT", "60"))
REPORT_RECIPIENTS: list[str] = [
    r.strip()
    for r in os.getenv("REPORT_RECIPIENTS", "").split(",")
    if r.strip()
]

# ──────────────────────────────────────────────
# Phase 1 — Ingestion
# ──────────────────────────────────────────────
REVIEW_WINDOW_WEEKS: int = int(os.getenv("REVIEW_WINDOW_WEEKS", "8"))
MIN_REVIEW_LENGTH: int = int(os.getenv("MIN_REVIEW_LENGTH", "10"))

# ──────────────────────────────────────────────
# Phase 1c — Deduplication
# ──────────────────────────────────────────────
NEAR_DUPLICATE_THRESHOLD: float = float(
    os.getenv("NEAR_DUPLICATE_THRESHOLD", "0.90")
)

# ──────────────────────────────────────────────
# Phase 2 — Embedding (JINA)
# ──────────────────────────────────────────────
JINA_API_URL: str = "https://api.jina.ai/v1/embeddings"
JINA_MODEL: str = os.getenv("JINA_MODEL", "jina-embeddings-v3")
JINA_DIMENSIONS: int = int(os.getenv("JINA_DIMENSIONS", "1024"))
JINA_BATCH_SIZE: int = int(os.getenv("JINA_BATCH_SIZE", "256"))
JINA_MAX_RETRIES: int = int(os.getenv("JINA_MAX_RETRIES", "3"))

# ──────────────────────────────────────────────
# Phase 3 — Clustering
# ──────────────────────────────────────────────
UMAP_N_COMPONENTS: int = int(os.getenv("UMAP_N_COMPONENTS", "15"))
UMAP_N_NEIGHBORS: int = int(os.getenv("UMAP_N_NEIGHBORS", "15"))
UMAP_MIN_DIST: float = float(os.getenv("UMAP_MIN_DIST", "0.1"))

HDBSCAN_MIN_CLUSTER_SIZE: int = int(os.getenv("HDBSCAN_MIN_CLUSTER_SIZE", "30"))
HDBSCAN_MIN_SAMPLES: int = int(os.getenv("HDBSCAN_MIN_SAMPLES", "5"))
HDBSCAN_RELAXED_MIN_CLUSTER_SIZE: int = int(
    os.getenv("HDBSCAN_RELAXED_MIN_CLUSTER_SIZE", "20")
)
KMEANS_FALLBACK_K: int = int(os.getenv("KMEANS_FALLBACK_K", "5"))
MIN_CLUSTERS: int = 3
MAX_CLUSTERS: int = int(os.getenv("MAX_CLUSTERS", "10"))
NOISE_THRESHOLD_PCT: float = 0.20
CLUSTER_SAMPLE_SIZE: int = 20  # reviews sampled per cluster for labelling
REPRESENTATIVE_QUOTES_COUNT: int = 3  # quotes per cluster

# ──────────────────────────────────────────────
# Phase 4 — Report Generation (OpenAI-compatible LLM)
# ──────────────────────────────────────────────
OPENAI_API_BASE: str = os.getenv("OPENAI_API_BASE", "https://api.groq.com/openai/v1")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "openai/gpt-oss-120b")
OPENAI_TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE", "0.3"))
OPENAI_MAX_TOKENS: int = int(os.getenv("OPENAI_MAX_TOKENS", "2048"))
MAX_REPORT_WORDS: int = int(os.getenv("MAX_REPORT_WORDS", "250"))

# ──────────────────────────────────────────────
# Phase 5 — Validation
# ──────────────────────────────────────────────
FUZZY_MATCH_THRESHOLD: int = int(os.getenv("FUZZY_MATCH_THRESHOLD", "85"))
MAX_VALIDATION_RETRIES: int = int(os.getenv("MAX_VALIDATION_RETRIES", "2"))

# ──────────────────────────────────────────────
# Phase 6b — Detailed PDF report
# ──────────────────────────────────────────────
PDF_ENABLED: bool = os.getenv("PDF_ENABLED", "true").lower() != "false"
PDF_MIME_TYPE: str = "application/pdf"

# ──────────────────────────────────────────────
# Data paths (relative to project root)
# ──────────────────────────────────────────────
DATA_DIR: str = os.getenv("DATA_DIR", "data")
RAW_DATA_DIR: str = os.path.join(DATA_DIR, "raw")
CLEAN_DATA_DIR: str = os.path.join(DATA_DIR, "clean")
EMBEDDINGS_DIR: str = os.path.join(DATA_DIR, "embeddings")
REPORTS_DIR: str = os.path.join(DATA_DIR, "reports")
