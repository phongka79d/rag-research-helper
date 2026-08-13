"""Runtime settings loaded from the local .env file and process environment."""

from __future__ import annotations

import os
import math

from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Small, direct configuration holder for one OpenAI-compatible provider."""

    def __init__(self) -> None:
        self.OPENAI_BASE_URL = os.getenv(
            "OPENAI_BASE_URL", "https://api.shopaikey.com/v1"
        ).rstrip("/")
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
        self.OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.OPENAI_EMBEDDING_MODEL = os.getenv(
            "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
        )
        self.OPENAI_EMBEDDING_DIM = int(
            os.getenv("OPENAI_EMBEDDING_DIM", "1536")
        )
        self.QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
        self.NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
        self.JINA_API_KEY = os.getenv("JINA_API_KEY", "")
        self.JINA_RERANK_URL = (
            os.getenv("JINA_RERANK_URL", "https://api.jina.ai/v1/rerank")
            or "https://api.jina.ai/v1/rerank"
        ).rstrip("/")
        self.JINA_RERANK_MODEL = os.getenv(
            "JINA_RERANK_MODEL", "jina-reranker-v2-base-multilingual"
        )
        try:
            self.JINA_RPM = int(os.getenv("JINA_RPM", "100") or "100")
        except (TypeError, ValueError, OverflowError):
            self.JINA_RPM = 100
        if self.JINA_RPM < 1:
            self.JINA_RPM = 100
        try:
            self.QDRANT_SEARCH_LIMIT = int(os.getenv("QDRANT_SEARCH_LIMIT", "25") or "25")
        except (TypeError, ValueError, OverflowError):
            self.QDRANT_SEARCH_LIMIT = 25
        if self.QDRANT_SEARCH_LIMIT < 1:
            self.QDRANT_SEARCH_LIMIT = 25
        self.QDRANT_SEARCH_LIMIT = min(self.QDRANT_SEARCH_LIMIT, 25)
        try:
            self.QDRANT_MAX_CANDIDATE_PARENTS = int(
                os.getenv("QDRANT_MAX_CANDIDATE_PARENTS", "5") or "5"
            )
        except (TypeError, ValueError, OverflowError):
            self.QDRANT_MAX_CANDIDATE_PARENTS = 5
        if self.QDRANT_MAX_CANDIDATE_PARENTS < 1:
            self.QDRANT_MAX_CANDIDATE_PARENTS = 5
        self.QDRANT_MAX_CANDIDATE_PARENTS = min(
            self.QDRANT_MAX_CANDIDATE_PARENTS, 5
        )
        try:
            self.JINA_RERANK_MARGIN = float(
                os.getenv("JINA_RERANK_MARGIN", "0.08") or "0.08"
            )
        except (TypeError, ValueError, OverflowError):
            self.JINA_RERANK_MARGIN = 0.08
        if not math.isfinite(self.JINA_RERANK_MARGIN):
            self.JINA_RERANK_MARGIN = 0.08
        self.JINA_RERANK_MARGIN = max(0.0, min(self.JINA_RERANK_MARGIN, 1.0))

    def validate(self, require_openai: bool = True) -> None:
        if require_openai and not self.OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY is required in .env or the process environment."
            )
        if not self.NEO4J_PASSWORD:
            raise RuntimeError(
                "NEO4J_PASSWORD is required in .env. Run python setup_env.py --start."
            )
