"""Runtime settings loaded from the local .env file."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Small, direct configuration holder for the application."""

    def __init__(self) -> None:
        self.OPENAI_BASE_URL = os.getenv(
            "OPENAI_BASE_URL", "https://api.openai.com/v1"
        ).rstrip("/")
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
        self.OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
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

    def validate(self, require_openai: bool = True) -> None:
        if require_openai and not self.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is required in .env.")
        if not self.NEO4J_PASSWORD:
            raise RuntimeError(
                "NEO4J_PASSWORD is required in .env. Run python setup_env.py --start."
            )
