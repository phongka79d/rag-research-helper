"""Run one Embeddings request against the configured provider."""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openai import OpenAI

from config.settings import Settings
from orchestrator.llm_service import _safe_provider_error


def main() -> int:
    settings = Settings()
    if not settings.OPENAI_API_KEY:
        print("Embeddings check failed: OPENAI_API_KEY is missing.", file=sys.stderr)
        return 1

    try:
        client = OpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL.rstrip("/"),
        )
        response = client.embeddings.create(
            model=settings.OPENAI_EMBEDDING_MODEL,
            input="OpenAI-compatible embedding compatibility check.",
        )
        vector = response.data[0].embedding
        if not isinstance(vector, list) or not vector:
            raise ValueError("Embeddings API did not return a non-empty vector.")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in vector
        ):
            raise ValueError("Embeddings API returned a non-numeric vector.")
    except Exception as error:
        print(
            "Embeddings check failed: "
            f"{_safe_provider_error(error, settings.OPENAI_API_KEY)}",
            file=sys.stderr,
        )
        return 1

    response_model = getattr(response, "model", "")
    served_model = (
        f", response_model={response_model}"
        if isinstance(response_model, str) and response_model
        else ""
    )
    print(
        "Embeddings check passed: "
        f"requested_model={settings.OPENAI_EMBEDDING_MODEL}, vector_size={len(vector)}"
        f"{served_model}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
