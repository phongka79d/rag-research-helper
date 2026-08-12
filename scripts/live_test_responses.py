"""Run one structured Responses request against the configured provider."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openai import OpenAI

from config.settings import Settings
from orchestrator.llm_service import _safe_provider_error


def main() -> int:
    settings = Settings()
    if not settings.OPENAI_API_KEY:
        print("Responses check failed: OPENAI_API_KEY is missing.", file=sys.stderr)
        return 1

    try:
        client = OpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL.rstrip("/"),
        )
        response = client.responses.create(
            model=settings.OPENAI_MODEL,
            input=[
                {
                    "role": "system",
                    "content": "Return only one valid JSON object with a status field.",
                },
                {"role": "user", "content": 'Set "status" to "ok".'},
            ],
            max_output_tokens=64,
            text={"format": {"type": "json_object"}},
        )
        payload: Any = json.loads(response.output_text)
        if not isinstance(payload, dict) or not payload:
            raise ValueError("Responses API did not return a non-empty JSON object.")
    except Exception as error:
        print(
            "Responses check failed: "
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
        "Responses check passed: "
        f"requested_model={settings.OPENAI_MODEL}, json_keys={len(payload)}{served_model}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
