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

    text_model = str(settings.OPENAI_MODEL or "").strip()
    graph_model = str(getattr(settings, "OPENAI_GRAPH_MODEL", "") or "").strip()
    resolved_graph_model = graph_model or text_model
    # The fallback uses the same request once; a distinct configured graph model
    # gets its own compatibility check.
    models = list(dict.fromkeys([text_model, resolved_graph_model]))

    try:
        client = OpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL.rstrip("/"),
        )
        for model in models:
            response = client.responses.create(
                model=model,
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
                raise ValueError(
                    f"Responses API did not return a non-empty JSON object for {model}."
                )
            response_model = getattr(response, "model", "")
            served_model = (
                f", response_model={response_model}"
                if isinstance(response_model, str) and response_model
                else ""
            )
            role = "text" if model == settings.OPENAI_MODEL else "graph"
            print(
                "Responses check passed: "
                f"role={role}, requested_model={model}, json_keys={len(payload)}{served_model}"
            )
        print(
            "Responses model routing: "
            f"text_model={text_model}, graph_model={resolved_graph_model}, requests={len(models)}"
        )
    except Exception as error:
        print(
            "Responses check failed: "
            f"{_safe_provider_error(error, settings.OPENAI_API_KEY)}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
