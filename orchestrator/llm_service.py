"""Direct official OpenAI Responses and Embeddings API calls."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from openai import OpenAI, OpenAIError

from core.schemas import HypotheticalQA, SectionAOTResult

logger = logging.getLogger(__name__)
JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


class LLMService:
    """The application's single direct OpenAI-compatible API client."""

    def __init__(self, settings: Any) -> None:
        self.client = OpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL.rstrip("/"),
        )
        self.model = settings.OPENAI_MODEL
        self.embedding_model = settings.OPENAI_EMBEDDING_MODEL

    def _chat(
        self,
        system: str,
        user: str,
        max_output_tokens: int | None = None,
        json_output: bool = False,
    ) -> str:
        request: dict[str, Any] = {
            "model": self.model,
            "input": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if max_output_tokens is not None:
            request["max_output_tokens"] = max_output_tokens
        if json_output:
            request["reasoning"] = {"effort": "minimal"}
            request["text"] = {"format": {"type": "json_object"}}
        try:
            response = self.client.responses.create(**request)
        except OpenAIError as error:
            raise RuntimeError(f"OpenAI Responses API request failed: {error}") from error
        content = response.output_text
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("OpenAI Responses API returned an empty completion.")
        return content.strip()

    @staticmethod
    def _json_object(text: str) -> dict[str, Any]:
        fenced = JSON_FENCE.search(text)
        candidate = fenced.group(1) if fenced else text
        start = candidate.find("{")
        if start < 0:
            raise RuntimeError("LLM response did not contain the required JSON object.")
        try:
            value, _ = json.JSONDecoder().raw_decode(candidate[start:])
        except json.JSONDecodeError as error:
            raise RuntimeError("LLM response did not contain valid JSON.") from error
        if not isinstance(value, dict):
            raise RuntimeError("LLM response must be a JSON object.")
        return value

    def embed(self, text: str) -> list[float]:
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model, input=text
            )
            embedding = response.data[0].embedding
        except (OpenAIError, AttributeError, IndexError, TypeError) as error:
            raise RuntimeError("OpenAI Embeddings API returned no embedding vector.") from error
        if not embedding:
            raise RuntimeError("OpenAI Embeddings API returned an empty embedding.")
        return [float(value) for value in embedding]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model, input=texts
            )
            vectors = [item.embedding for item in response.data]
        except (OpenAIError, AttributeError, TypeError) as error:
            raise RuntimeError("OpenAI Embeddings API returned no embedding vectors.") from error
        if len(vectors) != len(texts) or any(not vector for vector in vectors):
            raise RuntimeError("OpenAI Embeddings API returned incomplete embeddings.")
        return [[float(value) for value in vector] for vector in vectors]

    def extract_section_plan_and_graph(
        self, section_text: str, existing_nodes: list[str]
    ) -> dict[str, Any]:
        system = (
            "You are a research-paper extraction API. Return only one valid JSON object. "
            "Do not add markdown or commentary."
        )
        user = f"""
Analyze this research-paper section and return exactly this JSON shape:
{{
  "main_entities": ["concept"],
  "learning_roadmap": [
    {{"title": "step title", "content_focus": "what to explain", "concepts": ["concept"]}}
  ],
  "knowledge_graph": {{
    "nodes": [{{"name": "concept", "description": "one-sentence description"}}],
    "edges": [{{"source": "concept", "relation": "PREREQUISITE_OF", "target": "concept"}}]
  }}
}}

Rules:
- Produce 2 to 4 broad learning-roadmap steps when the section has enough material.
- Use only PREREQUISITE_OF, RELATES_TO, PART_OF, or DESCRIBES for relations.
- Reuse exact names from the existing concept list whenever the same or synonymous concept appears.
- Ground every field in the source text.

Existing concept names:
{json.dumps(existing_nodes, ensure_ascii=False)}

Source section:
{section_text}
""".strip()
        result = SectionAOTResult.model_validate(
            self._json_object(self._chat(system, user, json_output=True))
        )
        return result.model_dump()

    def generate_hypothetical_questions(
        self, section_text: str, num_questions: int = 5
    ) -> list[dict[str, str]]:
        system = (
            "You are a research-paper extraction API. Return only one valid JSON object. "
            "Every question and answer must be grounded only in the provided source text."
        )
        user = f"""
Generate exactly {num_questions} hypothetical user questions that this raw research-paper
section can answer. Include factual questions when the source contains values, methods,
or experimental details. Return:
{{
  "qa_pairs": [
    {{"question": "...", "key_knowledge": "one or two grounded sentences"}}
  ]
}}

Raw source section:
{section_text}
""".strip()
        payload = self._json_object(self._chat(system, user, json_output=True))
        pairs = [HypotheticalQA.model_validate(item) for item in payload.get("qa_pairs", [])]
        if len(pairs) != num_questions:
            raise RuntimeError(
                f"LLM returned {len(pairs)} hypothetical questions; expected {num_questions}."
            )
        return [pair.model_dump() for pair in pairs]

    def rerank_candidate_questions(
        self,
        user_query: str,
        candidates: list[dict[str, str]],
        limit: int = 2,
    ) -> list[str]:
        if not candidates:
            return []
        limit = max(1, min(limit, len(candidates)))
        system = "You select the best matching research questions. Return only valid JSON."
        user = f"""
User query:
{user_query}

Candidate questions:
{json.dumps(candidates, ensure_ascii=False)}

Return {{"best_parent_ids": ["up to {limit} parent IDs"]}}. Select only IDs present in the
candidate list, preserve relevance order, and do not repeat an ID.
""".strip()
        allowed = {candidate.get("parent_id", "") for candidate in candidates}
        try:
            selected = self._json_object(
                self._chat(
                    system,
                    user,
                    max_output_tokens=512,
                    json_output=True,
                )
            ).get("best_parent_ids", [])
        except RuntimeError:
            logger.warning("LLM rerank response was invalid; using the top vector candidate.")
            selected = []
        ordered = [
            parent_id
            for parent_id in selected
            if isinstance(parent_id, str) and parent_id in allowed
        ]
        if not ordered:
            ordered = [candidate["parent_id"] for candidate in candidates if candidate.get("parent_id")]
        return list(dict.fromkeys(ordered))[:limit]

    def answer(
        self,
        query: str,
        sections: list[dict[str, Any]],
        graph_context: list[dict[str, Any]],
    ) -> str:
        sources = []
        for section in sections:
            metadata = section.get("metadata", {})
            label = (
                f"{metadata.get('source', 'Unknown source')} — "
                f"{metadata.get('section', 'Unknown section')}"
            )
            sources.append(f"[{label}]\n{section.get('page_content', '')}")
        system = (
            "You are a precise research-paper assistant. Answer only from the provided paper "
            "sections and graph relationships. State when the evidence is insufficient. Cite "
            "claims inline with [source — section]. If the query asks for comparison, compare "
            "the evidence directly."
        )
        user = f"""
Question:
{query}

Paper sections:
{chr(10).join(sources)}

Concept-graph context:
{json.dumps(graph_context, ensure_ascii=False)}
""".strip()
        return self._chat(system, user)

    def teach_step(
        self,
        section_text: str,
        roadmap_step: dict[str, Any],
        graph_context: list[dict[str, Any]],
    ) -> str:
        system = (
            "You are a research-paper mentor. Teach the requested roadmap step clearly and "
            "accurately from the original section. Explain necessary prerequisites from the graph "
            "context, but do not invent material outside the evidence."
        )
        user = f"""
Roadmap step:
{json.dumps(roadmap_step, ensure_ascii=False)}

Original section:
{section_text}

Prerequisite and concept context:
{json.dumps(graph_context, ensure_ascii=False)}
""".strip()
        return self._chat(system, user)
