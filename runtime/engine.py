"""Direct Ask and Teach orchestration for the research helper."""

from __future__ import annotations

import json
from typing import Any


# Character bounds deliberately avoid a tokenizer dependency; they cap request size,
# not exact model tokens.
MAX_EVIDENCE_CHARS_PER_SECTION = 6_000
MAX_GRAPH_CONTEXT_ITEMS = 12
MAX_GRAPH_CONTEXT_CHARS = 4_000
_TRUNCATION_MARKER = "\n… (middle omitted for context limit) …\n"


def _bounded_text(text: Any, limit: int) -> str:
    """Keep the beginning and end of oversized evidence deterministically."""
    value = text if isinstance(text, str) else str(text or "")
    if len(value) <= limit:
        return value
    available = limit - len(_TRUNCATION_MARKER)
    head = available // 2
    tail = available - head
    return f"{value[:head].rstrip()}{_TRUNCATION_MARKER}{value[-tail:].lstrip()}"


def _bounded_sections(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy retrieved parents with a fixed evidence budget for answer generation."""
    return [
        {
            **section,
            "page_content": _bounded_text(
                section.get("page_content", ""), MAX_EVIDENCE_CHARS_PER_SECTION
            ),
            "metadata": dict(section.get("metadata", {})),
        }
        for section in sections
    ]


def _bounded_graph_context(graph_context: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Retain the first graph relations that fit a small, deterministic budget."""
    bounded: list[dict[str, Any]] = []
    used_chars = 0
    for relation in graph_context[:MAX_GRAPH_CONTEXT_ITEMS]:
        relation_size = len(json.dumps(relation, ensure_ascii=False, default=str))
        if used_chars + relation_size > MAX_GRAPH_CONTEXT_CHARS:
            break
        bounded.append(relation)
        used_chars += relation_size
    return bounded


def _source_label(metadata: dict[str, Any]) -> str:
    source = metadata.get("source", "Unknown source")
    title = metadata.get("section", "Unknown section")
    start = metadata.get("page_start")
    end = metadata.get("page_end")
    if start and end:
        pages = f"p.{start}" if start == end else f"p.{start}–{end}"
        return f"[{source} — {title}, {pages}]"
    return f"[{source} — {title}]"


def build_sources(sections: list[dict[str, Any]]) -> list[str]:
    """Format the source metadata already stored on parent sections."""
    return list(
        dict.fromkeys(_source_label(section.get("metadata", {})) for section in sections)
    )


def collect_anchor_nodes(sections: list[dict[str, Any]]) -> list[str]:
    """Read the parent-section graph anchors without another retrieval layer."""
    anchors: list[str] = []
    for section in sections:
        raw_anchors = section.get("metadata", {}).get("anchor_nodes", [])
        if isinstance(raw_anchors, str):
            anchors.extend(anchor.strip() for anchor in raw_anchors.split(","))
        else:
            anchors.extend(raw_anchors)
    return [anchor for anchor in dict.fromkeys(anchors) if anchor]


class RuntimeEngine:
    """The application's small, readable Ask and Teach flow."""

    def __init__(self, llm: Any, db: Any, dag: Any) -> None:
        self.llm = llm
        self.db = db
        self.dag = dag

    def ask(self, query: str, target_file: str = "") -> dict[str, Any]:
        sections = self.db.search_candidates_and_fetch_parent(
            query=query,
            llm_service=self.llm,
            target_file=target_file,
        )
        if not sections:
            return {
                "answer": "No relevant source found.",
                "sources": [],
                "graph_context": [],
            }
        sections = _bounded_sections(sections)
        graph_context = _bounded_graph_context(
            self.dag.get_graph_context(
                collect_anchor_nodes(sections), search_mode="search"
            )
        )
        return {
            "answer": self.llm.answer(
                query=query,
                sections=sections,
                graph_context=graph_context,
            ),
            "sources": build_sources(sections),
            "graph_context": graph_context,
        }

    def teach_section(
        self, target_file: str, target_section: str
    ) -> list[dict[str, Any]]:
        sections = self.db.get_section_exact(target_file, target_section)
        if not sections:
            return []
        section = sections[0]
        roadmap = self.db.get_roadmap(section["metadata"]["parent_id"])
        lessons = []
        for step in roadmap:
            graph_context = self.dag.get_graph_context(
                step.get("concepts", []), search_mode="semi_search"
            )
            lessons.append(
                {
                    "step": step,
                    "content": self.llm.teach_step(
                        section_text=section["page_content"],
                        roadmap_step=step,
                        graph_context=graph_context,
                    ),
                }
            )
        return lessons
