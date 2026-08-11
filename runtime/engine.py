"""Direct Ask and Teach orchestration for the research helper."""

from __future__ import annotations

from typing import Any


def build_sources(sections: list[dict[str, Any]]) -> list[str]:
    """Format the source metadata already stored on parent sections."""
    sources = []
    for section in sections:
        metadata = section.get("metadata", {})
        source = metadata.get("source", "Unknown source")
        title = metadata.get("section", "Unknown section")
        start = metadata.get("page_start")
        end = metadata.get("page_end")
        if start and end:
            pages = f"p.{start}" if start == end else f"p.{start}–{end}"
            sources.append(f"[{source} — {title}, {pages}]")
        else:
            sources.append(f"[{source} — {title}]")
    return list(dict.fromkeys(sources))


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
        graph_context = self.dag.get_graph_context(
            collect_anchor_nodes(sections), search_mode="search"
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
