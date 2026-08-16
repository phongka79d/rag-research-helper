"""Direct Ask and Teach orchestration for the research helper."""

from __future__ import annotations

import json
import re
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


def format_relation_line(relation: dict[str, Any]) -> str:
    """Render one stored relation as a readable triple line."""
    source = relation.get("source", "")
    target = relation.get("target", "")
    rel = relation.get("relation") or relation.get("label") or ""
    return f"{source} —{rel}→ {target}"


def format_graph_context_lines(graph_context: Any) -> list[str]:
    """Turn Ask/Teach graph context into `source —REL→ target` lines."""
    if not graph_context:
        return []
    if isinstance(graph_context, dict):
        relations: list[Any] = [graph_context]
    elif isinstance(graph_context, list):
        relations = graph_context
    else:
        return [str(graph_context)]
    lines: list[str] = []
    for item in relations:
        if isinstance(item, dict) and ("source" in item or "target" in item):
            lines.append(format_relation_line(item))
    return lines


def _escape_mermaid_text(text: Any) -> str:
    """Escape characters that break Mermaid node and edge labels."""
    value = text if isinstance(text, str) else str(text or "")
    return (
        value.replace("\\", "\\\\")
        .replace('"', "#quot;")
        .replace("\n", " ")
        .replace("\r", " ")
        .replace("[", "#91;")
        .replace("]", "#93;")
        .replace("(", "#40;")
        .replace(")", "#41;")
        .replace("{", "#123;")
        .replace("}", "#125;")
        .replace("|", "#124;")
        .replace(">", "#62;")
        .replace("<", "#60;")
    )


def edges_to_mermaid(payload: dict[str, Any] | None) -> str:
    """Build a Mermaid `graph LR` string from a visual-graph `{nodes, edges}` payload."""
    graph = payload if isinstance(payload, dict) else {}
    edges = graph.get("edges") or []
    nodes = graph.get("nodes") or []
    id_for: dict[str, str] = {}
    lines = ["graph LR"]

    def node_id(name: str) -> str:
        if name not in id_for:
            id_for[name] = f"n{len(id_for)}"
        return id_for[name]

    def declare(name: str) -> str:
        safe = node_id(name)
        return f'{safe}["{_escape_mermaid_text(name)}"]'

    seen_names: set[str] = set()
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        source = str(edge.get("source", "") or "")
        target = str(edge.get("target", "") or "")
        label = edge.get("label") or edge.get("relation") or ""
        if not source and not target:
            continue
        seen_names.add(source)
        seen_names.add(target)
        lines.append(
            f"  {declare(source)} -->|{_escape_mermaid_text(label)}| {declare(target)}"
        )

    for node in nodes:
        if not isinstance(node, dict):
            continue
        name = str(node.get("id", "") or "")
        if name and name not in seen_names:
            seen_names.add(name)
            lines.append(f"  {declare(name)}")

    return "\n".join(lines)


_INLINE_PAREN_MATH = re.compile(r"\\\((.+?)\\\)", re.DOTALL)
_DISPLAY_BRACKET_MATH = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)
_EQUATION_ENV = re.compile(
    r"\\begin\{(equation|align|gather)\*?\}(.+?)\\end\{\1\*?\}",
    re.DOTALL,
)
_DISPLAY_DOLLAR_MATH = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
_PIPE_ROW = re.compile(r"^\s*\|.+\|\s*$")
_PIPE_SEPARATOR = re.compile(r"^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?\s*$")


def _is_pipe_row(line: str) -> bool:
    return bool(_PIPE_ROW.match(line))


def _is_pipe_separator(line: str) -> bool:
    return bool(_PIPE_SEPARATOR.match(line))


def _pipe_column_count(line: str) -> int:
    cells = [cell for cell in line.strip().strip("|").split("|")]
    return max(len(cells), 1)


def _prepare_markdown_tables(text: str) -> str:
    """Insert GFM separators and blank lines so Streamlit can render pipe tables."""
    lines = text.splitlines()
    prepared: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if _is_pipe_row(line) and not _is_pipe_separator(line):
            if prepared and prepared[-1].strip() and not _is_pipe_row(prepared[-1]):
                prepared.append("")
            prepared.append(line)
            nxt = lines[index + 1] if index + 1 < len(lines) else ""
            if nxt and _is_pipe_row(nxt) and not _is_pipe_separator(nxt):
                prepared.append("|" + "|".join(["---"] * _pipe_column_count(line)) + "|")
            index += 1
            continue
        prepared.append(line)
        index += 1
    return "\n".join(prepared)


def normalize_research_markdown(text: Any) -> str:
    """Rewrite common LaTeX/table forms into Streamlit-friendly Markdown and KaTeX."""
    value = text if isinstance(text, str) else str(text or "")
    if not value.strip():
        return ""
    value = _EQUATION_ENV.sub(lambda match: f"\n\n$$\n{match.group(2).strip()}\n$$\n\n", value)
    value = _DISPLAY_BRACKET_MATH.sub(
        lambda match: f"\n\n$$\n{match.group(1).strip()}\n$$\n\n", value
    )
    value = _INLINE_PAREN_MATH.sub(lambda match: f"${match.group(1).strip()}$", value)
    return _prepare_markdown_tables(value)


def iter_research_display_blocks(text: Any) -> list[tuple[str, str]]:
    """Split normalized research Markdown into markdown and display-math blocks."""
    normalized = normalize_research_markdown(text)
    if not normalized.strip():
        return []
    blocks: list[tuple[str, str]] = []
    cursor = 0
    for match in _DISPLAY_DOLLAR_MATH.finditer(normalized):
        prefix = normalized[cursor : match.start()]
        if prefix.strip():
            blocks.append(("markdown", prefix.strip("\n")))
        latex = match.group(1).strip()
        if latex:
            blocks.append(("latex", latex))
        cursor = match.end()
    suffix = normalized[cursor:]
    if suffix.strip():
        blocks.append(("markdown", suffix.strip("\n")))
    return blocks


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
                collect_anchor_nodes(sections),
                search_mode="search",
                source=target_file,
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
                step.get("concepts", []),
                search_mode="one_hop",
                source=target_file,
            )
            lessons.append(
                {
                    "step": step,
                    "content": self.llm.teach_step(
                        section_text=section["page_content"],
                        roadmap_step=step,
                        graph_context=graph_context,
                    ),
                    "graph_context": graph_context,
                }
            )
        return lessons
