"""Ahead-of-time compilation of research-paper sections."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from hashlib import md5, sha256
from pathlib import Path
import re
from typing import Any

from database.document_processor import DocumentProcessor


_NON_ALPHANUMERIC = re.compile(r"[\W_]+", re.UNICODE)


def make_parent_id(metadata: dict[str, Any]) -> str:
    """Keep section IDs stable across runs for cache checks and upserts."""
    source = metadata.get("source", "")
    section = metadata.get("section", "")
    # Ponytail: retain the original source-plus-section ID contract to avoid a data migration.
    return md5(f"{source}__{section}".encode("utf-8")).hexdigest()


def make_content_hash(text: str) -> str:
    """Identify the exact source text compiled into a parent section."""
    return sha256(text.encode("utf-8")).hexdigest()


def collect_anchor_nodes(aot: dict[str, Any]) -> list[str]:
    """Collect the AOT graph concepts stored with a parent section."""
    names: list[str] = []
    graph = aot.get("knowledge_graph", {})
    for name in aot.get("main_entities", []):
        if name:
            names.append(str(name).strip())
    for node in graph.get("nodes", []):
        if node.get("name"):
            names.append(str(node["name"]).strip())
    for edge in graph.get("edges", []):
        for endpoint in (edge.get("source"), edge.get("target")):
            if endpoint:
                names.append(str(endpoint).strip())
    return [name for name in dict.fromkeys(names) if name]


def _normalized_phrase(value: Any) -> str:
    """Normalize model terms and source text for conservative phrase matching."""
    return " ".join(_NON_ALPHANUMERIC.sub(" ", str(value).casefold()).split())


def _is_grounded_name(name: Any, normalized_section: str) -> bool:
    phrase = _normalized_phrase(name)
    return bool(phrase) and f" {phrase} " in f" {normalized_section} "


def _grounded_names(values: Any, normalized_section: str) -> list[str]:
    """Keep unique model-proposed names only when this section contains them."""
    if not isinstance(values, list):
        return []
    names: list[str] = []
    seen: set[str] = set()
    for value in values:
        name = str(value).strip()
        normalized_name = _normalized_phrase(name)
        if (
            normalized_name
            and normalized_name not in seen
            and _is_grounded_name(name, normalized_section)
        ):
            names.append(name)
            seen.add(normalized_name)
    return names


def filter_aot_to_section(aot: dict[str, Any], section_text: str) -> dict[str, Any]:
    """Remove AOT concepts that cannot be evidenced in the current section."""
    normalized_section = _normalized_phrase(section_text)
    main_entities = _grounded_names(aot.get("main_entities", []), normalized_section)

    roadmap: list[dict[str, Any]] = []
    for raw_step in aot.get("learning_roadmap", []):
        if not isinstance(raw_step, dict):
            continue
        step = dict(raw_step)
        step["concepts"] = _grounded_names(
            step.get("concepts", []), normalized_section
        )
        if str(step.get("title", "")).strip() and str(
            step.get("content_focus", "")
        ).strip():
            roadmap.append(step)

    raw_graph = aot.get("knowledge_graph", {})
    graph = raw_graph if isinstance(raw_graph, dict) else {}
    nodes: list[dict[str, Any]] = []
    for raw_node in graph.get("nodes", []):
        if not isinstance(raw_node, dict):
            continue
        name = str(raw_node.get("name", "")).strip()
        if _is_grounded_name(name, normalized_section):
            nodes.append({**raw_node, "name": name})

    edges: list[dict[str, Any]] = []
    for raw_edge in graph.get("edges", []):
        if not isinstance(raw_edge, dict):
            continue
        source = str(raw_edge.get("source", "")).strip()
        target = str(raw_edge.get("target", "")).strip()
        if _is_grounded_name(source, normalized_section) and _is_grounded_name(
            target, normalized_section
        ):
            edges.append({**raw_edge, "source": source, "target": target})

    return {
        "main_entities": main_entities,
        "learning_roadmap": roadmap,
        "knowledge_graph": {"nodes": nodes, "edges": edges},
    }


def _append_paper_nodes(existing_nodes: list[str], aot: dict[str, Any]) -> None:
    """Reuse only terms accepted from earlier sections of this same paper."""
    known = {_normalized_phrase(name) for name in existing_nodes}
    for name in collect_anchor_nodes(aot):
        normalized_name = _normalized_phrase(name)
        if normalized_name and normalized_name not in known:
            existing_nodes.append(name)
            known.add(normalized_name)


def _report_progress(
    callback: Callable[[dict[str, Any]], None] | None,
    completed: int,
    total: int,
    section: str,
    status: str,
) -> None:
    if callback is not None:
        callback(
            {
                "completed": completed,
                "total": total,
                "section": section,
                "status": status,
            }
        )


def ingest_document(
    file_path: str | Path,
    db: Any,
    llm: Any,
    dag: Any,
    processor: DocumentProcessor | None = None,
    force_reingest: bool = False,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Compile each source section into graph, roadmap, parent, and HyDE children."""
    document_processor = processor or DocumentProcessor()
    sections = document_processor.process(file_path)
    report = dict(getattr(document_processor, "last_report", {}))
    report.setdefault("retained_section_count", len(sections))
    report.setdefault("bibliography_omitted", False)
    if not sections:
        raise ValueError("No retained paper body sections were found; nothing was replaced.")

    # Ponytail: do not leak global graph concepts into a different paper's extraction.
    existing_nodes: list[str] = []
    ingested: list[str] = []
    skipped: list[str] = []
    total = len(sections)
    completed = 0
    source = str(sections[0]["metadata"].get("source", ""))
    previous_sections = (
        db.get_section_exact(source, "") if force_reingest and source else []
    )
    current_parent_ids = {
        make_parent_id(dict(section["metadata"])) for section in sections
    }

    for section in sections:
        full_text = section["page_content"]
        metadata = dict(section["metadata"])
        parent_id = make_parent_id(metadata)
        content_hash = make_content_hash(full_text)
        label = f"{metadata['source']}::{metadata['section']}"

        if db.section_exists(parent_id, content_hash) and not force_reingest:
            skipped.append(label)
            completed += 1
            _report_progress(progress_callback, completed, total, label, "up_to_date")
            continue

        replacement_started = False
        try:
            _report_progress(progress_callback, completed, total, label, "compiling")
            with ThreadPoolExecutor(max_workers=2) as executor:
                aot_future = executor.submit(
                    llm.extract_section_plan_and_graph,
                    full_text,
                    existing_nodes=list(existing_nodes),
                )
                questions_future = executor.submit(
                    llm.generate_hypothetical_questions,
                    full_text,
                    num_questions=5,
                )
                aot = filter_aot_to_section(aot_future.result(), full_text)
                questions = questions_future.result()
            graph = aot.get("knowledge_graph", {})
            nodes = graph.get("nodes", [])
            edges = graph.get("edges", [])

            replacement_started = True
            db.delete_parent(parent_id)
            dag.remove_source_locator(metadata)

            _append_paper_nodes(existing_nodes, aot)

            dag.save_knowledge_graph(
                nodes=nodes,
                edges=edges,
                source=metadata,
                main_entities=aot.get("main_entities", []),
            )

            roadmap_steps = [
                {**step_data, "seq_id": seq_id}
                for seq_id, step_data in enumerate(aot.get("learning_roadmap", []))
            ]

            parent_metadata = {
                **metadata,
                "content_hash": content_hash,
                "main_entities": aot.get("main_entities", []),
                "anchor_nodes": collect_anchor_nodes(aot),
            }
            db.upsert_curriculum_section(
                roadmap_steps,
                full_text,
                metadata,
                parent_metadata,
                parent_id,
            )
            db.upsert_questions(questions, parent_id, metadata["source"])
            ingested.append(label)
            completed += 1
            _report_progress(progress_callback, completed, total, label, "compiled")
        except Exception:
            if replacement_started:
                # Ponytail: cross-store replacement is intentionally best-effort, not atomic.
                try:
                    db.delete_parent(parent_id)
                except Exception:
                    pass
                try:
                    dag.remove_source_locator(metadata)
                except Exception:
                    pass
            raise

    if force_reingest:
        for previous_section in previous_sections:
            previous_metadata = dict(previous_section.get("metadata", {}))
            previous_parent_id = str(
                previous_metadata.get("parent_id") or make_parent_id(previous_metadata)
            )
            if (
                previous_metadata.get("source") == source
                and previous_parent_id not in current_parent_ids
            ):
                db.delete_parent(previous_parent_id)
                dag.remove_source_locator(previous_metadata)

    return {"ingested": ingested, "skipped": skipped, "report": report}
