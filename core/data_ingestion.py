"""Ahead-of-time compilation of research-paper sections."""

from __future__ import annotations

from hashlib import md5
from pathlib import Path
from typing import Any

from database.document_processor import DocumentProcessor


def make_parent_id(metadata: dict[str, Any]) -> str:
    """Keep section IDs stable across runs for cache checks and upserts."""
    source = metadata.get("source", "")
    section = metadata.get("section", "")
    return md5(f"{source}__{section}".encode("utf-8")).hexdigest()


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


def ingest_document(
    file_path: str | Path,
    db: Any,
    llm: Any,
    dag: Any,
    processor: DocumentProcessor | None = None,
    force_reingest: bool = False,
) -> dict[str, list[str]]:
    """Compile each source section into graph, roadmap, parent, and HyDE children."""
    sections = (processor or DocumentProcessor()).process(file_path)
    existing_nodes = dag.get_all_concept_names()
    ingested: list[str] = []
    skipped: list[str] = []

    for section in sections:
        full_text = section["page_content"]
        metadata = dict(section["metadata"])
        parent_id = make_parent_id(metadata)
        label = f"{metadata['source']}::{metadata['section']}"

        if db.section_exists(parent_id) and not force_reingest:
            skipped.append(label)
            continue

        aot = llm.extract_section_plan_and_graph(
            full_text,
            existing_nodes=existing_nodes,
        )
        graph = aot.get("knowledge_graph", {})
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])

        for node in nodes:
            name = str(node.get("name", "")).strip()
            if name and name not in existing_nodes:
                existing_nodes.append(name)

        dag.save_knowledge_graph(
            nodes=nodes,
            edges=edges,
            source=metadata,
            main_entities=aot.get("main_entities", []),
        )

        for seq_id, step_data in enumerate(aot.get("learning_roadmap", [])):
            step = {**step_data, "seq_id": seq_id}
            db.upsert_roadmap_step(step, parent_id=parent_id, metadata=metadata)

        questions = llm.generate_hypothetical_questions(full_text, num_questions=5)
        parent_metadata = {
            **metadata,
            "main_entities": aot.get("main_entities", []),
            "anchor_nodes": collect_anchor_nodes(aot),
        }
        db.upsert_section(full_text, parent_metadata, parent_id)
        db.upsert_questions(questions, parent_id, metadata["source"])
        ingested.append(label)

    return {"ingested": ingested, "skipped": skipped}
