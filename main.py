"""Streamlit entry point for RAG Research Helper."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from config.settings import Settings
from core.data_ingestion import ingest_document
from database.semantic_dag import Neo4jManager
from database.structural_db import QdrantVectorStore
from orchestrator.llm_service import LLMService
from runtime.engine import RuntimeEngine

PAPERS_DIR = Path("data/papers")


def get_app_objects() -> tuple[RuntimeEngine, QdrantVectorStore, Neo4jManager]:
    settings = Settings()
    settings.validate()
    llm = LLMService(settings)
    db = QdrantVectorStore(settings, llm)
    dag = Neo4jManager(settings)
    dag.verify_connection()
    return RuntimeEngine(llm, db, dag), db, dag


def stored_sections(db: QdrantVectorStore) -> list[dict]:
    return db.get_section_exact("", "")


def paper_names(sections: list[dict]) -> list[str]:
    return sorted(
        {
            section["metadata"].get("source", "")
            for section in sections
            if section["metadata"].get("source")
        }
    )


def main() -> None:
    st.set_page_config(page_title="RAG Research Helper", layout="wide")
    st.title("RAG Research Helper")
    st.caption("Research-paper Ask, section-by-section Teach, and concept Graph.")

    try:
        engine, db, dag = get_app_objects()
    except Exception as error:
        st.error(str(error))
        st.stop()

    sections = stored_sections(db)
    papers = paper_names(sections)

    with st.sidebar:
        st.header("Papers")
        uploaded = st.file_uploader("Upload PDF or Markdown", type=["pdf", "md", "markdown"])
        force_reingest = st.checkbox("Force re-ingest", value=False)
        if uploaded and st.button("Ingest paper", use_container_width=True):
            PAPERS_DIR.mkdir(parents=True, exist_ok=True)
            destination = PAPERS_DIR / Path(uploaded.name).name
            destination.write_bytes(uploaded.getvalue())
            try:
                with st.spinner(f"Ingesting {destination.name}..."):
                    result = ingest_document(
                        destination,
                        db,
                        engine.llm,
                        dag,
                        force_reingest=force_reingest,
                    )
                st.success(
                    f"Ingested {len(result['ingested'])}; skipped {len(result['skipped'])}."
                )
                st.rerun()
            except Exception as error:
                st.error(str(error))

        st.divider()
        selected_paper = st.selectbox(
            "Paper filter",
            options=["All papers", *papers],
            disabled=not papers,
        )

    ask_tab, teach_tab, graph_tab = st.tabs(["Ask", "Teach", "Graph"])

    with ask_tab:
        query = st.text_area(
            "Ask about one paper or compare across papers",
            placeholder="How does QLoRA extend LoRA?",
        )
        if st.button("Ask", type="primary", use_container_width=True, disabled=not query.strip()):
            target_file = "" if selected_paper == "All papers" else selected_paper
            try:
                with st.spinner("Finding evidence and concept context..."):
                    result = engine.ask(query.strip(), target_file=target_file)
                st.markdown(result["answer"])
                if result["sources"]:
                    st.caption("Sources")
                    for source in result["sources"]:
                        st.write(source)
                if result["graph_context"]:
                    with st.expander("Concept graph context"):
                        st.json(result["graph_context"])
            except Exception as error:
                st.error(str(error))

    with teach_tab:
        if not papers:
            st.info("Ingest a paper to teach one of its sections.")
        else:
            teach_paper = st.selectbox("Paper", papers, key="teach-paper")
            available_sections = [
                section["metadata"]["section"]
                for section in sections
                if section["metadata"].get("source") == teach_paper
            ]
            teach_section = st.selectbox("Section", available_sections, key="teach-section")
            if st.button("Teach this section", use_container_width=True):
                try:
                    with st.spinner("Building the roadmap lesson..."):
                        lessons = engine.teach_section(teach_paper, teach_section)
                    if not lessons:
                        st.warning("No compiled roadmap was found for this section.")
                    for lesson in lessons:
                        st.subheader(
                            f"{lesson['step'].get('seq_id', 0) + 1}. {lesson['step']['title']}"
                        )
                        st.markdown(lesson["content"])
                except Exception as error:
                    st.error(str(error))

    with graph_tab:
        if not papers:
            st.info("Ingest a paper to inspect its concept graph.")
        else:
            graph_paper = st.selectbox("Paper", papers, key="graph-paper")
            graph_sections = [
                section["metadata"]["section"]
                for section in sections
                if section["metadata"].get("source") == graph_paper
            ]
            graph_section = st.selectbox("Section", graph_sections, key="graph-section")
            locator = f"{graph_paper}::{graph_section}"
            graph = dag.get_visual_graph(locator)
            st.caption(f"{len(graph['nodes'])} concepts, {len(graph['edges'])} relationships")
            st.subheader("Concepts")
            st.dataframe(graph["nodes"], use_container_width=True, hide_index=True)
            st.subheader("Relationships")
            st.dataframe(graph["edges"], use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
