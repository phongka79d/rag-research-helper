import threading

import pytest

from core.data_ingestion import (
    collect_anchor_nodes,
    filter_aot_to_section,
    ingest_document,
    make_content_hash,
    make_parent_id,
)
from core.schemas import SectionAOTResult
from database.document_processor import DocumentProcessor


SECTION_TEXT = "LoRA freezes base weights and trains a low-rank matrix."


def make_section(
    section: str = "Method", text: str = SECTION_TEXT, seq_id: int = 0
) -> dict:
    return {
        "page_content": text,
        "metadata": {
            "source": "lora.md",
            "section": section,
            "seq_id": seq_id,
            "page_start": 1,
            "page_end": 1,
        },
    }


def make_aot(include_unsupported: bool = False) -> dict:
    names = ["LoRA", "Low-Rank Matrix"]
    if include_unsupported:
        names.append("PEFT")
    return {
        "main_entities": ["LoRA", *( ["PEFT"] if include_unsupported else [])],
        "learning_roadmap": [
            {
                "title": "Mechanism",
                "content_focus": "Low-rank matrices update a frozen model.",
                "concepts": names,
            }
        ],
        "knowledge_graph": {
            "nodes": [
                {"name": "LoRA", "description": "Adaptation method."},
                {"name": "Low-Rank Matrix", "description": "Compact update."},
                *(
                    [{"name": "PEFT", "description": "Unsupported term."}]
                    if include_unsupported
                    else []
                ),
            ],
            "edges": [
                {
                    "source": "Low-Rank Matrix",
                    "target": "LoRA",
                    "relation": "PREREQUISITE_OF",
                },
                *(
                    [
                        {
                            "source": "PEFT",
                            "target": "LoRA",
                            "relation": "RELATES_TO",
                        }
                    ]
                    if include_unsupported
                    else []
                ),
            ],
        },
    }


class FakeProcessor:
    def __init__(self, sections: list[dict] | None = None, report: dict | None = None):
        self.sections = [make_section()] if sections is None else sections
        self.last_report = report or {
            "retained_section_count": len(self.sections),
            "bibliography_omitted": False,
        }

    def process(self, file_path):
        return self.sections


class FakeLLM:
    def __init__(self, aot: dict | None = None):
        self.aot = aot or make_aot()
        self.aot_text = None
        self.question_text = None
        self.existing_node_calls: list[list[str]] = []

    def extract_section_plan_and_graph(self, text, existing_nodes):
        self.aot_text = text
        self.existing_node_calls.append(list(existing_nodes))
        return self.aot

    def generate_hypothetical_questions(self, text, num_questions):
        self.question_text = text
        assert num_questions == 5
        return [
            {"question": f"Question {index}", "key_knowledge": "Grounded answer."}
            for index in range(num_questions)
        ]


class ConcurrentLLM(FakeLLM):
    def __init__(self):
        super().__init__()
        self.aot_started = threading.Event()
        self.questions_started = threading.Event()

    def extract_section_plan_and_graph(self, text, existing_nodes):
        self.aot_started.set()
        assert self.questions_started.wait(2)
        return super().extract_section_plan_and_graph(text, existing_nodes)

    def generate_hypothetical_questions(self, text, num_questions):
        self.questions_started.set()
        assert self.aot_started.wait(2)
        return super().generate_hypothetical_questions(text, num_questions)


class FailingQuestionsLLM(FakeLLM):
    def generate_hypothetical_questions(self, text, num_questions):
        raise RuntimeError("question generation failed")


class FakeDAG:
    def __init__(self):
        self.saved = []
        self.removed = []

    def get_all_concept_names(self):
        raise AssertionError("ingestion must not read global concepts")

    def save_knowledge_graph(self, **kwargs):
        self.saved.append(kwargs)

    def remove_source_locator(self, metadata):
        self.removed.append(metadata)


class FakeStore:
    def __init__(self):
        self.existing_hashes: dict[str, str] = {}
        self.previous_sections: list[dict] = []
        self.fail_on_questions = False
        self.calls = []
        self.get_section_calls = []

    def section_exists(self, parent_id, content_hash=None):
        return self.existing_hashes.get(parent_id) == content_hash

    def get_section_exact(self, source, section):
        self.get_section_calls.append((source, section))
        return self.previous_sections

    def delete_parent(self, parent_id):
        self.calls.append(("delete", parent_id))

    def upsert_curriculum_section(
        self, roadmap_steps, text, roadmap_metadata, section_metadata, parent_id
    ):
        self.calls.append(
            (
                "curriculum",
                roadmap_steps,
                text,
                roadmap_metadata,
                section_metadata,
                parent_id,
            )
        )

    def upsert_questions(self, questions, parent_id, source):
        self.calls.append(("questions", questions, parent_id, source))
        if self.fail_on_questions:
            raise RuntimeError("question persistence failed")


def expected_result(ingested: list[str], skipped: list[str]) -> dict:
    return {
        "ingested": ingested,
        "skipped": skipped,
        "report": {"retained_section_count": 1, "bibliography_omitted": False},
    }


def test_markdown_sections_keep_order_and_metadata():
    sections = DocumentProcessor().process_markdown(
        "# Test Paper\n\n## Abstract\nA summary.\n\n## Method\nThe method text.\n",
        "paper.md",
    )

    assert [section["metadata"]["section"] for section in sections] == [
        "Abstract",
        "Method",
    ]
    assert [section["metadata"]["seq_id"] for section in sections] == [0, 1]
    assert sections[1]["metadata"] == {
        "source": "paper.md",
        "section": "Method",
        "page_start": 1,
        "page_end": 1,
        "seq_id": 1,
    }


def test_aot_schema_normalizes_unknown_graph_relations():
    result = SectionAOTResult.model_validate(
        {
            "main_entities": ["LoRA"],
            "learning_roadmap": [
                {
                    "title": "How LoRA works",
                    "content_focus": "Low-rank updates",
                    "concepts": ["LoRA"],
                }
            ],
            "knowledge_graph": {
                "nodes": [{"name": "LoRA"}],
                "edges": [
                    {"source": "Matrix", "target": "LoRA", "relation": "unknown"}
                ],
            },
        }
    )

    assert result.learning_roadmap[0].seq_id == 0
    assert result.knowledge_graph.edges[0].relation == "RELATES_TO"


def test_ingestion_writes_grounded_aot_in_curriculum_question_order():
    llm = FakeLLM()
    dag = FakeDAG()
    store = FakeStore()

    result = ingest_document("ignored.md", store, llm, dag, processor=FakeProcessor())

    parent_id = make_parent_id({"source": "lora.md", "section": "Method"})
    assert result == expected_result(["lora.md::Method"], [])
    assert llm.aot_text == llm.question_text == SECTION_TEXT
    assert llm.existing_node_calls == [[]]
    assert [call[0] for call in store.calls] == ["delete", "curriculum", "questions"]
    assert store.calls[1][1][0]["seq_id"] == 0
    assert store.calls[1][4]["anchor_nodes"] == ["LoRA", "Low-Rank Matrix"]
    assert store.calls[1][4]["content_hash"] == make_content_hash(SECTION_TEXT)
    assert store.calls[1][5] == parent_id
    assert len(store.calls[2][1]) == 5
    assert dag.saved[0]["main_entities"] == ["LoRA"]
    assert dag.removed == [make_section()["metadata"]]


def test_existing_section_skips_unless_forced():
    llm = FakeLLM()
    dag = FakeDAG()
    store = FakeStore()
    store.existing_hashes[make_parent_id(make_section()["metadata"])] = make_content_hash(
        SECTION_TEXT
    )

    result = ingest_document("ignored.md", store, llm, dag, processor=FakeProcessor())

    assert result == expected_result([], ["lora.md::Method"])
    assert store.calls == []
    assert dag.saved == []
    assert collect_anchor_nodes(
        {
            "main_entities": ["A", "A"],
            "knowledge_graph": {
                "nodes": [{"name": "B"}],
                "edges": [{"source": "B", "target": "A"}],
            },
        }
    ) == ["A", "B"]


def test_changed_section_replaces_only_its_parent_points():
    llm = FakeLLM()
    dag = FakeDAG()
    store = FakeStore()
    store.existing_hashes[make_parent_id(make_section()["metadata"])] = "old-content-hash"

    result = ingest_document("ignored.md", store, llm, dag, processor=FakeProcessor())

    assert result == expected_result(["lora.md::Method"], [])
    assert [call[0] for call in store.calls] == ["delete", "curriculum", "questions"]
    assert len(dag.removed) == 1


def test_failed_ingestion_cleans_current_parent_and_locator():
    llm = FakeLLM()
    dag = FakeDAG()
    store = FakeStore()
    store.fail_on_questions = True

    with pytest.raises(RuntimeError, match="question persistence failed"):
        ingest_document("ignored.md", store, llm, dag, processor=FakeProcessor())

    assert [call[0] for call in store.calls] == [
        "delete",
        "curriculum",
        "questions",
        "delete",
    ]
    assert len(dag.removed) == 2


def test_aot_filter_keeps_only_current_section_terms_and_paper_local_reuse():
    sections = [
        make_section("First", SECTION_TEXT, 0),
        make_section("Second", SECTION_TEXT, 1),
    ]
    llm = FakeLLM(make_aot(include_unsupported=True))
    dag = FakeDAG()
    store = FakeStore()

    result = ingest_document(
        "ignored.md", store, llm, dag, processor=FakeProcessor(sections=sections)
    )

    assert result["ingested"] == ["lora.md::First", "lora.md::Second"]
    assert llm.existing_node_calls == [[], ["LoRA", "Low-Rank Matrix"]]
    assert all("Existing Concept" not in names for names in llm.existing_node_calls)
    for saved in dag.saved:
        assert saved["main_entities"] == ["LoRA"]
        assert [node["name"] for node in saved["nodes"]] == [
            "LoRA",
            "Low-Rank Matrix",
        ]
        assert [edge["source"] for edge in saved["edges"]] == ["Low-Rank Matrix"]
    assert filter_aot_to_section(make_aot(include_unsupported=True), SECTION_TEXT)[
        "learning_roadmap"
    ][0]["concepts"] == ["LoRA", "Low-Rank Matrix"]


def test_empty_parse_fails_before_database_or_graph_mutation():
    empty_processor = FakeProcessor(
        sections=[], report={"retained_section_count": 0, "bibliography_omitted": False}
    )
    store = FakeStore()
    dag = FakeDAG()

    with pytest.raises(ValueError, match="No retained paper body sections"):
        ingest_document("ignored.pdf", store, FakeLLM(), dag, processor=empty_processor)

    assert store.calls == []
    assert dag.saved == []
    assert dag.removed == []


def test_force_reingest_removes_stale_current_source_only_after_success():
    obsolete_metadata = {
        "source": "lora.md",
        "section": "References",
        "seq_id": 9,
        "page_start": 10,
        "page_end": 10,
    }
    obsolete_parent_id = make_parent_id(obsolete_metadata)
    store = FakeStore()
    store.previous_sections = [
        {
            "page_content": "Old bibliography.",
            "metadata": {**obsolete_metadata, "parent_id": obsolete_parent_id},
        }
    ]
    dag = FakeDAG()

    result = ingest_document(
        "ignored.md",
        store,
        FakeLLM(),
        dag,
        processor=FakeProcessor(),
        force_reingest=True,
    )

    assert result == expected_result(["lora.md::Method"], [])
    assert store.get_section_calls == [("lora.md", "")]
    assert store.calls[-1] == ("delete", obsolete_parent_id)
    assert dag.removed[-1] == {**obsolete_metadata, "parent_id": obsolete_parent_id}


def test_failed_force_reingest_does_not_clean_stale_sections():
    obsolete_metadata = {
        "source": "lora.md",
        "section": "References",
        "seq_id": 9,
        "page_start": 10,
        "page_end": 10,
    }
    obsolete_parent_id = make_parent_id(obsolete_metadata)
    store = FakeStore()
    store.previous_sections = [
        {"metadata": {**obsolete_metadata, "parent_id": obsolete_parent_id}}
    ]
    store.fail_on_questions = True
    dag = FakeDAG()

    with pytest.raises(RuntimeError, match="question persistence failed"):
        ingest_document(
            "ignored.md",
            store,
            FakeLLM(),
            dag,
            processor=FakeProcessor(),
            force_reingest=True,
        )

    assert ("delete", obsolete_parent_id) not in store.calls
    assert obsolete_metadata not in dag.removed


def test_generation_calls_overlap_without_concurrent_section_persistence():
    llm = ConcurrentLLM()
    store = FakeStore()
    dag = FakeDAG()

    ingest_document("ignored.md", store, llm, dag, processor=FakeProcessor())

    assert llm.aot_started.is_set()
    assert llm.questions_started.is_set()
    assert [call[0] for call in store.calls] == ["delete", "curriculum", "questions"]


def test_generation_failure_writes_nothing_for_current_section():
    store = FakeStore()
    dag = FakeDAG()

    with pytest.raises(RuntimeError, match="question generation failed"):
        ingest_document(
            "ignored.md", store, FailingQuestionsLLM(), dag, processor=FakeProcessor()
        )

    assert store.calls == []
    assert dag.saved == []
    assert dag.removed == []


def test_progress_reports_compiling_only_before_finished_sections():
    first = make_section("Abstract", "LoRA freezes base weights.", 0)
    second = make_section("Method", SECTION_TEXT, 1)
    store = FakeStore()
    store.existing_hashes[make_parent_id(first["metadata"])] = make_content_hash(
        first["page_content"]
    )
    events = []

    result = ingest_document(
        "ignored.md",
        store,
        FakeLLM(),
        FakeDAG(),
        processor=FakeProcessor(sections=[first, second]),
        progress_callback=events.append,
    )

    assert result["skipped"] == ["lora.md::Abstract"]
    assert events == [
        {
            "completed": 1,
            "total": 2,
            "section": "lora.md::Abstract",
            "status": "up_to_date",
        },
        {
            "completed": 1,
            "total": 2,
            "section": "lora.md::Method",
            "status": "compiling",
        },
        {
            "completed": 2,
            "total": 2,
            "section": "lora.md::Method",
            "status": "compiled",
        },
    ]
