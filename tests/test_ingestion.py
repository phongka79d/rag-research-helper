import pytest

from core.data_ingestion import (
    collect_anchor_nodes,
    ingest_document,
    make_content_hash,
    make_parent_id,
)
from core.schemas import SectionAOTResult
from database.document_processor import DocumentProcessor


class FakeProcessor:
    def process(self, file_path):
        return [
            {
                "page_content": "LoRA freezes base weights and trains low-rank matrices.",
                "metadata": {
                    "source": "lora.md",
                    "section": "Method",
                    "seq_id": 0,
                    "page_start": 1,
                    "page_end": 1,
                },
            }
        ]


class FakeLLM:
    def __init__(self):
        self.aot_text = None
        self.question_text = None

    def extract_section_plan_and_graph(self, text, existing_nodes):
        self.aot_text = text
        return {
            "main_entities": ["LoRA"],
            "learning_roadmap": [
                {
                    "title": "Mechanism",
                    "content_focus": "Low-rank matrices update a frozen model.",
                    "concepts": ["LoRA", "Low-Rank Matrix"],
                }
            ],
            "knowledge_graph": {
                "nodes": [
                    {"name": "LoRA", "description": "Adaptation method."},
                    {"name": "Low-Rank Matrix", "description": "Compact update."},
                ],
                "edges": [
                    {
                        "source": "Low-Rank Matrix",
                        "target": "LoRA",
                        "relation": "PREREQUISITE_OF",
                    }
                ],
            },
        }

    def generate_hypothetical_questions(self, text, num_questions):
        self.question_text = text
        assert num_questions == 5
        return [
            {"question": f"Question {index}", "key_knowledge": "Grounded answer."}
            for index in range(num_questions)
        ]


class FakeDAG:
    def __init__(self):
        self.saved = []
        self.removed = []

    def get_all_concept_names(self):
        return ["Existing Concept"]

    def save_knowledge_graph(self, **kwargs):
        self.saved.append(kwargs)

    def remove_source_locator(self, metadata):
        self.removed.append(metadata)


class FakeStore:
    def __init__(self):
        self.existing = False
        self.content_hash = ""
        self.fail_on_questions = False
        self.calls = []

    def section_exists(self, parent_id, content_hash=None):
        return self.existing and (
            content_hash is None or content_hash == self.content_hash
        )

    def delete_parent(self, parent_id):
        self.calls.append(("delete", parent_id))

    def upsert_roadmap_step(self, step, parent_id, metadata):
        self.calls.append(("roadmap", step, parent_id, metadata))

    def upsert_section(self, text, metadata, parent_id):
        self.calls.append(("section", text, metadata, parent_id))

    def upsert_questions(self, questions, parent_id, source):
        self.calls.append(("questions", questions, parent_id, source))
        if self.fail_on_questions:
            raise RuntimeError("question persistence failed")


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


def test_ingestion_follows_aot_graph_roadmap_parent_child_order():
    llm = FakeLLM()
    dag = FakeDAG()
    store = FakeStore()

    result = ingest_document("ignored.md", store, llm, dag, processor=FakeProcessor())

    parent_id = make_parent_id(
        {"source": "lora.md", "section": "Method"}
    )
    assert result == {"ingested": ["lora.md::Method"], "skipped": []}
    assert llm.aot_text == llm.question_text
    assert [call[0] for call in store.calls] == [
        "delete",
        "roadmap",
        "section",
        "questions",
    ]
    assert store.calls[1][1]["seq_id"] == 0
    assert store.calls[2][2]["anchor_nodes"] == ["LoRA", "Low-Rank Matrix"]
    assert store.calls[2][2]["content_hash"] == make_content_hash(llm.aot_text)
    assert store.calls[2][3] == parent_id
    assert len(store.calls[3][1]) == 5
    assert dag.saved[0]["main_entities"] == ["LoRA"]
    assert dag.removed == [FakeProcessor().process("ignored.md")[0]["metadata"]]


def test_existing_section_skips_unless_forced():
    llm = FakeLLM()
    dag = FakeDAG()
    store = FakeStore()
    store.existing = True
    store.content_hash = make_content_hash(
        FakeProcessor().process("ignored.md")[0]["page_content"]
    )

    result = ingest_document("ignored.md", store, llm, dag, processor=FakeProcessor())

    assert result == {"ingested": [], "skipped": ["lora.md::Method"]}
    assert store.calls == []
    assert dag.saved == []
    assert collect_anchor_nodes({"main_entities": ["A", "A"], "knowledge_graph": {"nodes": [{"name": "B"}], "edges": [{"source": "B", "target": "A"}]}}) == ["A", "B"]


def test_changed_section_replaces_only_its_parent_points():
    llm = FakeLLM()
    dag = FakeDAG()
    store = FakeStore()
    store.existing = True
    store.content_hash = "old-content-hash"

    result = ingest_document("ignored.md", store, llm, dag, processor=FakeProcessor())

    assert result == {"ingested": ["lora.md::Method"], "skipped": []}
    assert [call[0] for call in store.calls] == [
        "delete",
        "roadmap",
        "section",
        "questions",
    ]
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
        "roadmap",
        "section",
        "questions",
        "delete",
    ]
    assert len(dag.removed) == 2
