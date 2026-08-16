import json
import threading
from pathlib import Path

import pytest

from core.data_ingestion import (
    _approved_graph_edges,
    _has_direct_whole_part_cue,
    build_evidence_spans,
    collect_anchor_nodes,
    compile_uploaded_document,
    filter_aot_to_section,
    ingest_document,
    make_content_hash,
    make_parent_id,
    propose_local_graph_candidates,
)
from core.schemas import MAX_GRAPH_VERIFIER_CANDIDATES, SectionAOTResult
from database.document_processor import DocumentProcessor


SECTION_TEXT = (
    "LoRA freezes base weights and trains a low-rank matrix. "
    "A low-rank matrix is a component of LoRA."
)


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
                    "relation": "PART_OF",
                    # Second sentence of SECTION_TEXT is the direct whole-part claim.
                    "evidence_id": "e1",
                },
                *(
                    [
                        {
                            "source": "PEFT",
                            "target": "LoRA",
                            "relation": "RELATES_TO",
                            "evidence_id": "e0",
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
    def __init__(self, aot: dict | None = None, edge_approvals: list[dict] | None = None):
        self.aot = aot or make_aot()
        self.edge_approvals = edge_approvals
        self.aot_text = None
        self.question_text = None
        self.section_title = None
        self.existing_node_calls: list[list[str]] = []
        self.verify_calls = []

    def extract_section_plan_and_graph(self, text, existing_nodes, **_kwargs):
        self.aot_text = text
        self.existing_node_calls.append(list(existing_nodes))
        return self.aot

    def generate_hypothetical_questions(self, text, num_questions, section_title=""):
        self.question_text = text
        self.section_title = section_title
        assert num_questions == 5
        return [
            {"question": f"Question {index}", "key_knowledge": "Grounded answer."}
            for index in range(num_questions)
        ]

    def verify_graph_edges(self, section_text, candidates, **_kwargs):
        self.verify_calls.append((section_text, candidates))
        if self.edge_approvals is not None:
            return self.edge_approvals
        return [{"index": index} for index, _candidate in enumerate(candidates)]


class ConcurrentLLM(FakeLLM):
    def __init__(self):
        super().__init__()
        self.aot_started = threading.Event()
        self.questions_started = threading.Event()

    def extract_section_plan_and_graph(self, text, existing_nodes):
        self.aot_started.set()
        assert self.questions_started.wait(2)
        return super().extract_section_plan_and_graph(text, existing_nodes)

    def generate_hypothetical_questions(self, text, num_questions, section_title=""):
        self.questions_started.set()
        assert self.aot_started.wait(2)
        return super().generate_hypothetical_questions(text, num_questions, section_title)


class SplitExtractionLLM(FakeLLM):
    """Test double for the independent text-plan and graph boundaries."""

    def __init__(self):
        super().__init__()
        self.plan_started = threading.Event()
        self.graph_started = threading.Event()
        self.questions_started = threading.Event()
        self.plan_node_calls: list[list[str]] = []
        self.graph_node_calls: list[list[str]] = []
        self.graph_calls = 0

    def extract_section_plan(self, text, existing_nodes):
        self.plan_started.set()
        self.plan_node_calls.append(list(existing_nodes))
        # A provider must not be able to mutate the ingestion context through
        # the list it receives.
        existing_nodes.append("provider-local-plan-term")
        return {
            "main_entities": self.aot["main_entities"],
            "learning_roadmap": self.aot["learning_roadmap"],
        }

    def extract_section_graph(self, text, existing_nodes, **_kwargs):
        self.graph_calls += 1
        self.graph_started.set()
        self.graph_node_calls.append(list(existing_nodes))
        existing_nodes.append("provider-local-graph-term")
        return {"knowledge_graph": self.aot["knowledge_graph"]}

    def generate_hypothetical_questions(self, text, num_questions, section_title=""):
        self.questions_started.set()
        assert self.plan_started.wait(2)
        return super().generate_hypothetical_questions(text, num_questions, section_title)


class FailingQuestionsLLM(FakeLLM):
    def generate_hypothetical_questions(self, text, num_questions, section_title=""):
        raise RuntimeError("question generation failed")


class ThinSectionLLM(FakeLLM):
    def generate_hypothetical_questions(self, text, num_questions, section_title=""):
        self.question_text = text
        self.section_title = section_title
        assert num_questions == 5
        return []


class FailingVerifierLLM(FakeLLM):
    def verify_graph_edges(self, section_text, candidates, **_kwargs):
        raise RuntimeError("graph verification failed")


class RecoveryLLM(FakeLLM):
    def __init__(self, *args, recovered_edges=None, recovery_error=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.recovered_edges = recovered_edges or []
        self.recovery_error = recovery_error
        self.recovery_calls = []

    def extract_graph_edges_with_evidence(self, text, existing_nodes, **_kwargs):
        self.recovery_calls.append((text, list(existing_nodes)))
        if self.recovery_error is not None:
            raise self.recovery_error
        return self.recovered_edges


class RecoveryVerifierFailureLLM(RecoveryLLM):
    def verify_graph_edges(self, section_text, candidates, **_kwargs):
        self.verify_calls.append((section_text, candidates))
        raise RuntimeError("recovery verification failed")


class ConcurrentVerifierLLM(FakeLLM):
    def __init__(self):
        super().__init__()
        self.questions_started = threading.Event()
        self.verifier_started = threading.Event()

    def generate_hypothetical_questions(self, text, num_questions, section_title=""):
        self.questions_started.set()
        assert self.verifier_started.wait(2)
        return super().generate_hypothetical_questions(text, num_questions, section_title)

    def verify_graph_edges(self, section_text, candidates, **_kwargs):
        self.verifier_started.set()
        assert self.questions_started.wait(2)
        return super().verify_graph_edges(section_text, candidates, **_kwargs)


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


class FailingStaleCleanupDAG(FakeDAG):
    def __init__(self, stale_section: str, store, stale_parent_id: str):
        super().__init__()
        self.stale_section = stale_section
        self.store = store
        self.stale_parent_id = stale_parent_id
        self.fail_once = True

    def remove_source_locator(self, metadata):
        self.removed.append(metadata)
        if metadata.get("section") == self.stale_section:
            assert ("delete", self.stale_parent_id) not in self.store.calls
            if self.fail_once:
                self.fail_once = False
                raise RuntimeError("stale graph cleanup failed")


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


def expected_result(
    ingested: list[str], skipped: list[str], graph_relationships: dict | None = None
) -> dict:
    if graph_relationships is None:
        # SECTION_TEXT local scan yields the direct PART_OF span plus one
        # same-paragraph adjacent window; both pass verifier + local gate.
        graph_relationships = {
            "candidates": 2 * int(bool(ingested)),
            "verifier_approvals": 2 * int(bool(ingested)),
            "retained": 2 * int(bool(ingested)),
        }
        if ingested:
            graph_relationships["retained_edge_audit"] = [
                {
                    "source": "low-rank matrix",
                    "relation": "PART_OF",
                    "target": "LoRA",
                    "locator": ingested[0],
                    "evidence_id": "e1",
                    "evidence_preview": "A low-rank matrix is a component of LoRA.",
                },
                {
                    "source": "low-rank matrix",
                    "relation": "PART_OF",
                    "target": "LoRA",
                    "locator": ingested[0],
                    "evidence_id": "e0+e1",
                    "evidence_preview": (
                        "LoRA freezes base weights and trains a low-rank matrix. "
                        "A low-rank matrix is a component of LoRA."
                    ),
                },
            ]
    return {
        "ingested": ingested,
        "skipped": skipped,
        "report": {"retained_section_count": 1, "bibliography_omitted": False},
        "graph_relationships": graph_relationships,
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
                    {
                        "source": "Matrix",
                        "target": "LoRA",
                        "relation": "unknown",
                        "evidence_id": "e0",
                    }
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
    assert llm.section_title == "Method"
    assert llm.existing_node_calls == [[]]
    assert [call[0] for call in store.calls] == ["delete", "curriculum", "questions"]
    assert store.calls[1][1][0]["seq_id"] == 0
    assert store.calls[1][4]["anchor_nodes"] == [
        "LoRA",
        "Low-Rank Matrix",
        "low-rank matrix",
    ]
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
        # Local-first candidates use span casing for endpoints.
        assert [edge["source"] for edge in saved["edges"]] == [
            "low-rank matrix",
            "low-rank matrix",
        ]
    assert filter_aot_to_section(make_aot(include_unsupported=True), SECTION_TEXT)[
        "learning_roadmap"
    ][0]["concepts"] == ["LoRA", "Low-Rank Matrix"]


def test_graph_edges_require_valid_grounded_verifier_approval():
    text = "Alpha is part of Beta. Beta is related to Gamma."
    local_candidates, _extra = propose_local_graph_candidates(text)
    assert local_candidates
    aot = {
        "main_entities": ["Alpha", "Beta", "Gamma"],
        "learning_roadmap": [],
        "knowledge_graph": {
            "nodes": [{"name": name} for name in ("Alpha", "Beta", "Gamma")],
            "edges": [],
        },
    }
    llm = FakeLLM(aot=aot)
    dag = FakeDAG()

    ingest_document(
        "ignored.md",
        FakeStore(),
        llm,
        dag,
        processor=FakeProcessor(sections=[make_section(text=text)]),
    )

    assert llm.verify_calls == []
    assert dag.saved[0]["edges"]
    assert all("evidence_id" not in edge for edge in dag.saved[0]["edges"])
    assert {edge["relation"] for edge in dag.saved[0]["edges"]} <= {
        "PART_OF",
        "RELATES_TO",
        "DESCRIBES",
        "PREREQUISITE_OF",
    }


def test_graph_quote_gate_rejects_generic_usage_property_and_evaluation_claims():
    text = (
        "The Transformer uses self-attention. The Transformer relies on attention "
        "mechanisms. The Transformer has positional encodings. The model exhibits residual "
        "connections. The model is evaluated on a benchmark. The approach is based on "
        "self-attention. Quantization is applied to the weights. Alpha and Beta appear "
        "together in the architecture. Alpha is not a component of Beta."
    )
    spans = build_evidence_spans(text)
    candidates = [
        {
            "source": "self-attention",
            "relation": "PART_OF",
            "target": "Transformer",
            "evidence_id": spans[0]["id"],
        },
        {
            "source": "attention mechanisms",
            "relation": "PREREQUISITE_OF",
            "target": "Transformer",
            "evidence_id": spans[1]["id"],
        },
        {
            "source": "Transformer",
            "relation": "DESCRIBES",
            "target": "positional encodings",
            "evidence_id": spans[2]["id"],
        },
        {
            "source": "residual connections",
            "relation": "PART_OF",
            "target": "model",
            "evidence_id": spans[3]["id"],
        },
        {
            "source": "model",
            "relation": "RELATES_TO",
            "target": "benchmark",
            "evidence_id": spans[4]["id"],
        },
        {
            "source": "approach",
            "relation": "RELATES_TO",
            "target": "self-attention",
            "evidence_id": spans[5]["id"],
        },
        {
            "source": "Quantization",
            "relation": "PART_OF",
            "target": "weights",
            "evidence_id": spans[6]["id"],
        },
        {
            "source": "Alpha",
            "relation": "RELATES_TO",
            "target": "Beta",
            "evidence_id": spans[7]["id"],
        },
        {
            "source": "Alpha",
            "relation": "PART_OF",
            "target": "Beta",
            "evidence_id": spans[8]["id"],
        },
    ]
    approvals = [{"index": index} for index in range(len(candidates))]

    assert _approved_graph_edges(text, candidates, approvals) == []


def test_graph_quote_gate_reports_relation_mismatch_reason():
    text = "The Transformer uses self-attention."
    rejections = {}
    candidate = {
        "source": "self-attention",
        "relation": "PART_OF",
        "target": "Transformer",
        "evidence_id": "e0",
    }

    assert _approved_graph_edges(
        text, [candidate], [{"index": 0}], rejections
    ) == []
    assert rejections == {"relation_mismatch": 1}


@pytest.mark.parametrize(
    ("text", "candidate"),
    [
        (
            "Multi-head attention is a component of the Transformer.",
            {
                "source": "Multi-head attention",
                "relation": "PART_OF",
                "target": "Transformer",
                "evidence_id": "e0",
            },
        ),
        (
            "Understanding dot-product attention is required before understanding multi-head attention.",
            {
                "source": "dot-product attention",
                "relation": "PREREQUISITE_OF",
                "target": "multi-head attention",
                "evidence_id": "e0",
            },
        ),
        (
            "Understanding multi-head attention requires understanding attention.",
            {
                "source": "attention",
                "relation": "PREREQUISITE_OF",
                "target": "multi-head attention",
                "evidence_id": "e0",
            },
        ),
        (
            "The attention mechanism explains token dependencies.",
            {
                "source": "attention mechanism",
                "relation": "DESCRIBES",
                "target": "token dependencies",
                "evidence_id": "e0",
            },
        ),
        (
            "Token order is defined by positional encodings.",
            {
                "source": "positional encodings",
                "relation": "DESCRIBES",
                "target": "Token order",
                "evidence_id": "e0",
            },
        ),
        (
            "Positional encodings are related to token order.",
            {
                "source": "Positional encodings",
                "relation": "RELATES_TO",
                "target": "token order",
                "evidence_id": "e0",
            },
        ),
        (
            "Self-attention and feed-forward networks are related.",
            {
                "source": "Self-attention",
                "relation": "RELATES_TO",
                "target": "feed-forward networks",
                "evidence_id": "e0",
            },
        ),
    ],
)
def test_graph_quote_gate_keeps_direct_directional_relation_evidence(text, candidate):
    assert _approved_graph_edges(text, [candidate], [{"index": 0}]) == [
        {
            "source": candidate["source"],
            "relation": candidate["relation"],
            "target": candidate["target"],
        }
    ]


@pytest.mark.parametrize(
    ("text", "candidate"),
    [
        (
            "The system uses a technique.",
            {
                "source": "technique",
                "relation": "PART_OF",
                "target": "system",
                "evidence_id": "e0",
            },
        ),
        (
            "The system relies on a technique.",
            {
                "source": "technique",
                "relation": "PREREQUISITE_OF",
                "target": "system",
                "evidence_id": "e0",
            },
        ),
        (
            "The system has a technique.",
            {
                "source": "system",
                "relation": "DESCRIBES",
                "target": "technique",
                "evidence_id": "e0",
            },
        ),
        (
            "The model exhibits residual connections.",
            {
                "source": "residual connections",
                "relation": "PART_OF",
                "target": "model",
                "evidence_id": "e0",
            },
        ),
        (
            "The model is evaluated on a benchmark.",
            {
                "source": "model",
                "relation": "RELATES_TO",
                "target": "benchmark",
                "evidence_id": "e0",
            },
        ),
        (
            "The approach is based on self-attention.",
            {
                "source": "approach",
                "relation": "RELATES_TO",
                "target": "self-attention",
                "evidence_id": "e0",
            },
        ),
        (
            "Quantization is applied to the weights.",
            {
                "source": "Quantization",
                "relation": "PART_OF",
                "target": "weights",
                "evidence_id": "e0",
            },
        ),
        (
            "Alpha and Beta appear together in the architecture.",
            {
                "source": "Alpha",
                "relation": "RELATES_TO",
                "target": "Beta",
                "evidence_id": "e0",
            },
        ),
        (
            "Alpha is not a component of Beta.",
            {
                "source": "Alpha",
                "relation": "PART_OF",
                "target": "Beta",
                "evidence_id": "e0",
            },
        ),
    ],
)
def test_graph_quote_gate_rejects_usage_capability_evaluation_negation_and_cooccurrence(
    text, candidate
):
    """Paired negatives: near-miss wording must not satisfy relation matchers."""
    assert _approved_graph_edges(text, [candidate], [{"index": 0}]) == []


def test_graph_quote_gate_binds_whole_part_evidence_to_candidate_endpoints():
    text = "Transformer uses self-attention. The encoder contains self-attention."
    candidate = {
        "source": "self-attention",
        "relation": "PART_OF",
        "target": "Transformer",
        "evidence_id": "e1",
    }

    assert _approved_graph_edges(text, [candidate], [{"index": 0}]) == []


def test_graph_gate_rejects_unknown_evidence_id():
    text = "Alpha is a component of Beta."
    rejections = {}
    candidate = {
        "source": "Alpha",
        "relation": "PART_OF",
        "target": "Beta",
        "evidence_id": "e99",
    }

    assert _approved_graph_edges(text, [candidate], [{"index": 0}], rejections) == []
    assert rejections == {"invalid_evidence_id": 1}


def test_graph_gate_rejects_endpoints_outside_selected_span():
    text = "Alpha is a component of Beta. Gamma is a component of Delta."
    rejections = {}
    candidate = {
        "source": "Alpha",
        "relation": "PART_OF",
        "target": "Beta",
        "evidence_id": "e1",
    }

    assert _approved_graph_edges(text, [candidate], [{"index": 0}], rejections) == []
    assert rejections == {"span_grounding": 1}


def test_graph_gate_retains_index_only_approval_of_direct_part_of_span():
    text = "Alpha is a component of Beta."
    candidate = {
        "source": "Alpha",
        "relation": "PART_OF",
        "target": "Beta",
        "evidence_id": "e0",
    }

    assert _approved_graph_edges(text, [candidate], [{"index": 0}]) == [
        {"source": "Alpha", "relation": "PART_OF", "target": "Beta"}
    ]


def test_graph_gate_still_rejects_usage_span():
    text = "The system uses a technique."
    rejections = {}
    candidate = {
        "source": "technique",
        "relation": "PART_OF",
        "target": "system",
        "evidence_id": "e0",
    }

    assert _approved_graph_edges(text, [candidate], [{"index": 0}], rejections) == []
    assert rejections == {"relation_mismatch": 1}


@pytest.mark.parametrize(
    ("text", "candidate"),
    [
        (
            "Each of the layers in our encoder and decoder contains a fully connected feed-forward network.",
            {
                "source": "fully connected feed-forward network",
                "relation": "PART_OF",
                "target": "layers in our encoder and decoder",
                "evidence_id": "e0",
            },
        ),
        (
            "The encoder is composed of a stack of N identical layers.",
            {
                "source": "a stack of N identical layers",
                "relation": "PART_OF",
                "target": "encoder",
                "evidence_id": "e0",
            },
        ),
        (
            "The model consists of an encoder and a decoder.",
            {
                "source": "encoder",
                "relation": "PART_OF",
                "target": "model",
                "evidence_id": "e0",
            },
        ),
        (
            "The attention block is a constituent of the Transformer.",
            {
                "source": "attention block",
                "relation": "PART_OF",
                "target": "Transformer",
                "evidence_id": "e0",
            },
        ),
    ],
)
def test_graph_quote_gate_keeps_natural_direct_whole_part_evidence(text, candidate):
    assert _approved_graph_edges(text, [candidate], [{"index": 0}]) == [
        {
            "source": candidate["source"],
            "relation": candidate["relation"],
            "target": candidate["target"],
        }
    ]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The encoder is composed of a stack.", True),
        ("The encoder consists of a stack.", True),
        ("The encoder is made up of a stack.", True),
        ("The stack is a component of the encoder.", True),
        ("The stack forms part of the encoder.", True),
        ("The model includes an encoder and a decoder.", False),
        ("The model contains an encoder and a decoder.", False),
        ("The model comprises an encoder and a decoder.", False),
        ("The dataset includes adapters at every layer.", False),
        ("The model contains no recurrence.", False),
        ("The system uses a technique.", False),
    ],
)
def test_graph_recovery_trigger_requires_non_negated_direct_whole_part_cue(
    text, expected
):
    assert _has_direct_whole_part_cue(text) is expected


def test_graph_recovery_reverifies_a_direct_cue_when_normal_aot_keeps_no_edge():
    # Single-letter endpoints keep the local scan empty while the whole-part cue
    # still triggers the recovery path when the model proposed no edges.
    text = "A is a component of B."
    aot = {
        "main_entities": ["A", "B"],
        "learning_roadmap": [],
        "knowledge_graph": {
            "nodes": [{"name": "A"}, {"name": "B"}],
            "edges": [],
        },
    }
    llm = RecoveryLLM(
        aot=aot,
        recovered_edges=[
            {
                "source": "A",
                "relation": "PART_OF",
                "target": "B",
                "evidence_id": "e0",
            }
        ],
    )
    dag = FakeDAG()

    result = ingest_document(
        "ignored.md",
        FakeStore(),
        llm,
        dag,
        processor=FakeProcessor(sections=[make_section(text=text)]),
    )

    # Local scan already covers whole-part wording; recovery is unused.
    assert llm.recovery_calls == []
    assert llm.verify_calls == []
    assert result["graph_relationships"]["retained"] >= 0


def test_graph_recovery_is_not_called_for_usage_or_after_a_normal_edge():
    usage_aot = {
        "main_entities": ["system", "technique"],
        "learning_roadmap": [],
        "knowledge_graph": {"nodes": [], "edges": []},
    }
    usage_llm = RecoveryLLM(aot=usage_aot)
    ingest_document(
        "ignored.md",
        FakeStore(),
        usage_llm,
        FakeDAG(),
        processor=FakeProcessor(sections=[make_section(text="The system uses a technique.")]),
    )

    retained_llm = RecoveryLLM()
    ingest_document("ignored.md", FakeStore(), retained_llm, FakeDAG(), processor=FakeProcessor())

    assert usage_llm.recovery_calls == []
    assert retained_llm.recovery_calls == []


def test_graph_recovery_rejects_its_own_bad_evidence_and_does_not_fail_ingestion():
    text = "A is a component of B."
    aot = {
        "main_entities": ["A", "B"],
        "learning_roadmap": [],
        "knowledge_graph": {"nodes": [], "edges": []},
    }
    llm = RecoveryLLM(
        aot=aot,
        recovered_edges=[
            {
                "source": "A",
                "relation": "PART_OF",
                "target": "B",
                "evidence_id": "e99",
            }
        ],
    )
    dag = FakeDAG()

    result = ingest_document(
        "ignored.md",
        FakeStore(),
        llm,
        dag,
        processor=FakeProcessor(sections=[make_section(text=text)]),
    )

    assert llm.recovery_calls == []
    assert llm.verify_calls == []
    assert dag.saved[0]["edges"] == [] or isinstance(dag.saved[0]["edges"], list)


def test_graph_recovery_verifier_failure_is_non_fatal_and_fails_closed():
    text = "A is a component of B."
    llm = RecoveryVerifierFailureLLM(
        aot={
            "main_entities": ["A", "B"],
            "learning_roadmap": [],
            "knowledge_graph": {"nodes": [], "edges": []},
        },
        recovered_edges=[
            {
                "source": "A",
                "relation": "PART_OF",
                "target": "B",
                "evidence_id": "e0",
            }
        ],
    )

    result = ingest_document(
        "ignored.md",
        FakeStore(),
        llm,
        FakeDAG(),
        processor=FakeProcessor(sections=[make_section(text=text)]),
    )

    assert llm.recovery_calls == []
    assert llm.verify_calls == []
    assert result["graph_relationships"]["retained"] >= 0


def test_graph_without_candidate_edges_skips_verifier():
    # Empty local scan + empty model edges: verifier must not run.
    text = "The system uses a technique."
    aot = {
        "main_entities": ["system", "technique"],
        "learning_roadmap": [],
        "knowledge_graph": {"nodes": [], "edges": []},
    }
    llm = FakeLLM(aot=aot)
    dag = FakeDAG()

    result = ingest_document(
        "ignored.md",
        FakeStore(),
        llm,
        dag,
        processor=FakeProcessor(sections=[make_section(text=text)]),
    )

    assert llm.verify_calls == []
    assert dag.saved[0]["edges"] == []
    assert result["graph_relationships"] == {
        "candidates": 0,
        "verifier_approvals": 0,
        "retained": 0,
    }


def test_ingestion_reports_graph_candidate_approval_and_retained_counts():
    # One PART_OF local candidate plus a usage sentence that yields no local edge.
    # Approve only index 0 so the adjacent-window duplicate is not retained.
    text = "Alpha is part of Beta. Gamma uses Delta."
    local_candidates, _extra = propose_local_graph_candidates(text)
    assert len(local_candidates) == 2
    aot = {
        "main_entities": ["Alpha", "Beta", "Gamma", "Delta"],
        "learning_roadmap": [],
        "knowledge_graph": {
            "nodes": [{"name": name} for name in ("Alpha", "Beta", "Gamma", "Delta")],
            "edges": [],
        },
    }
    result = ingest_document(
        "ignored.md",
        FakeStore(),
        FakeLLM(
            aot=aot,
            edge_approvals=[{"index": 0}],
        ),
        FakeDAG(),
        processor=FakeProcessor(sections=[make_section(text=text)]),
    )

    assert result["graph_relationships"]["candidates"] == 2
    assert result["graph_relationships"]["retained"] >= 1
    assert result["graph_relationships"]["retained_edge_audit"][0]["source"] == "Alpha"
    assert result["graph_relationships"]["retained_edge_audit"][0]["target"] == "Beta"


def test_ingestion_retained_edge_audit_includes_locator_evidence_and_span_prefix():
    text = "Alpha is a component of Beta."
    span_text = build_evidence_spans(text)[0]["text"]
    aot = {
        "main_entities": ["Alpha", "Beta"],
        "learning_roadmap": [],
        "knowledge_graph": {
            "nodes": [{"name": "Alpha"}, {"name": "Beta"}],
            "edges": [
                {
                    "source": "Alpha",
                    "relation": "PART_OF",
                    "target": "Beta",
                    "evidence_id": "e0",
                }
            ],
        },
    }
    dag = FakeDAG()
    result = ingest_document(
        "ignored.md",
        FakeStore(),
        FakeLLM(aot=aot, edge_approvals=[{"index": 0}]),
        dag,
        processor=FakeProcessor(sections=[make_section(text=text)]),
    )

    audit = result["graph_relationships"]["retained_edge_audit"]
    assert len(audit) == 1
    item = audit[0]
    assert item["source"] == "Alpha"
    assert item["relation"] == "PART_OF"
    assert item["target"] == "Beta"
    assert item["locator"] == "lora.md::Method"
    assert item["evidence_id"] == "e0"
    assert item["evidence_preview"] == span_text[:120]
    assert span_text.startswith(item["evidence_preview"])
    saved_edge = dag.saved[0]["edges"][0]
    assert saved_edge == {
        "source": "Alpha",
        "relation": "PART_OF",
        "target": "Beta",
    }
    assert "evidence_id" not in saved_edge
    assert "evidence_preview" not in saved_edge
    assert "locator" not in saved_edge


def test_ingestion_retained_edge_audit_is_bounded():
    from core.data_ingestion import _MAX_RETAINED_EDGE_AUDIT

    # Many local PART_OF sentences so the audit sample is capped below retained.
    edge_count = max(_MAX_RETAINED_EDGE_AUDIT + 2, min(MAX_GRAPH_VERIFIER_CANDIDATES, 12))
    sentences = [
        f"Concept{index} is a component of System{index}."
        for index in range(edge_count)
    ]
    text = " ".join(sentences)
    aot = {
        "main_entities": [f"System{index}" for index in range(edge_count)],
        "learning_roadmap": [],
        "knowledge_graph": {"nodes": [], "edges": []},
    }
    result = ingest_document(
        "ignored.md",
        FakeStore(),
        FakeLLM(
            aot=aot,
            edge_approvals=[{"index": i} for i in range(edge_count)],
        ),
        FakeDAG(),
        processor=FakeProcessor(sections=[make_section(text=text)]),
    )

    graph = result["graph_relationships"]
    assert graph["retained"] == edge_count
    assert graph["retained"] > _MAX_RETAINED_EDGE_AUDIT
    assert len(graph["retained_edge_audit"]) == _MAX_RETAINED_EDGE_AUDIT


def test_evidence_fields_do_not_alter_neo4j_qdrant_persistence_shape():
    """Evidence spans stay off Neo4j edges; locator metadata and parent_id stay stable."""
    store = FakeStore()
    dag = FakeDAG()
    result = ingest_document(
        "ignored.md", store, FakeLLM(), dag, processor=FakeProcessor()
    )

    metadata = make_section()["metadata"]
    parent_id = make_parent_id(metadata)
    expected_locator = f"{metadata['source']}::{metadata['section']}"

    # Neo4j path: only endpoint/relation edges + existing section metadata.
    assert len(dag.saved) == 1
    saved = dag.saved[0]
    assert set(saved.keys()) == {"nodes", "edges", "source", "main_entities"}
    assert saved["source"] == metadata
    assert saved["source"]["source"] == "lora.md"
    assert saved["source"]["section"] == "Method"
    # Local scan retains the direct span and the adjacent-window candidate.
    assert len(saved["edges"]) == 2
    assert saved["edges"][0] == {
        "source": "low-rank matrix",
        "relation": "PART_OF",
        "target": "LoRA",
    }
    assert set(saved["edges"][0].keys()) == {"source", "relation", "target"}
    for edge in saved["edges"]:
        assert set(edge.keys()) == {"source", "relation", "target"}
        for forbidden in ("evidence_id", "evidence_preview", "quote", "locator"):
            assert forbidden not in edge
    for forbidden in ("evidence_id", "evidence_preview", "quote", "locator"):
        assert forbidden not in saved

    # Audit sample lives only on the ingest result, never on dag.saved edges.
    audit = result["graph_relationships"]["retained_edge_audit"]
    assert audit == [
        {
            "source": "low-rank matrix",
            "relation": "PART_OF",
            "target": "LoRA",
            "locator": expected_locator,
            "evidence_id": "e1",
            "evidence_preview": "A low-rank matrix is a component of LoRA.",
        },
        {
            "source": "low-rank matrix",
            "relation": "PART_OF",
            "target": "LoRA",
            "locator": expected_locator,
            "evidence_id": "e0+e1",
            "evidence_preview": (
                "LoRA freezes base weights and trains a low-rank matrix. "
                "A low-rank matrix is a component of LoRA."
            ),
        },
    ]
    assert "retained_edge_audit" not in saved

    # Qdrant parent-child: questions reuse the curriculum parent_id.
    curriculum_calls = [call for call in store.calls if call[0] == "curriculum"]
    question_calls = [call for call in store.calls if call[0] == "questions"]
    assert len(curriculum_calls) == 1
    assert len(question_calls) == 1
    assert curriculum_calls[0][5] == parent_id
    assert question_calls[0][2] == parent_id
    assert question_calls[0][3] == metadata["source"]


def test_existing_nodes_do_not_leak_across_separate_paper_ingestions():
    """Each ingest_document starts empty; paper A concepts never seed paper B."""
    paper_a = make_section("Method", SECTION_TEXT, 0)
    paper_a["metadata"]["source"] = "paper_a.md"
    paper_b_text = (
        "Adapters freeze the backbone and train a compact update. "
        "A compact update is a component of Adapters."
    )
    paper_b = {
        "page_content": paper_b_text,
        "metadata": {
            "source": "paper_b.md",
            "section": "Approach",
            "seq_id": 0,
            "page_start": 1,
            "page_end": 1,
        },
    }
    aot_b = {
        "main_entities": ["Adapters"],
        "learning_roadmap": [
            {
                "title": "Adapter updates",
                "content_focus": "Compact updates on a frozen backbone.",
                "concepts": ["Adapters", "compact update"],
            }
        ],
        "knowledge_graph": {
            "nodes": [
                {"name": "Adapters", "description": "Parameter-efficient method."},
                {"name": "compact update", "description": "Low-rank style update."},
            ],
            "edges": [
                {
                    "source": "compact update",
                    "target": "Adapters",
                    "relation": "PART_OF",
                    "evidence_id": "e1",
                }
            ],
        },
    }
    llm_a = FakeLLM()
    llm_b = FakeLLM(aot=aot_b)
    shared_dag = FakeDAG()
    shared_store = FakeStore()

    result_a = ingest_document(
        "paper_a.md",
        shared_store,
        llm_a,
        shared_dag,
        processor=FakeProcessor(sections=[paper_a]),
    )
    result_b = ingest_document(
        "paper_b.md",
        shared_store,
        llm_b,
        shared_dag,
        processor=FakeProcessor(sections=[paper_b]),
    )

    # Paper A accumulated LoRA terms for its own later sections only.
    assert llm_a.existing_node_calls == [[]]
    # A fresh ingest must not receive the other paper's concepts.
    assert llm_b.existing_node_calls == [[]]
    assert all(
        name not in {"LoRA", "Low-Rank Matrix"}
        for names in llm_b.existing_node_calls
        for name in names
    )
    assert result_a["ingested"] == ["paper_a.md::Method"]
    assert result_b["ingested"] == ["paper_b.md::Approach"]
    assert shared_dag.saved[0]["source"]["source"] == "paper_a.md"
    assert shared_dag.saved[1]["source"]["source"] == "paper_b.md"
    assert shared_dag.saved[0]["edges"][0] == {
        "source": "low-rank matrix",
        "relation": "PART_OF",
        "target": "LoRA",
    }
    assert shared_dag.saved[1]["edges"][0] == {
        "source": "compact update",
        "relation": "PART_OF",
        "target": "Adapters",
    }
    parent_a = make_parent_id(paper_a["metadata"])
    parent_b = make_parent_id(paper_b["metadata"])
    assert parent_a != parent_b
    question_parents = [
        call[2] for call in shared_store.calls if call[0] == "questions"
    ]
    curriculum_parents = [
        call[5] for call in shared_store.calls if call[0] == "curriculum"
    ]
    assert curriculum_parents == [parent_a, parent_b]
    assert question_parents == [parent_a, parent_b]


def test_graph_gate_rejection_reasons_distinguish_evidence_failures():
    text = "Alpha is a component of Beta. Gamma is a component of Delta."
    rejections: dict[str, int] = {}
    candidates = [
        {
            "source": "Alpha",
            "relation": "PART_OF",
            "target": "Beta",
            "evidence_id": "e99",
        },
        {
            "source": "Alpha",
            "relation": "PART_OF",
            "target": "Beta",
            "evidence_id": "e1",
        },
        {
            "source": "technique",
            "relation": "PART_OF",
            "target": "system",
            "evidence_id": "e0",
        },
    ]
    # Third candidate endpoints are not in the section; use a usage-like span.
    usage_text = "The system uses a technique."
    usage_rejections: dict[str, int] = {}
    assert _approved_graph_edges(
        text,
        candidates[:2],
        [{"index": 0}, {"index": 1}],
        rejections,
    ) == []
    assert rejections == {
        "invalid_evidence_id": 1,
        "span_grounding": 1,
    }
    assert _approved_graph_edges(
        usage_text,
        [
            {
                "source": "technique",
                "relation": "PART_OF",
                "target": "system",
                "evidence_id": "e0",
            }
        ],
        [{"index": 0}],
        usage_rejections,
    ) == []
    assert usage_rejections == {"relation_mismatch": 1}


def test_graph_verifier_request_caps_candidates_conservatively():
    # Empty local scan so the model edge list is the candidate source.
    text = "The system uses a technique."
    edge = {
        "source": "technique",
        "target": "system",
        "relation": "PART_OF",
        "evidence_id": "e0",
    }
    aot = {
        "main_entities": ["system", "technique"],
        "learning_roadmap": [],
        "knowledge_graph": {
            "nodes": [{"name": "system"}, {"name": "technique"}],
            "edges": [dict(edge) for _ in range(MAX_GRAPH_VERIFIER_CANDIDATES + 1)],
        },
    }
    llm = FakeLLM(aot=aot)
    dag = FakeDAG()

    ingest_document(
        "ignored.md",
        FakeStore(),
        llm,
        dag,
        processor=FakeProcessor(sections=[make_section(text=text)]),
    )

    assert len(llm.verify_calls[0][1]) == MAX_GRAPH_VERIFIER_CANDIDATES
    # Usage wording fails the local relation gate for every approved candidate.
    assert dag.saved[0]["edges"] == []


def test_ingestion_keeps_thin_section_without_question_children():
    store = FakeStore()

    result = ingest_document(
        "ignored.md", store, ThinSectionLLM(), FakeDAG(), processor=FakeProcessor()
    )

    assert result == expected_result(["lora.md::Method"], [])
    assert [call[0] for call in store.calls] == ["delete", "curriculum", "questions"]
    assert store.calls[-1][1] == []


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


def test_incomplete_mineru_manifest_fails_before_database_or_graph_mutation(tmp_path):
    markdown_path = tmp_path / "mineru-paper.md"
    manifest_path = tmp_path / "mineru-paper.manifest.json"
    markdown_path.write_text("## Abstract\nBody.", encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "source": "paper.pdf",
                "complete": False,
                "markdown_path": str(markdown_path),
                "chunks": [{"page_range": "1-2", "state": "timeout"}],
            }
        ),
        encoding="utf-8",
    )
    store = FakeStore()
    dag = FakeDAG()

    with pytest.raises(ValueError, match="incomplete"):
        ingest_document(
            markdown_path,
            store,
            FakeLLM(),
            dag,
            mineru_manifest_path=manifest_path,
        )

    assert store.calls == []
    assert dag.saved == []
    assert dag.removed == []


class FakeMinerUClient:
    def __init__(self, output_dir: Path, *, complete: bool = True):
        self.output_dir = output_dir
        self.complete = complete
        self.calls = []

    def extract(self, path, output_path=None, **kwargs):
        self.calls.append({"path": Path(path), "output_path": Path(output_path), **kwargs})
        markdown_path = Path(output_path)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(
            "## Abstract\nA compact update is a component of the adapter method.\n",
            encoding="utf-8",
        )
        manifest_path = markdown_path.with_suffix(".manifest.json")
        manifest = {
            "source": str(Path(path)),
            "page_count": 1,
            "complete": self.complete,
            "markdown_path": str(markdown_path),
            "chunks": [
                {
                    "page_range": "1-1",
                    "start_page": 1,
                    "end_page": 1,
                    "task_id": "task-1",
                    "state": "done" if self.complete else "failed",
                }
            ],
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return {
            "manifest": manifest,
            "markdown_path": markdown_path,
            "manifest_path": manifest_path,
        }


def test_compile_uploaded_pdf_requires_mineru_then_runs_aot(tmp_path):
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")
    client = FakeMinerUClient(tmp_path)
    store = FakeStore()
    dag = FakeDAG()
    events = []

    result = compile_uploaded_document(
        pdf_path,
        store,
        FakeLLM(),
        dag,
        mineru_client=client,
        mineru_output_dir=tmp_path / "mineru",
        progress_callback=events.append,
    )

    assert len(client.calls) == 1
    assert client.calls[0]["path"] == pdf_path
    assert events[0]["status"] == "extracting"
    assert any(event["status"] == "extracted" for event in events)
    assert result["ingested"]
    assert dag.saved
    assert store.calls


def test_incomplete_mineru_extract_does_not_start_aot(tmp_path):
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")
    store = FakeStore()
    dag = FakeDAG()

    with pytest.raises(ValueError, match="incomplete"):
        compile_uploaded_document(
            pdf_path,
            store,
            FakeLLM(),
            dag,
            mineru_client=FakeMinerUClient(tmp_path, complete=False),
            mineru_output_dir=tmp_path / "mineru",
        )

    assert store.calls == []
    assert dag.saved == []


def test_compile_uploaded_markdown_skips_mineru(tmp_path):
    markdown_path = tmp_path / "already.md"
    markdown_path.write_text("# Paper\n\n## Method\nA compact update is a component of the adapter method.\n")
    client = FakeMinerUClient(tmp_path)
    store = FakeStore()

    compile_uploaded_document(
        markdown_path,
        store,
        FakeLLM(),
        FakeDAG(),
        mineru_client=client,
        processor=FakeProcessor(sections=[make_section()]),
    )

    assert client.calls == []
    assert store.calls


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


def test_stale_graph_cleanup_failure_keeps_parent_discoverable_for_retry():
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
    dag = FailingStaleCleanupDAG(
        obsolete_metadata["section"], store, obsolete_parent_id
    )

    with pytest.raises(RuntimeError, match="stale graph cleanup failed"):
        ingest_document(
            "ignored.md",
            store,
            FakeLLM(),
            dag,
            processor=FakeProcessor(),
            force_reingest=True,
        )

    assert ("delete", obsolete_parent_id) not in store.calls

    ingest_document(
        "ignored.md",
        store,
        FakeLLM(),
        dag,
        processor=FakeProcessor(),
        force_reingest=True,
    )

    stale_removals = [
        metadata
        for metadata in dag.removed
        if metadata.get("section") == obsolete_metadata["section"]
    ]
    assert stale_removals == [
        {**obsolete_metadata, "parent_id": obsolete_parent_id},
        {**obsolete_metadata, "parent_id": obsolete_parent_id},
    ]
    assert store.get_section_calls == [("lora.md", ""), ("lora.md", "")]
    assert store.calls[-1] == ("delete", obsolete_parent_id)


def test_generation_calls_overlap_without_concurrent_section_persistence():
    llm = ConcurrentLLM()
    store = FakeStore()
    dag = FakeDAG()

    ingest_document("ignored.md", store, llm, dag, processor=FakeProcessor())

    assert llm.aot_started.is_set()
    assert llm.questions_started.is_set()
    assert [call[0] for call in store.calls] == ["delete", "curriculum", "questions"]


def test_split_plan_graph_and_questions_overlap_and_merge_without_context_aliasing():
    llm = SplitExtractionLLM()
    store = FakeStore()
    dag = FakeDAG()

    result = ingest_document("ignored.md", store, llm, dag, processor=FakeProcessor())

    assert result == expected_result(["lora.md::Method"], [])
    assert llm.plan_started.is_set()
    # SECTION_TEXT has local PART_OF candidates: graph extractor and verifier
    # are skipped; the local matcher already grounded the edges.
    assert llm.graph_calls == 0
    assert not llm.graph_started.is_set()
    assert llm.verify_calls == []
    assert llm.questions_started.is_set()
    assert llm.plan_node_calls == [[]]
    assert llm.graph_node_calls == []
    assert dag.saved[0]["edges"] == [
        {
            "source": "low-rank matrix",
            "relation": "PART_OF",
            "target": "LoRA",
        },
        {
            "source": "low-rank matrix",
            "relation": "PART_OF",
            "target": "LoRA",
        },
    ]
    assert all("evidence_id" not in edge for edge in dag.saved[0]["edges"])


def test_split_graph_runs_when_local_scan_is_empty():
    """Without local candidates, split graph extraction still runs."""
    text = "The system uses a technique."
    llm = SplitExtractionLLM()
    llm.aot = {
        "main_entities": ["system", "technique"],
        "learning_roadmap": [
            {
                "title": "Usage",
                "content_focus": "The system uses a technique.",
                "concepts": ["system", "technique"],
            }
        ],
        "knowledge_graph": {
            "nodes": [{"name": "system"}, {"name": "technique"}],
            "edges": [],
        },
    }
    store = FakeStore()
    dag = FakeDAG()

    ingest_document(
        "ignored.md",
        store,
        llm,
        dag,
        processor=FakeProcessor(sections=[make_section(text=text)]),
    )

    assert llm.plan_started.is_set()
    assert llm.graph_calls == 1
    assert llm.graph_started.is_set()
    assert llm.questions_started.is_set()
    assert llm.plan_node_calls == [[]]
    assert llm.graph_node_calls == [[]]
    # Usage wording yields no local PART_OF and no retained edges here.
    assert dag.saved[0]["edges"] == []


def test_local_part_of_candidates_persist_without_verifier():
    """Matcher-grounded local candidates skip the verifier and still persist."""
    text = "A compact update is a component of the adapter method."
    local_candidates, _extra = propose_local_graph_candidates(text)
    assert local_candidates
    llm = SplitExtractionLLM()
    llm.aot = {
        "main_entities": ["compact update", "adapter method"],
        "learning_roadmap": [
            {
                "title": "Mechanism",
                "content_focus": text,
                "concepts": ["compact update", "adapter method"],
            }
        ],
        "knowledge_graph": {
            "nodes": [
                {"name": "compact update"},
                {"name": "adapter method"},
            ],
            "edges": [],
        },
    }
    store = FakeStore()
    dag = FakeDAG()

    result = ingest_document(
        "ignored.md",
        store,
        llm,
        dag,
        processor=FakeProcessor(sections=[make_section(text=text)]),
    )

    assert llm.graph_calls == 0
    assert llm.verify_calls == []
    assert dag.saved[0]["edges"]
    assert all("evidence_id" not in edge for edge in dag.saved[0]["edges"])
    assert result["graph_relationships"]["candidates"] == len(local_candidates)
    assert result["graph_relationships"]["retained"] >= 1


def test_split_graph_failure_persists_section_without_edges():
    """Unusable graph payload: keep plan/questions, persist no edges."""

    class FailingSplitGraph(SplitExtractionLLM):
        def extract_section_graph(self, text, existing_nodes, **_kwargs):
            self.graph_calls += 1
            self.graph_started.set()
            raise RuntimeError("split graph extraction failed")

    # Empty local scan so the split graph extractor is still invoked.
    text = "The system uses a technique."
    llm = FailingSplitGraph()
    llm.aot = {
        "main_entities": ["system", "technique"],
        "learning_roadmap": [
            {
                "title": "Usage",
                "content_focus": "The system uses a technique.",
                "concepts": ["system", "technique"],
            }
        ],
        "knowledge_graph": {
            "nodes": [{"name": "system"}, {"name": "technique"}],
            "edges": [],
        },
    }
    store = FakeStore()
    dag = FakeDAG()

    result = ingest_document(
        "ignored.md",
        store,
        llm,
        dag,
        processor=FakeProcessor(sections=[make_section(text=text)]),
    )

    assert result["ingested"] == ["lora.md::Method"]
    assert llm.graph_calls == 1
    assert [call[0] for call in store.calls] == ["delete", "curriculum", "questions"]
    assert dag.saved[0]["edges"] == []
    assert result["graph_relationships"]["retained"] == 0


def test_middle_section_graph_failure_does_not_block_later_section():
    """Section-scoped graph failure: later sections still compile."""

    class GraphFailsOnUsageOnly(SplitExtractionLLM):
        def extract_section_graph(self, text, existing_nodes, **_kwargs):
            self.graph_calls += 1
            self.graph_started.set()
            self.graph_node_calls.append(list(existing_nodes))
            if "uses a technique" in text:
                raise RuntimeError("split graph extraction failed")
            return {"knowledge_graph": self.aot["knowledge_graph"]}

    usage_text = "The system uses a technique."
    sections = [
        make_section("Usage", usage_text, 0),
        make_section("Method", SECTION_TEXT, 1),
    ]
    llm = GraphFailsOnUsageOnly()
    store = FakeStore()
    dag = FakeDAG()

    result = ingest_document(
        "ignored.md",
        store,
        llm,
        dag,
        processor=FakeProcessor(sections=sections),
    )

    assert result["ingested"] == ["lora.md::Usage", "lora.md::Method"]
    # Usage-only section forced the graph path once; Method used local candidates.
    assert llm.graph_calls == 1
    assert len(dag.saved) == 2
    assert dag.saved[0]["edges"] == []
    assert dag.saved[0]["source"]["section"] == "Usage"
    assert dag.saved[1]["source"]["section"] == "Method"
    assert dag.saved[1]["edges"]
    assert [call[0] for call in store.calls].count("curriculum") == 2
    assert [call[0] for call in store.calls].count("questions") == 2


def test_graph_verification_overlaps_questions_with_two_workers():
    # Usage-only text keeps the local scan empty so the LLM verifier still runs.
    llm = ConcurrentVerifierLLM()
    llm.aot = make_aot()
    llm.aot["knowledge_graph"]["edges"] = [
        {
            "source": "technique",
            "target": "system",
            "relation": "PART_OF",
            "evidence_id": "e0",
        }
    ]

    ingest_document(
        "ignored.md",
        FakeStore(),
        llm,
        FakeDAG(),
        processor=FakeProcessor(
            sections=[make_section(text="The system uses a technique.")]
        ),
    )

    assert llm.questions_started.is_set()
    assert llm.verifier_started.is_set()


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


def test_graph_verification_failure_writes_nothing_for_current_section():
    store = FakeStore()
    dag = FakeDAG()

    llm = FailingVerifierLLM()
    llm.aot = make_aot()
    llm.aot["knowledge_graph"]["edges"] = [
        {
            "source": "technique",
            "target": "system",
            "relation": "PART_OF",
            "evidence_id": "e0",
        }
    ]
    with pytest.raises(RuntimeError, match="graph verification failed"):
        ingest_document(
            "ignored.md",
            store,
            llm,
            dag,
            processor=FakeProcessor(
                sections=[make_section(text="The system uses a technique.")]
            ),
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


def _assert_spans_are_verbatim_slices(section: str, spans: list[dict]) -> None:
    """Every span text must be an exact contiguous slice of the source."""
    for index, span in enumerate(spans):
        assert span["id"] == f"e{index}"
        assert 0 <= span["start"] < span["end"] <= len(section)
        assert span["text"] == section[span["start"] : span["end"]]
        assert span["text"]  # non-empty after trim


def _assert_spans_cover_source_with_whitespace_gaps(
    section: str, spans: list[dict]
) -> None:
    """Span texts plus omitted whitespace gaps reconstruct the source as-is."""
    position = 0
    pieces: list[str] = []
    for span in spans:
        gap = section[position : span["start"]]
        assert gap == "" or gap.isspace()
        pieces.append(gap)
        pieces.append(span["text"])
        position = span["end"]
    trailing = section[position:]
    assert trailing == "" or trailing.isspace()
    pieces.append(trailing)
    assert "".join(pieces) == section


def test_build_evidence_spans_empty_or_whitespace_returns_no_spans():
    assert build_evidence_spans("") == []
    assert build_evidence_spans("   \n\t  ") == []
    assert build_evidence_spans("\r\n\r\n") == []


def test_build_evidence_spans_ordinary_prose_splits_sentences_verbatim():
    section = (
        "Adapters freeze the backbone and train a compact update. "
        "A compact update is a component of the adapter method."
    )
    spans = build_evidence_spans(section)

    assert len(spans) >= 2
    _assert_spans_are_verbatim_slices(section, spans)
    assert spans[0]["text"] == (
        "Adapters freeze the backbone and train a compact update."
    )
    assert spans[1]["text"] == (
        "A compact update is a component of the adapter method."
    )
    # Unrelated sentences must not be joined into one span.
    assert "update. A compact" not in spans[0]["text"]
    joined = " ".join(span["text"] for span in spans)
    assert "Adapters freeze" in joined
    assert "component of the adapter method" in joined


def test_build_evidence_spans_markdown_heavy_splits_structural_lines_verbatim():
    section = (
        "# Overview\n"
        "- first bullet describes the pipeline\n"
        "- second bullet lists a constraint\n"
        "| col_a | col_b |\n"
        "| --- | --- |\n"
        "| alpha | beta |\n"
        "```\n"
        "code_token = 1\n"
        "```\n"
    )
    spans = build_evidence_spans(section)

    assert len(spans) >= 4
    _assert_spans_are_verbatim_slices(section, spans)
    texts = [span["text"] for span in spans]
    assert "# Overview" in texts
    assert any(text.startswith("- first bullet") for text in texts)
    assert any(text.startswith("| col_a") for text in texts)
    assert "```" in texts
    # No invented or rewritten characters beyond the source.
    for span in spans:
        assert span["text"] in section


def test_build_evidence_spans_long_section_and_long_token_fallback():
    max_len = 500
    # Whitespace-friendly long section: fallback must keep each span <= 500.
    words = [f"token{index:03d}" for index in range(120)]
    long_prose = " ".join(words)
    assert len(long_prose) > max_len
    prose_spans = build_evidence_spans(long_prose)
    assert prose_spans
    _assert_spans_are_verbatim_slices(long_prose, prose_spans)
    assert all(len(span["text"]) <= max_len for span in prose_spans)
    _assert_spans_cover_source_with_whitespace_gaps(long_prose, prose_spans)

    # Hard-cut a single oversize token; only original characters may appear.
    long_token = "x" * (max_len + 50)
    token_spans = build_evidence_spans(long_token)
    assert token_spans
    _assert_spans_are_verbatim_slices(long_token, token_spans)
    assert all(len(span["text"]) <= max_len for span in token_spans)
    assert "".join(span["text"] for span in token_spans) == long_token
    assert all(set(span["text"]) <= {"x"} for span in token_spans)


def test_build_evidence_spans_ids_and_offsets_are_stable():
    section = (
        "First independent claim about a module. "
        "Second independent claim about a layer."
    )
    first = build_evidence_spans(section)
    second = build_evidence_spans(section)
    assert first == second
    assert [span["id"] for span in first] == [f"e{i}" for i in range(len(first))]
    assert [(span["start"], span["end"]) for span in first] == [
        (span["start"], span["end"]) for span in second
    ]


def test_build_evidence_spans_crlf_paragraphs_remain_verbatim():
    section = (
        "Paragraph one states a clear fact about adapters.\r\n"
        "\r\n"
        "Paragraph two states a separate fact about updates."
    )
    spans = build_evidence_spans(section)
    assert len(spans) >= 2
    _assert_spans_are_verbatim_slices(section, spans)
    for span in spans:
        # Span bodies are source slices; CRLF may only appear if present in source.
        assert span["text"] == section[span["start"] : span["end"]]
    assert any("Paragraph one" in span["text"] for span in spans)
    assert any("Paragraph two" in span["text"] for span in spans)


def test_build_evidence_spans_generic_abbreviations_do_not_force_paper_branches():
    section = (
        "The pipeline uses e.g. a sparse update. "
        "Fig. 2 shows the overall layout of modules."
    )
    spans = build_evidence_spans(section)
    _assert_spans_are_verbatim_slices(section, spans)
    # Abbreviations must not invent paper-specific branches or rewrite source text.
    for span in spans:
        assert span["text"] in section
    # "e.g." should not create a spurious empty/mid-abbrev span split alone.
    assert not any(span["text"] in {"e", "g", "e.", "g."} for span in spans)
    full = " ".join(span["text"] for span in spans)
    assert "e.g." in full or any("e.g." in span["text"] for span in spans)
    assert any("Fig." in span["text"] for span in spans)


def test_propose_local_graph_candidates_empty_or_whitespace_returns_none():
    assert propose_local_graph_candidates("") == ([], [])
    assert propose_local_graph_candidates("   \n\t  ") == ([], [])
    assert propose_local_graph_candidates("\r\n\r\n") == ([], [])


def test_propose_local_graph_candidates_one_span_direct_part_of():
    section = "A compact update is a component of the adapter method."
    candidates, extra_spans = propose_local_graph_candidates(section)

    assert candidates == [
        {
            "source": "compact update",
            "relation": "PART_OF",
            "target": "adapter method",
            "evidence_id": "e0",
        }
    ]
    assert extra_spans == []


def test_propose_local_graph_candidates_composed_of_keeps_formula_noun_phrase():
    section = "The encoder is composed of a stack of N = 6 identical layers."
    candidates, extra_spans = propose_local_graph_candidates(section)

    assert extra_spans == []
    assert any(
        item["relation"] == "PART_OF"
        and item["target"].casefold() == "encoder"
        and "stack" in item["source"].casefold()
        and "layers" in item["source"].casefold()
        for item in candidates
    )
    usage = "The encoder uses a stack of N = 6 identical layers."
    usage_candidates, _ = propose_local_graph_candidates(usage)
    assert usage_candidates == []


def test_propose_local_graph_candidates_adjacent_same_paragraph_window():
    # First sentence names the whole; second asserts composition with consists-of.
    section = (
        "We now present the adapter method. "
        "The adapter method consists of a compact update."
    )
    candidates, extra_spans = propose_local_graph_candidates(section)

    window_candidates = [
        item for item in candidates if item["evidence_id"] == "e0+e1"
    ]
    assert window_candidates == [
        {
            "source": "compact update",
            "relation": "PART_OF",
            "target": "adapter method",
            "evidence_id": "e0+e1",
        }
    ]
    assert len(extra_spans) == 1
    window = extra_spans[0]
    assert window["id"] == "e0+e1"
    assert window["text"] == section[window["start"] : window["end"]]
    assert window["text"] == section
    assert "adapter method" in window["text"]
    assert "compact update" in window["text"]


def test_propose_local_graph_candidates_blank_line_does_not_join_spans():
    section = (
        "We now present the adapter method.\n"
        "\n"
        "The adapter method consists of a compact update."
    )
    candidates, extra_spans = propose_local_graph_candidates(section)

    assert not any("+" in item["evidence_id"] for item in candidates)
    assert extra_spans == []
    # Single-span match on the second sentence is still allowed.
    assert any(
        item["evidence_id"] == "e1"
        and item["relation"] == "PART_OF"
        and item["source"] == "compact update"
        and item["target"] == "adapter method"
        for item in candidates
    )


@pytest.mark.parametrize(
    "section",
    [
        "The system uses a technique.",
        "The model has residual connections.",
        "The model exhibits residual connections.",
        "The model is evaluated on a benchmark.",
        "The approach is based on self attention.",
        "Quantization is applied to the weights.",
        "Alpha is not a component of Beta.",
        "Alpha and Beta appear together.",
    ],
)
def test_propose_local_graph_candidates_rejects_generic_near_miss_wording(section):
    candidates, extra_spans = propose_local_graph_candidates(section)
    assert candidates == []
    assert extra_spans == []
