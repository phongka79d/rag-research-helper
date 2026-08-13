import threading

import pytest

from core.data_ingestion import (
    _approved_graph_edges,
    _has_direct_whole_part_cue,
    collect_anchor_nodes,
    filter_aot_to_section,
    ingest_document,
    make_content_hash,
    make_parent_id,
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
    def __init__(self, aot: dict | None = None, edge_approvals: list[dict] | None = None):
        self.aot = aot or make_aot()
        self.edge_approvals = edge_approvals
        self.aot_text = None
        self.question_text = None
        self.section_title = None
        self.existing_node_calls: list[list[str]] = []
        self.verify_calls = []

    def extract_section_plan_and_graph(self, text, existing_nodes):
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

    def verify_graph_edges(self, section_text, candidates):
        self.verify_calls.append((section_text, candidates))
        if self.edge_approvals is not None:
            return self.edge_approvals
        return [
            {"index": index, "quote": section_text}
            for index, _candidate in enumerate(candidates)
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

    def generate_hypothetical_questions(self, text, num_questions, section_title=""):
        self.questions_started.set()
        assert self.aot_started.wait(2)
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
    def verify_graph_edges(self, section_text, candidates):
        raise RuntimeError("graph verification failed")


class RecoveryLLM(FakeLLM):
    def __init__(self, *args, recovered_edges=None, recovery_error=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.recovered_edges = recovered_edges or []
        self.recovery_error = recovery_error
        self.recovery_calls = []

    def extract_graph_edges_with_evidence(self, text, existing_nodes):
        self.recovery_calls.append((text, list(existing_nodes)))
        if self.recovery_error is not None:
            raise self.recovery_error
        return self.recovered_edges


class RecoveryVerifierFailureLLM(RecoveryLLM):
    def verify_graph_edges(self, section_text, candidates):
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

    def verify_graph_edges(self, section_text, candidates):
        self.verifier_started.set()
        assert self.questions_started.wait(2)
        return super().verify_graph_edges(section_text, candidates)


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
        graph_relationships = {
            "candidates": int(bool(ingested)),
            "verifier_approvals": int(bool(ingested)),
            "retained": int(bool(ingested)),
        }
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
    assert llm.section_title == "Method"
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


def test_graph_edges_require_valid_grounded_verifier_approval():
    text = "Alpha is part of Beta. Beta is related to Gamma."
    edges = [
        {"source": "Alpha", "target": "Beta", "relation": "PART_OF"},
        {"source": "Beta", "target": "Gamma", "relation": "RELATES_TO"},
        {"source": "Alpha", "target": "Alpha", "relation": "RELATES_TO"},
        {"source": "Alpha", "target": "Gamma", "relation": "RELATES_TO"},
        {"source": "Alpha", "target": "Gamma", "relation": "DESCRIBES"},
    ]
    aot = {
        "main_entities": ["Alpha", "Beta", "Gamma"],
        "learning_roadmap": [],
        "knowledge_graph": {
            "nodes": [{"name": name} for name in ("Alpha", "Beta", "Gamma")],
            "edges": edges,
        },
    }
    approvals = [
        {"index": "0", "quote": "Alpha is part of Beta."},
        {"index": True, "quote": "Beta is related to Gamma."},
        {"index": -1, "quote": "Alpha is part of Beta."},
        {"index": 9, "quote": "Alpha is part of Beta."},
        {"index": 0, "quote": "Alpha appears elsewhere."},
        {"index": 0, "quote": "Alpha is part of Beta."},
        {"index": 1, "quote": "Beta is related to Gamma."},
        {"index": 2, "quote": "Alpha is part of Beta."},
        {"index": 3, "quote": "Alpha directly relates to Gamma."},
        {"index": 4, "quote": "Alpha is part of Beta."},
    ]
    llm = FakeLLM(aot=aot, edge_approvals=approvals)
    dag = FakeDAG()

    ingest_document(
        "ignored.md",
        FakeStore(),
        llm,
        dag,
        processor=FakeProcessor(sections=[make_section(text=text)]),
    )

    assert llm.verify_calls == [(text, edges)]
    assert dag.saved[0]["edges"] == [edges[1]]
    assert "quote" not in dag.saved[0]["edges"][0]


def test_graph_quote_gate_rejects_generic_usage_property_and_evaluation_claims():
    text = (
        "The Transformer uses self-attention. The Transformer relies on attention "
        "mechanisms. The Transformer has positional encodings. The model is evaluated "
        "on a benchmark."
    )
    candidates = [
        {"source": "self-attention", "relation": "PART_OF", "target": "Transformer"},
        {
            "source": "attention mechanisms",
            "relation": "PREREQUISITE_OF",
            "target": "Transformer",
        },
        {
            "source": "Transformer",
            "relation": "DESCRIBES",
            "target": "positional encodings",
        },
        {"source": "model", "relation": "RELATES_TO", "target": "benchmark"},
    ]
    approvals = [{"index": index, "quote": text.split(". ")[index] + "."} for index in range(4)]

    assert _approved_graph_edges(text, candidates, approvals) == []


@pytest.mark.parametrize(
    ("text", "candidate"),
    [
        (
            "Multi-head attention is a component of the Transformer.",
            {
                "source": "Multi-head attention",
                "relation": "PART_OF",
                "target": "Transformer",
            },
        ),
        (
            "Understanding dot-product attention is required before understanding multi-head attention.",
            {
                "source": "dot-product attention",
                "relation": "PREREQUISITE_OF",
                "target": "multi-head attention",
            },
        ),
        (
            "The attention mechanism explains token dependencies.",
            {
                "source": "attention mechanism",
                "relation": "DESCRIBES",
                "target": "token dependencies",
            },
        ),
        (
            "Positional encodings are related to token order.",
            {
                "source": "Positional encodings",
                "relation": "RELATES_TO",
                "target": "token order",
            },
        ),
    ],
)
def test_graph_quote_gate_keeps_direct_directional_relation_evidence(text, candidate):
    assert _approved_graph_edges(text, [candidate], [{"index": 0, "quote": text}]) == [
        candidate
    ]


def test_graph_quote_gate_binds_whole_part_evidence_to_candidate_endpoints():
    text = "Transformer uses self-attention. The encoder contains self-attention."
    candidate = {
        "source": "self-attention",
        "relation": "PART_OF",
        "target": "Transformer",
    }

    assert _approved_graph_edges(text, [candidate], [{"index": 0, "quote": text}]) == []


@pytest.mark.parametrize(
    ("text", "candidate"),
    [
        (
            "Each of the layers in our encoder and decoder contains a fully connected feed-forward network.",
            {
                "source": "fully connected feed-forward network",
                "relation": "PART_OF",
                "target": "layers in our encoder and decoder",
            },
        ),
        (
            "The encoder is composed of a stack of N identical layers.",
            {
                "source": "a stack of N identical layers",
                "relation": "PART_OF",
                "target": "encoder",
            },
        ),
        (
            "The model consists of an encoder and a decoder.",
            {"source": "encoder", "relation": "PART_OF", "target": "model"},
        ),
    ],
)
def test_graph_quote_gate_keeps_natural_direct_whole_part_evidence(text, candidate):
    assert _approved_graph_edges(text, [candidate], [{"index": 0, "quote": text}]) == [
        candidate
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
    text = "The encoder is composed of a stack."
    aot = {
        "main_entities": ["encoder", "stack"],
        "learning_roadmap": [],
        "knowledge_graph": {
            "nodes": [{"name": "encoder"}, {"name": "stack"}],
            "edges": [],
        },
    }
    llm = RecoveryLLM(
        aot=aot,
        recovered_edges=[
            {
                "source": "stack",
                "relation": "PART_OF",
                "target": "encoder",
                "quote": text,
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

    assert len(llm.recovery_calls) == 1
    assert llm.verify_calls == [(text, [{"source": "stack", "relation": "PART_OF", "target": "encoder"}])]
    assert dag.saved[0]["edges"] == [
        {"source": "stack", "relation": "PART_OF", "target": "encoder"}
    ]
    assert result["graph_relationships"] == {
        "candidates": 1,
        "verifier_approvals": 1,
        "retained": 1,
    }


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
    text = "The encoder is composed of a stack."
    aot = {
        "main_entities": ["encoder", "stack"],
        "learning_roadmap": [],
        "knowledge_graph": {"nodes": [], "edges": []},
    }
    llm = RecoveryLLM(
        aot=aot,
        recovered_edges=[
            {
                "source": "stack",
                "relation": "PART_OF",
                "target": "encoder",
                "quote": "The encoder uses a stack.",
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

    assert len(llm.recovery_calls) == 1
    assert llm.verify_calls == []
    assert dag.saved[0]["edges"] == []
    assert result["graph_relationships"] == {
        "candidates": 0,
        "verifier_approvals": 0,
        "retained": 0,
    }


def test_graph_recovery_verifier_failure_is_non_fatal_and_fails_closed():
    text = "The encoder is composed of a stack."
    llm = RecoveryVerifierFailureLLM(
        aot={
            "main_entities": ["encoder", "stack"],
            "learning_roadmap": [],
            "knowledge_graph": {"nodes": [], "edges": []},
        },
        recovered_edges=[
            {
                "source": "stack",
                "relation": "PART_OF",
                "target": "encoder",
                "quote": text,
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

    assert len(llm.recovery_calls) == 1
    assert len(llm.verify_calls) == 1
    assert result["graph_relationships"] == {
        "candidates": 1,
        "verifier_approvals": 0,
        "retained": 0,
    }


def test_graph_without_candidate_edges_skips_verifier():
    aot = make_aot()
    aot["knowledge_graph"]["edges"] = []
    llm = FakeLLM(aot=aot)
    dag = FakeDAG()

    result = ingest_document("ignored.md", FakeStore(), llm, dag, processor=FakeProcessor())

    assert llm.verify_calls == []
    assert dag.saved[0]["edges"] == []
    assert result["graph_relationships"] == {
        "candidates": 0,
        "verifier_approvals": 0,
        "retained": 0,
    }


def test_ingestion_reports_graph_candidate_approval_and_retained_counts():
    text = "Alpha is part of Beta. Gamma uses Delta."
    aot = {
        "main_entities": ["Alpha", "Beta", "Gamma", "Delta"],
        "learning_roadmap": [],
        "knowledge_graph": {
            "nodes": [{"name": name} for name in ("Alpha", "Beta", "Gamma", "Delta")],
            "edges": [
                {"source": "Alpha", "relation": "PART_OF", "target": "Beta"},
                {"source": "Delta", "relation": "PART_OF", "target": "Gamma"},
            ],
        },
    }
    result = ingest_document(
        "ignored.md",
        FakeStore(),
        FakeLLM(
            aot=aot,
            edge_approvals=[
                {"index": 0, "quote": "Alpha is part of Beta."},
                {"index": 1, "quote": "Gamma uses Delta."},
            ],
        ),
        FakeDAG(),
        processor=FakeProcessor(sections=[make_section(text=text)]),
    )

    assert result["graph_relationships"] == {
        "candidates": 2,
        "verifier_approvals": 2,
        "retained": 1,
    }


def test_graph_verifier_request_caps_candidates_conservatively():
    aot = make_aot()
    edge = aot["knowledge_graph"]["edges"][0]
    aot["knowledge_graph"]["edges"] = [
        dict(edge) for _ in range(MAX_GRAPH_VERIFIER_CANDIDATES + 1)
    ]
    llm = FakeLLM(aot=aot)
    dag = FakeDAG()

    ingest_document("ignored.md", FakeStore(), llm, dag, processor=FakeProcessor())

    assert len(llm.verify_calls[0][1]) == MAX_GRAPH_VERIFIER_CANDIDATES
    assert len(dag.saved[0]["edges"]) == MAX_GRAPH_VERIFIER_CANDIDATES


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


def test_graph_verification_overlaps_questions_with_two_workers():
    llm = ConcurrentVerifierLLM()

    ingest_document(
        "ignored.md", FakeStore(), llm, FakeDAG(), processor=FakeProcessor()
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

    with pytest.raises(RuntimeError, match="graph verification failed"):
        ingest_document(
            "ignored.md", store, FailingVerifierLLM(), dag, processor=FakeProcessor()
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
