from runtime.engine import MAX_EVIDENCE_CHARS_PER_SECTION, RuntimeEngine, build_sources


class FakeDB:
    def __init__(self, sections):
        self.sections = sections
        self.search_args = None

    def search_candidates_and_fetch_parent(self, **kwargs):
        self.search_args = kwargs
        return self.sections

    def get_section_exact(self, target_file, target_section):
        return self.sections

    def get_roadmap(self, parent_id):
        return [
            {"seq_id": 0, "title": "Motivation", "concepts": ["LoRA"]},
            {"seq_id": 1, "title": "Mechanism", "concepts": ["Matrix"]},
        ]


class FakeDAG:
    def __init__(self, graph_context=None):
        self.calls = []
        self.graph_context = graph_context or [
            {"source": "Matrix", "relation": "PREREQUISITE_OF", "target": "LoRA"}
        ]

    def get_graph_context(self, concepts, search_mode, source=""):
        self.calls.append((concepts, search_mode, source))
        return self.graph_context


class FakeLLM:
    def __init__(self):
        self.answer_args = None
        self.teach_args = []

    def answer(self, **kwargs):
        self.answer_args = kwargs
        return "Grounded answer"

    def teach_step(self, **kwargs):
        self.teach_args.append(kwargs)
        return f"Lesson: {kwargs['roadmap_step']['title']}"


def section():
    return {
        "page_content": "LoRA freezes base weights.",
        "metadata": {
            "source": "lora.pdf",
            "section": "Method",
            "page_start": 4,
            "page_end": 5,
            "parent_id": "parent-1",
            "anchor_nodes": ["LoRA", "Matrix"],
        },
    }


def test_ask_returns_answer_graph_context_and_sources():
    llm = FakeLLM()
    dag = FakeDAG()
    db = FakeDB([section()])

    result = RuntimeEngine(llm, db, dag).ask("How does LoRA work?", "lora.pdf")

    assert result == {
        "answer": "Grounded answer",
        "sources": ["[lora.pdf — Method, p.4–5]"],
        "graph_context": [
            {"source": "Matrix", "relation": "PREREQUISITE_OF", "target": "LoRA"}
        ],
    }
    assert db.search_args["target_file"] == "lora.pdf"
    assert dag.calls == [(["LoRA", "Matrix"], "search", "lora.pdf")]
    assert llm.answer_args["sections"] == [section()]


def test_ask_reports_missing_sources_without_calling_llm():
    llm = FakeLLM()
    result = RuntimeEngine(llm, FakeDB([]), FakeDAG()).ask("Unknown question")

    assert result == {
        "answer": "No relevant source found.",
        "sources": [],
        "graph_context": [],
    }
    assert llm.answer_args is None


def test_ask_bounds_evidence_and_graph_context_before_answering():
    oversized_section = section()
    oversized_section["page_content"] = "start " + ("middle " * 2_000) + "end"
    oversized_graph = [
        {"source": str(index), "description": "x" * 1_000}
        for index in range(20)
    ]
    llm = FakeLLM()
    dag = FakeDAG(oversized_graph)

    result = RuntimeEngine(llm, FakeDB([oversized_section]), dag).ask("Explain LoRA")

    evidence = llm.answer_args["sections"][0]["page_content"]
    assert len(evidence) <= MAX_EVIDENCE_CHARS_PER_SECTION
    assert evidence.startswith("start")
    assert evidence.endswith("end")
    assert len(llm.answer_args["graph_context"]) < len(oversized_graph)
    assert result["graph_context"] == llm.answer_args["graph_context"]
    assert dag.calls == [(["LoRA", "Matrix"], "search", "")]


def test_teach_uses_original_section_and_step_concepts():
    llm = FakeLLM()
    dag = FakeDAG()
    lessons = RuntimeEngine(llm, FakeDB([section()]), dag).teach_section("lora.pdf", "Method")

    assert [lesson["content"] for lesson in lessons] == [
        "Lesson: Motivation",
        "Lesson: Mechanism",
    ]
    assert dag.calls == [
        (["LoRA"], "semi_search", "lora.pdf"),
        (["Matrix"], "semi_search", "lora.pdf"),
    ]
    assert all(call["section_text"] == "LoRA freezes base weights." for call in llm.teach_args)


def test_source_formatting_deduplicates_repeated_sections():
    assert build_sources([section(), section()]) == ["[lora.pdf — Method, p.4–5]"]
