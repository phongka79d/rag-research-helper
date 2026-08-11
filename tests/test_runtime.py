from runtime.engine import RuntimeEngine, build_sources


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
    def __init__(self):
        self.calls = []

    def get_graph_context(self, concepts, search_mode):
        self.calls.append((concepts, search_mode))
        return [{"source": "Matrix", "relation": "PREREQUISITE_OF", "target": "LoRA"}]


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
    assert dag.calls == [(["LoRA", "Matrix"], "search")]
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


def test_teach_uses_original_section_and_step_concepts():
    llm = FakeLLM()
    dag = FakeDAG()
    lessons = RuntimeEngine(llm, FakeDB([section()]), dag).teach_section("lora.pdf", "Method")

    assert [lesson["content"] for lesson in lessons] == [
        "Lesson: Motivation",
        "Lesson: Mechanism",
    ]
    assert [call[1] for call in dag.calls] == ["semi_search", "semi_search"]
    assert all(call["section_text"] == "LoRA freezes base weights." for call in llm.teach_args)


def test_source_formatting_deduplicates_repeated_sections():
    assert build_sources([section(), section()]) == ["[lora.pdf — Method, p.4–5]"]
