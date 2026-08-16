import json
from types import SimpleNamespace

import pytest
from qdrant_client import models

import evaluate
from evaluate import (
    effective_retrieval_limits,
    expected_parent_ids,
    retrieval_metrics,
    runtime_metrics,
    sample_stored_graph_edges,
    summarize_method,
    unique_visual_graph_edges,
)


def test_effective_retrieval_limits_report_store_capacity_and_caps():
    limits = effective_retrieval_limits(
        SimpleNamespace(_search_limit=100, _max_candidate_parents=100)
    )

    assert limits == {
        "qdrant_search_limit": 25,
        "max_candidate_parents": 5,
        "final_parent_limit": 2,
    }


def test_retrieval_metrics_measure_first_relevant_rank_and_recall_at_five():
    metrics = retrieval_metrics(
        rankings=[["wrong", "right"], ["one", "two", "three", "four", "hit"], []],
        expected_ids=[{"right"}, {"hit"}, {"missing"}],
    )

    assert metrics["recall_at_5"] == pytest.approx(2 / 3)
    assert metrics["mrr"] == pytest.approx((1 / 2 + 1 / 5) / 3)


def test_expected_parent_ids_use_ingestion_identifier():
    expected = expected_parent_ids(
        {"expected": [{"source": "demo_lora.md", "section": "Method"}]}
    )

    assert expected == {"4aa2d587f9e3aa3844e811ca996f82ec"}


def test_graph_summary_preserves_retrieval_metrics_and_reports_context():
    result = summarize_method([["parent"]], [{"parent"}], graph_edges=[3])

    assert result == {"recall_at_5": 1.0, "mrr": 1.0, "average_graph_edges": 3}


def test_unique_visual_graph_edges_dedupes_and_accepts_label_aliases():
    edges = unique_visual_graph_edges(
        {
            "edges": [
                {"source": "Attention", "label": "PART_OF", "target": "Transformer"},
                {"source": "Attention", "label": "PART_OF", "target": "Transformer"},
                {"source": "LoRA", "relation": "USES", "target": "low-rank adapters"},
                {"source": "", "label": "USES", "target": "x"},
                {"source": "A", "type": "ENABLES", "target": "B"},
            ]
        }
    )

    assert edges == [
        ("Attention", "PART_OF", "Transformer"),
        ("LoRA", "USES", "low-rank adapters"),
        ("A", "ENABLES", "B"),
    ]


def test_sample_stored_graph_edges_uses_source_filter_and_uniquifies():
    class FakeDAG:
        def __init__(self):
            self.calls: list[dict[str, str | None]] = []

        def get_visual_graph(self, locator=None, *, source=None):
            self.calls.append({"locator": locator, "source": source})
            if source == "a.pdf":
                return {
                    "edges": [
                        {"source": "X", "label": "USES", "target": "Y"},
                        {"source": "X", "label": "USES", "target": "Y"},
                    ]
                }
            if source == "b.pdf":
                return {
                    "edges": [
                        {"source": "X", "label": "USES", "target": "Y"},
                        {"source": "P", "label": "PART_OF", "target": "Q"},
                    ]
                }
            return {
                "edges": [
                    {"source": "All", "label": "DESCRIBES", "target": "Graph"},
                ]
            }

    dag = FakeDAG()

    assert sample_stored_graph_edges(dag) == [("All", "DESCRIBES", "Graph")]
    assert dag.calls[-1] == {"locator": None, "source": None}

    assert sample_stored_graph_edges(dag, ["a.pdf", "b.pdf"]) == [
        ("X", "USES", "Y"),
        ("P", "PART_OF", "Q"),
    ]
    assert dag.calls[-2:] == [
        {"locator": None, "source": "a.pdf"},
        {"locator": None, "source": "b.pdf"},
    ]


def test_runtime_metrics_measure_parent_coverage_sources_latency_and_mrr():
    metrics = runtime_metrics(
        rankings=[["expected-b", "expected-a"], ["wrong", "expected-c"], []],
        expected_ids=[{"expected-a", "expected-b"}, {"expected-c", "missing"}, {"none"}],
        returned_sources=[
            ["papers/b.pdf", "a.pdf"],
            ["wrong.pdf", "c.pdf"],
            [],
        ],
        expected_sources=[{"a.pdf", "b.pdf"}, {"c.pdf", "missing.pdf"}, {"none.pdf"}],
        retrieval_latencies=[1.0, 2.0, 3.0],
    )

    assert metrics == {
        "recall_at_2": pytest.approx((1.0 + 0.5 + 0.0) / 3),
        "mrr": pytest.approx((1.0 + 0.5 + 0.0) / 3),
        "all_expected_sources_rate": pytest.approx(1 / 3),
        "average_retrieval_latency_seconds": pytest.approx(2.0),
    }


def test_parent_baseline_applies_optional_source_filter():
    class FakeClient:
        def query_points(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(
                points=[SimpleNamespace(payload={"parent_id": "parent"})]
            )

    client = FakeClient()
    db = SimpleNamespace(client=client, curriculum_collection="curriculum")

    assert evaluate.parent_baseline(db, [1.0], "paper.pdf") == ["parent"]
    conditions = client.kwargs["query_filter"].must
    assert client.kwargs["limit"] == 5
    assert any(
        isinstance(condition, models.FieldCondition)
        and condition.key == "source"
        and condition.match.value == "paper.pdf"
        for condition in conditions
    )


def test_evaluate_case_uses_runtime_retrieval_sections_for_metrics_and_graph(
    monkeypatch,
):
    class FakeDB:
        def search_candidates_and_fetch_parent(self, **kwargs):
            self.search_kwargs = kwargs
            return [
                {
                    "metadata": {
                        "parent_id": "selected-parent",
                        "source": "paper.pdf",
                        "anchor_nodes": ["Attention", "Transformer", "Attention"],
                    }
                }
            ]

    class FakeDAG:
        def get_graph_context(self, anchors, search_mode, source):
            self.call = (anchors, search_mode, source)
            return [{"relation": "DESCRIBES"}]

    db = FakeDB()
    dag = FakeDAG()
    llm = object()
    monkeypatch.setattr(
        evaluate, "parent_baseline", lambda db, vector, target_file="": ["baseline"]
    )
    ticks = iter([10.0, 10.25])
    monkeypatch.setattr(evaluate, "perf_counter", lambda: next(ticks))

    result = evaluate.evaluate_case(
        "How does attention work?", [1.0], db, llm, dag, "paper.pdf"
    )

    assert result[:5] == (
        ["baseline"],
        ["selected-parent"],
        ["paper.pdf"],
        1,
        pytest.approx(0.25),
    )
    assert result[5] in ("jina", "llm_fallback", "llm", "vector")
    assert db.search_kwargs == {
        "query": "How does attention work?",
        "llm_service": llm,
        "target_file": "paper.pdf",
        "query_vector": [1.0],
    }
    assert dag.call == (["Attention", "Transformer"], "search", "paper.pdf")


def test_run_evaluation_records_shared_runtime_rankings(monkeypatch, tmp_path):
    case = {
        "category": "factual",
        "query": "What stays frozen?",
        "expected": [{"source": "demo_lora.md", "section": "Method"}],
    }
    dataset_path = tmp_path / "eval.json"
    dataset_path.write_text(json.dumps([case] * 20), encoding="utf-8")
    expected_id = next(evaluate.expected_parent_ids(case).__iter__())

    class FakeLLM:
        def embed(self, query):
            return [1.0]

        def embed_many(self, queries):
            return [[1.0] for _ in queries]

    class FakeDAG:
        def verify_connection(self):
            pass

        def get_graph_context(self, anchors, search_mode, source):
            return []

        def close(self):
            pass

    monkeypatch.setattr(
        evaluate, "Settings", lambda: SimpleNamespace(validate=lambda: None)
    )
    monkeypatch.setattr(evaluate, "LLMService", lambda settings: FakeLLM())
    class FakeDB:
        def search_candidates_and_fetch_parent(self, **kwargs):
            assert kwargs["target_file"] == "demo_lora.md"
            assert kwargs["query_vector"] == [1.0]
            return [
                {
                    "metadata": {
                        "parent_id": expected_id,
                        "source": "demo_lora.md",
                        "anchor_nodes": [],
                    }
                }
            ]

    monkeypatch.setattr(evaluate, "QdrantVectorStore", lambda settings, llm: FakeDB())
    monkeypatch.setattr(evaluate, "Neo4jManager", lambda settings: FakeDAG())
    monkeypatch.setattr(
        evaluate,
        "parent_baseline",
        lambda db, vector, target_file="": [],
    )

    case["target_file"] = "demo_lora.md"
    dataset_path.write_text(json.dumps([case] * 20), encoding="utf-8")

    result = evaluate.run_evaluation(dataset_path)

    runtime = result["hyde_question_rerank"]
    assert runtime["recall_at_2"] == 1.0
    assert runtime["mrr"] == 1.0
    assert runtime["all_expected_sources_rate"] == 1.0
    assert runtime["average_retrieval_latency_seconds"] >= 0.0
    assert runtime["rerank_source_rate"] == {
        "jina": 0.0,
        "llm_fallback": 0.0,
        "llm": 0.0,
        "vector": 1.0,
    }
    assert runtime["effective_retrieval_limits"] == {
        "qdrant_search_limit": 25,
        "max_candidate_parents": 5,
        "final_parent_limit": 2,
    }
    assert "recall_at_5" not in runtime
    assert result["parent_section_vector_baseline"] == {
        "recall_at_5": 0.0,
        "mrr": 0.0,
    }
    assert result["question_count"] == 20
