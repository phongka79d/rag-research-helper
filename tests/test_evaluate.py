import json
from types import SimpleNamespace

import pytest

import evaluate
from evaluate import expected_parent_ids, retrieval_metrics, summarize_method


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


def test_run_evaluation_records_hyde_rankings(monkeypatch, tmp_path):
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

        def get_graph_context(self, anchors, search_mode):
            return []

        def close(self):
            pass

    monkeypatch.setattr(
        evaluate, "Settings", lambda: SimpleNamespace(validate=lambda: None)
    )
    monkeypatch.setattr(evaluate, "LLMService", lambda settings: FakeLLM())
    monkeypatch.setattr(evaluate, "QdrantVectorStore", lambda settings, llm: object())
    monkeypatch.setattr(evaluate, "Neo4jManager", lambda settings: FakeDAG())
    monkeypatch.setattr(evaluate, "parent_baseline", lambda db, vector: [])
    monkeypatch.setattr(
        evaluate,
        "hyde_rerank",
        lambda db, llm, query, vector: ([expected_id], []),
    )
    monkeypatch.setattr(evaluate, "anchors_for_parents", lambda db, parent_ids: [])

    result = evaluate.run_evaluation(dataset_path)

    assert result["hyde_question_rerank"] == {"recall_at_5": 1.0, "mrr": 1.0}
