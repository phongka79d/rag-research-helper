"""Run minimal retrieval ablations over data/eval.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from qdrant_client import models

from config.settings import Settings
from core.data_ingestion import make_parent_id
from database.semantic_dag import Neo4jManager
from database.structural_db import QdrantVectorStore
from orchestrator.llm_service import LLMService

DATASET_PATH = Path("data/eval.json")
RESULTS_PATH = Path("eval_results.json")


def expected_parent_ids(case: dict[str, Any]) -> set[str]:
    return {
        make_parent_id({"source": item["source"], "section": item["section"]})
        for item in case["expected"]
    }


def retrieval_metrics(
    rankings: list[list[str]], expected_ids: list[set[str]]
) -> dict[str, float]:
    if not rankings:
        return {"recall_at_5": 0.0, "mrr": 0.0}
    recall_hits = 0
    reciprocal_ranks = []
    for ranked_ids, expected in zip(rankings, expected_ids):
        first_match = next(
            (index for index, parent_id in enumerate(ranked_ids, start=1) if parent_id in expected),
            None,
        )
        recall_hits += int(any(parent_id in expected for parent_id in ranked_ids[:5]))
        reciprocal_ranks.append(0.0 if first_match is None else 1 / first_match)
    return {
        "recall_at_5": recall_hits / len(rankings),
        "mrr": mean(reciprocal_ranks),
    }


def parent_baseline(db: QdrantVectorStore, query_vector: list[float]) -> list[str]:
    points = db.client.query_points(
        collection_name=db.curriculum_collection,
        query=query_vector,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="type", match=models.MatchValue(value="section_anchor")
                )
            ]
        ),
        limit=5,
        with_payload=True,
    ).points
    return [
        str((point.payload or {}).get("parent_id", ""))
        for point in points
        if (point.payload or {}).get("parent_id")
    ]


def hyde_rerank(
    db: QdrantVectorStore, llm: LLMService, query: str, query_vector: list[float]
) -> tuple[list[str], list[dict[str, str]]]:
    points = db.client.query_points(
        collection_name=db.questions_collection,
        query=query_vector,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="type", match=models.MatchValue(value="question")
                )
            ]
        ),
        limit=25,
        with_payload=True,
    ).points
    candidates = [
        {
            "question": str((point.payload or {}).get("page_content", "")),
            "parent_id": str((point.payload or {}).get("parent_id", "")),
            "key_knowledge": str((point.payload or {}).get("key_knowledge", "")),
        }
        for point in points
        if (point.payload or {}).get("parent_id")
    ]
    return llm.rerank_candidate_questions(query, candidates, limit=5), candidates


def anchors_for_parents(db: QdrantVectorStore, parent_ids: list[str]) -> list[str]:
    anchors: list[str] = []
    for parent_id in parent_ids:
        records, _ = db.client.scroll(
            collection_name=db.curriculum_collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="type", match=models.MatchValue(value="section_anchor")
                    ),
                    models.FieldCondition(
                        key="parent_id", match=models.MatchValue(value=parent_id)
                    ),
                ]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        if records:
            anchors.extend((records[0].payload or {}).get("anchor_nodes", []))
    return list(dict.fromkeys(anchor for anchor in anchors if anchor))


def summarize_method(
    rankings: list[list[str]], expected: list[set[str]], graph_edges: list[int] | None = None
) -> dict[str, float]:
    result = retrieval_metrics(rankings, expected)
    if graph_edges is not None:
        result["average_graph_edges"] = mean(graph_edges) if graph_edges else 0.0
    return result


def run_evaluation(dataset_path: Path = DATASET_PATH) -> dict[str, Any]:
    cases = json.loads(dataset_path.read_text(encoding="utf-8"))
    if len(cases) < 20:
        raise ValueError("Evaluation dataset must contain at least 20 questions.")

    settings = Settings()
    settings.validate()
    llm = LLMService(settings)
    db = QdrantVectorStore(settings, llm)
    dag = Neo4jManager(settings)
    dag.verify_connection()

    expected = [expected_parent_ids(case) for case in cases]
    baseline_rankings: list[list[str]] = []
    hyde_rankings: list[list[str]] = []
    graph_edges: list[int] = []

    for case in cases:
        query = case["query"]
        query_vector = llm.embed(query)
        baseline_rankings.append(parent_baseline(db, query_vector))
        reranked_ids, _ = hyde_rerank(db, llm, query, query_vector)
        graph_context = dag.get_graph_context(
            anchors_for_parents(db, reranked_ids), search_mode="search"
        )
        graph_edges.append(len(graph_context))

    dag.close()
    return {
        "question_count": len(cases),
        "categories": sorted({case["category"] for case in cases}),
        "parent_section_vector_baseline": summarize_method(baseline_rankings, expected),
        "hyde_question_rerank": summarize_method(hyde_rankings, expected),
        "hyde_question_rerank_graph_context": summarize_method(
            hyde_rankings, expected, graph_edges
        ),
        "note": (
            "Graph context enriches answer generation after retrieval, so its Recall@5 and "
            "MRR match HyDE plus rerank; average_graph_edges reports the added context."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--output", type=Path, default=RESULTS_PATH)
    args = parser.parse_args()
    result = run_evaluation(args.dataset)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
