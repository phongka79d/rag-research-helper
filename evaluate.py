"""Run minimal retrieval ablations over data/eval.json."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

from qdrant_client import models

from config.settings import Settings
from core.data_ingestion import make_parent_id
from database.semantic_dag import Neo4jManager
from database.structural_db import QdrantVectorStore
from orchestrator.llm_service import LLMService
from runtime.engine import collect_anchor_nodes

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


def parent_baseline(
    db: QdrantVectorStore, query_vector: list[float], target_file: str = ""
) -> list[str]:
    conditions = [
        models.FieldCondition(
            key="type", match=models.MatchValue(value="section_anchor")
        )
    ]
    if target_file:
        conditions.append(
            models.FieldCondition(
                key="source", match=models.MatchValue(value=target_file)
            )
        )
    points = db.client.query_points(
        collection_name=db.curriculum_collection,
        query=query_vector,
        query_filter=models.Filter(must=conditions),
        limit=5,
        with_payload=True,
    ).points
    return [
        str((point.payload or {}).get("parent_id", ""))
        for point in points
        if (point.payload or {}).get("parent_id")
    ]


def summarize_method(
    rankings: list[list[str]], expected: list[set[str]], graph_edges: list[int] | None = None
) -> dict[str, float]:
    result = retrieval_metrics(rankings, expected)
    if graph_edges is not None:
        result["average_graph_edges"] = mean(graph_edges) if graph_edges else 0.0
    return result


def runtime_metrics(
    rankings: list[list[str]],
    expected_ids: list[set[str]],
    returned_sources: list[list[str]],
    expected_sources: list[set[str]],
    retrieval_latencies: list[float],
) -> dict[str, float]:
    if not rankings:
        return {
            "recall_at_2": 0.0,
            "mrr": 0.0,
            "all_expected_sources_rate": 0.0,
            "average_retrieval_latency_seconds": 0.0,
        }

    recall_scores: list[float] = []
    reciprocal_ranks: list[float] = []
    source_matches: list[float] = []
    for ranked_ids, relevant_ids, sources, relevant_sources in zip(
        rankings, expected_ids, returned_sources, expected_sources
    ):
        returned_ids = set(ranked_ids[:2])
        recall_scores.append(
            len(returned_ids & relevant_ids) / len(relevant_ids)
            if relevant_ids
            else 0.0
        )
        first_match = next(
            (
                index
                for index, parent_id in enumerate(ranked_ids[:2], start=1)
                if parent_id in relevant_ids
            ),
            None,
        )
        reciprocal_ranks.append(0.0 if first_match is None else 1 / first_match)
        source_matches.append(
            float(
                bool(relevant_sources)
                and relevant_sources.issubset(
                    {Path(source).name for source in sources if source}
                )
            )
        )

    return {
        "recall_at_2": mean(recall_scores),
        "mrr": mean(reciprocal_ranks),
        "all_expected_sources_rate": mean(source_matches),
        "average_retrieval_latency_seconds": (
            mean(retrieval_latencies) if retrieval_latencies else 0.0
        ),
    }


def summarize_runtime(
    rankings: list[list[str]],
    expected_ids: list[set[str]],
    returned_sources: list[list[str]],
    expected_sources: list[set[str]],
    retrieval_latencies: list[float],
    graph_edges: list[int] | None = None,
) -> dict[str, float]:
    result = runtime_metrics(
        rankings,
        expected_ids,
        returned_sources,
        expected_sources,
        retrieval_latencies,
    )
    if graph_edges is not None:
        result["average_graph_edges"] = mean(graph_edges) if graph_edges else 0.0
    return result


def evaluate_case(
    query: str,
    query_vector: list[float],
    db: QdrantVectorStore,
    llm: LLMService,
    dag: Neo4jManager,
    target_file: str = "",
) -> tuple[list[str], list[str], list[str], int, float]:
    baseline = parent_baseline(db, query_vector, target_file)
    retrieval_started = perf_counter()
    sections = db.search_candidates_and_fetch_parent(
        query=query,
        llm_service=llm,
        target_file=target_file,
        query_vector=query_vector,
    )
    retrieval_latency = perf_counter() - retrieval_started

    metadata = [section.get("metadata", {}) for section in sections]
    parent_ids = [
        str(item.get("parent_id", "")) for item in metadata if item.get("parent_id")
    ]
    sources = [str(item.get("source", "")) for item in metadata if item.get("source")]
    graph_context = dag.get_graph_context(
        collect_anchor_nodes(sections),
        search_mode="search",
        source=target_file,
    )
    return baseline, parent_ids, sources, len(graph_context), retrieval_latency


def run_evaluation(dataset_path: Path = DATASET_PATH, workers: int = 4) -> dict[str, Any]:
    cases = json.loads(dataset_path.read_text(encoding="utf-8"))
    if len(cases) < 20:
        raise ValueError("Evaluation dataset must contain at least 20 questions.")
    if workers < 1:
        raise ValueError("workers must be at least 1.")

    settings = Settings()
    settings.validate()
    llm = LLMService(settings)
    db = QdrantVectorStore(settings, llm)
    dag = Neo4jManager(settings)
    dag.verify_connection()

    try:
        expected = [expected_parent_ids(case) for case in cases]
        expected_sources = [
            {Path(item["source"]).name for item in case["expected"]}
            for case in cases
        ]
        queries = [case["query"] for case in cases]
        target_files = [str(case.get("target_file", "")) for case in cases]
        query_vectors = llm.embed_many(queries)
        with ThreadPoolExecutor(max_workers=min(workers, len(cases))) as executor:
            case_results = list(
                executor.map(
                    evaluate_case,
                    queries,
                    query_vectors,
                    [db] * len(cases),
                    [llm] * len(cases),
                    [dag] * len(cases),
                    target_files,
                )
            )
    finally:
        dag.close()
    baseline_rankings = [result[0] for result in case_results]
    hyde_rankings = [result[1] for result in case_results]
    returned_sources = [result[2] for result in case_results]
    graph_edges = [result[3] for result in case_results]
    retrieval_latencies = [result[4] for result in case_results]
    return {
        "question_count": len(cases),
        "categories": sorted({case["category"] for case in cases}),
        "parent_section_vector_baseline": summarize_method(baseline_rankings, expected),
        "hyde_question_rerank": summarize_runtime(
            hyde_rankings,
            expected,
            returned_sources,
            expected_sources,
            retrieval_latencies,
        ),
        "hyde_question_rerank_graph_context": summarize_runtime(
            hyde_rankings,
            expected,
            returned_sources,
            expected_sources,
            retrieval_latencies,
            graph_edges,
        ),
        "note": (
            "Graph context enriches answer generation after retrieval, so its Recall@2 and "
            "MRR match runtime retrieval; average_graph_edges reports the added context. "
            "Recall@5 is retained only for the direct parent-vector baseline."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--output", type=Path, default=RESULTS_PATH)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    result = run_evaluation(args.dataset, workers=args.workers)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
