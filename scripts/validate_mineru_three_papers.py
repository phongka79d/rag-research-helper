"""Run an isolated MinerU Flash ingestion and evaluation for three papers."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable

if __package__ in {None, ""}:  # Support the documented direct script invocation.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evaluate
from config.settings import Settings
from core.data_ingestion import ingest_document
from database.document_processor import DocumentProcessor
from database.semantic_dag import Neo4jManager
from database.structural_db import QdrantVectorStore
from orchestrator.llm_service import LLMService
from runtime.engine import RuntimeEngine
from scripts.mineru_flash import MinerUError, MinerUFlashClient


PAPER_NAMES = ("attention.pdf", "qlora_paper.pdf", "slm_paper.pdf")
DEFAULT_SOURCE_PREFIX = "mineru_"
_SOURCE_PREFIX_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*_$")


def _safe_error(error: BaseException, secret: str = "") -> str:
    message = str(error).strip() or error.__class__.__name__
    if secret:
        message = message.replace(secret, "[redacted]")
    message = re.sub(r"https?://[^\s)]+", "[url]", message)
    return message[:500]


def _print_ingest_progress(event: dict[str, Any]) -> None:
    """Print compile/skip progress without secrets (section label and status only)."""
    completed = int(event.get("completed", 0))
    total = int(event.get("total", 0))
    status = str(event.get("status", ""))
    section = str(event.get("section", ""))
    print(f"[ingest] {completed}/{total} {status} {section}", flush=True)


def validate_source_prefix(prefix: str) -> str:
    """Validate a caller-selected derived-source prefix.

    Rules: non-empty; only letters, digits, and underscore; starts with a
    letter; ends with ``_``; no path separators, spaces, dashes, or ``..``.
    Invalid prefixes are rejected rather than rewritten.
    """
    if not isinstance(prefix, str) or not prefix:
        raise ValueError(
            "source prefix must be a non-empty string of letters, digits, "
            "and underscore, starting with a letter and ending with '_'"
        )
    if any(sep in prefix for sep in ("/", "\\", "..", " ", "-")):
        raise ValueError(
            "source prefix must not contain path separators, '..', spaces, or dashes"
        )
    if not _SOURCE_PREFIX_PATTERN.fullmatch(prefix):
        raise ValueError(
            "source prefix must start with a letter, contain only letters, "
            "digits, and underscore, and end with '_'"
        )
    return prefix


def _source_name(pdf_name: str, source_prefix: str = DEFAULT_SOURCE_PREFIX) -> str:
    return f"{source_prefix}{Path(pdf_name).stem}.md"


def _normalized_title(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def _title_tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", value.casefold()))


def resolve_section_title(expected: str, actual_titles: list[str]) -> str | None:
    """Resolve harmless OCR/spacing title differences without paper rules."""
    expected_key = _normalized_title(expected)
    exact = [title for title in actual_titles if _normalized_title(title) == expected_key]
    if exact:
        return exact[0]
    expected_tokens = _title_tokens(expected)
    if not expected_tokens:
        return None
    candidates: list[tuple[float, str]] = []
    for title in actual_titles:
        actual_key = _normalized_title(title)
        actual_tokens = _title_tokens(title)
        overlap = len(expected_tokens & actual_tokens) / len(expected_tokens)
        containment = float(expected_key in actual_key or actual_key in expected_key)
        if overlap >= 0.6 or containment:
            candidates.append((overlap + containment, title))
    return max(candidates, default=(0.0, None))[1]


def map_evaluation_cases(
    cases: list[dict[str, Any]],
    source_map: dict[str, str],
    titles_by_source: dict[str, list[str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Map PDF source/section expectations onto isolated MinerU source names."""
    mapped: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for case_index, case in enumerate(cases):
        mapped_case = dict(case)
        expected_items = []
        for item in case.get("expected", []):
            original_source = str(item.get("source", ""))
            mapped_source = source_map.get(original_source)
            resolved_title = resolve_section_title(
                str(item.get("section", "")),
                titles_by_source.get(mapped_source or "", []),
            )
            if not mapped_source or not resolved_title:
                unresolved.append(
                    {
                        "case_index": case_index,
                        "source": original_source,
                        "section": item.get("section", ""),
                    }
                )
                continue
            expected_items.append(
                {"source": mapped_source, "section": resolved_title}
            )
        mapped_case["expected"] = expected_items
        target_file = case.get("target_file")
        if target_file:
            mapped_case["target_file"] = source_map.get(str(target_file), "")
        if expected_items:
            mapped.append(mapped_case)
    return mapped, unresolved


def source_scoped_graph_counts(dag: Any, source: str) -> dict[str, Any]:
    """Count graph nodes and deduplicated edges contributed by one source."""
    graph = dag.get_visual_graph()
    prefix = f"{source}::"
    nodes = [
        node
        for node in graph.get("nodes", [])
        if any(str(locator).startswith(prefix) for locator in node.get("source_locators", []))
    ]
    locators = sorted(
        {
            str(locator)
            for node in nodes
            for locator in node.get("source_locators", [])
            if str(locator).startswith(prefix)
        }
    )
    edges: dict[tuple[str, str, str], dict[str, Any]] = {}
    for locator in locators:
        for edge in dag.get_visual_graph(locator).get("edges", []):
            key = (str(edge.get("source")), str(edge.get("label")), str(edge.get("target")))
            edges[key] = edge
    return {
        "nodes": len(nodes),
        "edges": len(edges),
        "edge_sample": list(edges.values())[:10],
    }


def source_storage_counts(db: Any, source: str) -> dict[str, int]:
    sections = db.get_section_exact(source, "")
    records, _ = db.client.scroll(
        collection_name=db.questions_collection,
        scroll_filter=db._filter(source=source),
        limit=10_000,
        with_payload=False,
        with_vectors=False,
    )
    return {"sections": len(sections), "questions": len(records)}


_MAX_REPORT_EDGE_AUDIT = 10


def build_source_ingestion_report(result: dict[str, Any]) -> dict[str, Any]:
    """Derive truthful per-source validation fields from one ingest result.

    Distinguishes sections compiled this run from sections skipped as current.
    When every section is skipped, graph_relationships counters are labeled as
    not current-run extraction so a zero-candidate read is not misreported.
    """
    sections_compiled = list(result.get("ingested") or [])
    sections_skipped = list(result.get("skipped") or [])
    if sections_compiled and sections_skipped:
        ingestion_run_status = "partial"
    elif sections_compiled:
        ingestion_run_status = "compiled"
    else:
        ingestion_run_status = "skipped_already_current"

    raw_relationships = result.get("graph_relationships")
    if not isinstance(raw_relationships, dict):
        raw_relationships = {}
    graph_relationships: dict[str, Any] = {
        "candidates": max(int(raw_relationships.get("candidates", 0) or 0), 0),
        "verifier_approvals": max(
            int(raw_relationships.get("verifier_approvals", 0) or 0), 0
        ),
        "retained": max(int(raw_relationships.get("retained", 0) or 0), 0),
    }
    local_rejections = raw_relationships.get("local_rejections")
    if isinstance(local_rejections, dict) and local_rejections:
        graph_relationships["local_rejections"] = {
            str(key): max(int(value or 0), 0) for key, value in local_rejections.items()
        }

    audit = raw_relationships.get("retained_edge_audit")
    retained_edge_audit = (
        list(audit)[:_MAX_REPORT_EDGE_AUDIT] if isinstance(audit, list) else []
    )

    if ingestion_run_status == "skipped_already_current":
        # Counters stayed at zero because compilation did not run.
        counts_scope = "not_current_run_skipped"
        extraction_note = (
            "run skipped compilation; graph_relationships counters are not "
            "current-run extraction results"
        )
    else:
        counts_scope = "current_run"
        extraction_note = None

    report: dict[str, Any] = {
        "sections_compiled": sections_compiled,
        "sections_skipped": sections_skipped,
        "ingestion_run_status": ingestion_run_status,
        "graph_relationships": graph_relationships,
        "graph_relationship_counts_scope": counts_scope,
        "retained_edge_audit": retained_edge_audit,
    }
    if extraction_note is not None:
        report["graph_extraction_note"] = extraction_note
    return report


def compare_evaluation_results(
    mineru: dict[str, Any], pdf: dict[str, Any]
) -> dict[str, dict[str, float]]:
    comparisons: dict[str, dict[str, float]] = {}
    fields = (
        ("parent_section_vector_baseline", "recall_at_5"),
        ("parent_section_vector_baseline", "mrr"),
        ("hyde_question_rerank", "recall_at_2"),
        ("hyde_question_rerank", "mrr"),
        ("hyde_question_rerank", "all_expected_sources_rate"),
        ("hyde_question_rerank", "average_retrieval_latency_seconds"),
    )
    for method, field in fields:
        mineru_value = float(mineru.get(method, {}).get(field, 0.0))
        pdf_value = float(pdf.get(method, {}).get(field, 0.0))
        comparisons[f"{method}.{field}"] = {
            "mineru": mineru_value,
            "pdf": pdf_value,
            "delta": mineru_value - pdf_value,
        }
    return comparisons


def _extract_paper(
    client: MinerUFlashClient,
    pdf_path: Path,
    output_dir: Path,
    batch_size: int,
    poll_interval: float,
    timeout: float,
    source_prefix: str = DEFAULT_SOURCE_PREFIX,
) -> dict[str, Any]:
    source = _source_name(pdf_path.name, source_prefix)
    markdown_path = output_dir / source
    result = client.extract(
        pdf_path,
        output_path=markdown_path,
        batch_size=batch_size,
        poll_interval=poll_interval,
        timeout=timeout,
        language="en",
    )
    manifest = result["manifest"]
    return {
        "paper": pdf_path.name,
        "input_path": str(pdf_path),
        "source": source,
        "markdown_path": str(markdown_path),
        "manifest_path": str(result["manifest_path"]),
        "manifest": manifest,
        "extraction_status": "complete" if manifest.get("complete") else "partial",
    }


def run_validation(
    input_dir: Path,
    output_dir: Path,
    *,
    source_prefix: str = DEFAULT_SOURCE_PREFIX,
    batch_size: int = 10,
    poll_interval: float = 3.0,
    timeout: float = 600.0,
    workers: int = 1,
    force_reingest: bool = True,
    client: MinerUFlashClient | None = None,
    settings_factory: Callable[[], Any] = Settings,
    llm_factory: Callable[[Any], Any] = LLMService,
    db_factory: Callable[[Any, Any], Any] = QdrantVectorStore,
    dag_factory: Callable[[Any], Any] = Neo4jManager,
    evaluate_fn: Callable[..., dict[str, Any]] = evaluate.run_evaluation,
) -> dict[str, Any]:
    source_prefix = validate_source_prefix(source_prefix)
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    mineru_client = client or MinerUFlashClient()
    report: dict[str, Any] = {
        "papers": [],
        "output_dir": str(output_dir),
        "source_prefix": source_prefix,
        "overall_complete": False,
        "limitations": [],
    }

    for paper_name in PAPER_NAMES:
        pdf_path = (input_dir / paper_name).resolve()
        if not pdf_path.is_file():
            report["papers"].append(
                {"paper": paper_name, "extraction_status": "failed", "error": "input PDF missing"}
            )
            report["limitations"].append(f"missing input: {paper_name}")
            continue
        try:
            record = _extract_paper(
                mineru_client,
                pdf_path,
                output_dir,
                batch_size,
                poll_interval,
                timeout,
                source_prefix=source_prefix,
            )
        except (MinerUError, OSError, ValueError) as error:
            record = {
                "paper": paper_name,
                "extraction_status": "failed",
                "error": _safe_error(error),
            }
            report["limitations"].append(f"MinerU extraction failed for {paper_name}")
        report["papers"].append(record)

    if any(item.get("extraction_status") != "complete" for item in report["papers"]):
        report["limitations"].append("not all three MinerU manifests completed")
        return report

    dag = None
    try:
        settings = settings_factory()
        settings.validate()
        llm = llm_factory(settings)
        db = db_factory(settings, llm)
        dag = dag_factory(settings)
        dag.verify_connection()
    except Exception as error:
        if dag is not None:
            try:
                dag.close()
            except Exception:
                pass
        report["limitations"].append(
            f"live service setup failed: {_safe_error(error, str(getattr(locals().get('settings', None), 'OPENAI_API_KEY', '')))}"
        )
        return report
    try:
        titles_by_source: dict[str, list[str]] = {}
        source_map = {
            paper_name: _source_name(paper_name, source_prefix)
            for paper_name in PAPER_NAMES
        }
        for record in report["papers"]:
            try:
                processor = DocumentProcessor()
                sections = processor.process_mineru_markdown(
                    record["markdown_path"], record["manifest_path"]
                )
                titles_by_source[record["source"]] = [
                    str(section["metadata"].get("section", "")) for section in sections
                ]
                record["parser"] = dict(processor.last_report)
                result = ingest_document(
                    record["markdown_path"],
                    db,
                    llm,
                    dag,
                    processor=processor,
                    force_reingest=force_reingest,
                    mineru_manifest_path=record["manifest_path"],
                    progress_callback=_print_ingest_progress,
                )
                record["ingestion"] = result
                record.update(build_source_ingestion_report(result))
                record["storage"] = source_storage_counts(db, record["source"])
                # Always report post-ingest source-scoped graph state, including skips.
                record["graph"] = source_scoped_graph_counts(dag, record["source"])
                record["ingestion_status"] = "complete"
            except Exception as error:
                record["ingestion_status"] = "failed"
                record["error"] = _safe_error(error, str(getattr(settings, "OPENAI_API_KEY", "")))
                report["limitations"].append(f"parser or ingestion failed for {record['paper']}")

        if any(item.get("ingestion_status") != "complete" for item in report["papers"]):
            report["limitations"].append("not all three MinerU sources were ingested")
            return report

        pdf_baseline_sources: dict[str, Any] = {}
        for paper_name in PAPER_NAMES:
            original_path = (input_dir / paper_name).resolve()
            original_processor = DocumentProcessor()
            original_sections = original_processor.process(original_path)
            pdf_baseline_sources[paper_name] = {
                "parser": dict(original_processor.last_report),
                "sections": len(original_sections),
                "storage": source_storage_counts(db, paper_name),
                "graph": source_scoped_graph_counts(dag, paper_name),
            }
        report["pdf_baseline_sources"] = pdf_baseline_sources

        cases = json.loads(Path("data/eval_real_papers.json").read_text(encoding="utf-8"))
        mapped_cases, unresolved = map_evaluation_cases(cases, source_map, titles_by_source)
        mapped_dataset = output_dir / "eval_mineru_real_papers.json"
        mapped_dataset.write_text(
            json.dumps(mapped_cases, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        report["evaluation_dataset"] = str(mapped_dataset)
        report["unresolved_cases"] = unresolved
        if unresolved or len(mapped_cases) < 20:
            report["limitations"].append("MinerU section titles could not map all evaluation cases")
            return report
        try:
            report["evaluation"] = evaluate_fn(mapped_dataset, workers=workers)
        except Exception as error:
            report["limitations"].append(
                f"evaluation failed: {_safe_error(error, str(getattr(settings, 'OPENAI_API_KEY', '')))}"
            )
            return report
        baseline_path = Path("eval_results_real_papers.json")
        if baseline_path.is_file():
            report["pdf_baseline_evaluation"] = json.loads(
                baseline_path.read_text(encoding="utf-8")
            )
            report["comparison"] = compare_evaluation_results(
                report["evaluation"], report["pdf_baseline_evaluation"]
            )
        else:
            report["limitations"].append("existing PDF evaluation result is missing")

        engine = RuntimeEngine(llm, db, dag)
        answers: list[dict[str, Any]] = []
        for paper_name in PAPER_NAMES:
            representative = next(
                (
                    case
                    for case in cases
                    if case.get("target_file") == paper_name
                ),
                None,
            )
            if representative is None:
                continue
            source = source_map[paper_name]
            try:
                answer = engine.ask(representative["query"], target_file=source)
                answers.append(
                    {
                        "paper": paper_name,
                        "query": representative["query"],
                        "sources": answer.get("sources", []),
                        "graph_context_count": len(answer.get("graph_context", [])),
                        "answer_preview": str(answer.get("answer", ""))[:500],
                    }
                )
            except Exception as error:
                answers.append(
                    {
                        "paper": paper_name,
                        "query": representative["query"],
                        "error": _safe_error(
                            error, str(getattr(settings, "OPENAI_API_KEY", ""))
                        ),
                    }
                )
                report["limitations"].append(f"representative Ask failed for {paper_name}")
        report["representative_answers"] = answers
        report["overall_complete"] = not report["limitations"]
    finally:
        dag.close()
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/papers"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/mineru-validation"))
    parser.add_argument(
        "--source-prefix",
        default=DEFAULT_SOURCE_PREFIX,
        help=(
            "Derived-source name prefix (default: mineru_). "
            "Must start with a letter, contain only letters/digits/underscore, "
            "and end with '_'."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--poll-interval", type=float, default=3.0)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--no-force-reingest", dest="force_reingest", action="store_false")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_validation(
            args.input_dir,
            args.output_dir,
            source_prefix=args.source_prefix,
            batch_size=args.batch_size,
            poll_interval=args.poll_interval,
            timeout=args.timeout,
            workers=args.workers,
            force_reingest=args.force_reingest,
        )
    except Exception as error:
        print(f"MinerU three-paper validation failed: {_safe_error(error)}", file=sys.stderr)
        return 1
    report_path = Path(report["output_dir"]) / "mineru_three_paper_validation.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Report: {report_path}")
    return 0 if report.get("overall_complete") else 2


if __name__ == "__main__":
    raise SystemExit(main())
