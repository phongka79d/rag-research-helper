from pathlib import Path

import pytest

import scripts.validate_mineru_three_papers as validate_mod
from scripts.validate_mineru_three_papers import (
    PAPER_NAMES,
    _print_ingest_progress,
    _safe_error,
    _source_name,
    build_source_ingestion_report,
    compare_evaluation_results,
    map_evaluation_cases,
    resolve_section_title,
    run_validation,
    validate_source_prefix,
)


def test_build_source_ingestion_report_compiled_status():
    result = {
        "ingested": ["mineru_a.md::Intro", "mineru_a.md::Method"],
        "skipped": [],
        "graph_relationships": {
            "candidates": 4,
            "verifier_approvals": 2,
            "retained": 1,
            "local_rejections": {"relation_mismatch": 1},
            "retained_edge_audit": [
                {
                    "source": "A",
                    "relation": "PART_OF",
                    "target": "B",
                    "locator": "mineru_a.md::Method",
                    "evidence_id": "s0",
                    "evidence_preview": "A is part of B.",
                }
            ],
        },
    }
    report = build_source_ingestion_report(result)
    assert report["ingestion_run_status"] == "compiled"
    assert report["sections_compiled"] == result["ingested"]
    assert report["sections_skipped"] == []
    assert report["graph_relationship_counts_scope"] == "current_run"
    assert "graph_extraction_note" not in report
    assert report["graph_relationships"]["candidates"] == 4
    assert report["graph_relationships"]["local_rejections"] == {
        "relation_mismatch": 1
    }
    assert len(report["retained_edge_audit"]) == 1


def test_build_source_ingestion_report_skipped_is_not_zero_extraction():
    result = {
        "ingested": [],
        "skipped": ["mineru_a.md::Intro", "mineru_a.md::Method"],
        "graph_relationships": {
            "candidates": 0,
            "verifier_approvals": 0,
            "retained": 0,
        },
    }
    report = build_source_ingestion_report(result)
    assert report["ingestion_run_status"] == "skipped_already_current"
    assert report["sections_compiled"] == []
    assert report["sections_skipped"] == result["skipped"]
    assert report["graph_relationship_counts_scope"] == "not_current_run_skipped"
    assert "skipped compilation" in report["graph_extraction_note"]
    assert report["retained_edge_audit"] == []
    # Zero counters may be present, but must not be labeled as current-run extraction.
    assert report["graph_relationships"]["candidates"] == 0
    assert report["graph_relationship_counts_scope"] != "current_run"


def test_build_source_ingestion_report_partial_and_audit_bound():
    audit = [
        {"source": f"S{i}", "relation": "RELATES_TO", "target": f"T{i}"}
        for i in range(15)
    ]
    result = {
        "ingested": ["mineru_a.md::New"],
        "skipped": ["mineru_a.md::Old"],
        "graph_relationships": {
            "candidates": 3,
            "verifier_approvals": 1,
            "retained": 1,
            "retained_edge_audit": audit,
        },
    }
    report = build_source_ingestion_report(result)
    assert report["ingestion_run_status"] == "partial"
    assert report["graph_relationship_counts_scope"] == "current_run"
    assert len(report["retained_edge_audit"]) == 10


def test_evaluation_comparison_reports_signed_metric_deltas():
    result = compare_evaluation_results(
        {
            "parent_section_vector_baseline": {"recall_at_5": 0.8, "mrr": 0.7},
            "hyde_question_rerank": {
                "recall_at_2": 0.6,
                "mrr": 0.5,
                "all_expected_sources_rate": 0.4,
                "average_retrieval_latency_seconds": 1.5,
            },
        },
        {
            "parent_section_vector_baseline": {"recall_at_5": 0.7, "mrr": 0.6},
            "hyde_question_rerank": {
                "recall_at_2": 0.5,
                "mrr": 0.4,
                "all_expected_sources_rate": 0.3,
                "average_retrieval_latency_seconds": 2.0,
            },
        },
    )

    assert result["hyde_question_rerank.recall_at_2"]["mineru"] == pytest.approx(0.6)
    assert result["hyde_question_rerank.recall_at_2"]["pdf"] == pytest.approx(0.5)
    assert result["hyde_question_rerank.recall_at_2"]["delta"] == pytest.approx(0.1)
    assert result["hyde_question_rerank.average_retrieval_latency_seconds"]["delta"] == -0.5


def test_default_source_prefix_produces_legacy_names():
    assert _source_name("attention.pdf") == "mineru_attention.md"
    assert validate_source_prefix("mineru_") == "mineru_"


def test_custom_source_prefix_produces_distinct_source_names():
    prefix = "mineru_evidence_"
    assert _source_name("attention.pdf", prefix) == "mineru_evidence_attention.md"
    assert _source_name("qlora_paper.pdf", prefix) == "mineru_evidence_qlora_paper.md"
    assert _source_name("slm_paper.pdf", prefix) == "mineru_evidence_slm_paper.md"


@pytest.mark.parametrize(
    "invalid_prefix",
    [
        "",
        "mineru",
        "mineru-evidence_",
        "1mineru_",
        "mineru evidence_",
        "../mineru_",
        "mineru/_",
        "mineru\\_",
        "_mineru_",
    ],
)
def test_invalid_source_prefix_is_rejected(invalid_prefix):
    with pytest.raises(ValueError):
        validate_source_prefix(invalid_prefix)


def test_section_mapping_handles_spacing_and_source_aliases():
    cases = [
        {
            "category": "factual",
            "query": "What is QLoRA?",
            "target_file": "qlora_paper.pdf",
            "expected": [
                {"source": "qlora_paper.pdf", "section": "3 QL ORA Finetuning"}
            ],
        }
    ]
    source_map = {name: f"mineru_{Path(name).stem}.md" for name in PAPER_NAMES}

    mapped, unresolved = map_evaluation_cases(
        cases,
        source_map,
        {source_map["qlora_paper.pdf"]: ["3 QLORA Finetuning"]},
    )

    assert unresolved == []
    assert mapped[0]["target_file"] == "mineru_qlora_paper.md"
    assert mapped[0]["expected"] == [
        {"source": "mineru_qlora_paper.md", "section": "3 QLORA Finetuning"}
    ]
    assert resolve_section_title(
        "3 QL ORA Finetuning", ["3 QLORA Finetuning"]
    ) == "3 QLORA Finetuning"


def test_section_mapping_uses_selected_source_prefix():
    prefix = "mineru_evidence_"
    cases = [
        {
            "category": "factual",
            "query": "What is attention?",
            "target_file": "attention.pdf",
            "expected": [
                {"source": "attention.pdf", "section": "3.2 Attention"}
            ],
        }
    ]
    source_map = {
        name: _source_name(name, prefix) for name in PAPER_NAMES
    }

    mapped, unresolved = map_evaluation_cases(
        cases,
        source_map,
        {source_map["attention.pdf"]: ["3.2 Attention"]},
    )

    assert unresolved == []
    assert mapped[0]["target_file"] == "mineru_evidence_attention.md"
    assert mapped[0]["expected"] == [
        {"source": "mineru_evidence_attention.md", "section": "3.2 Attention"}
    ]


def test_incomplete_mineru_run_stops_before_database_setup(tmp_path):
    input_dir = tmp_path / "papers"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    for paper_name in PAPER_NAMES:
        (input_dir / paper_name).write_bytes(b"test pdf")

    class PartialClient:
        def extract(self, pdf_path, **kwargs):
            return {
                "manifest": {"complete": False},
                "manifest_path": str(output_dir / f"{pdf_path.stem}.manifest.json"),
            }

    def unexpected_settings():
        raise AssertionError("database/provider setup must not run for partial extraction")

    report = run_validation(
        input_dir,
        output_dir,
        client=PartialClient(),
        settings_factory=unexpected_settings,
    )

    assert report["overall_complete"] is False
    assert report["source_prefix"] == "mineru_"
    assert "not all three MinerU manifests completed" in report["limitations"]
    assert all(item["extraction_status"] == "partial" for item in report["papers"])
    assert all(item["source"] == f"mineru_{Path(item['paper']).stem}.md" for item in report["papers"])


def test_run_validation_propagates_custom_source_prefix(tmp_path):
    input_dir = tmp_path / "papers"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    for paper_name in PAPER_NAMES:
        (input_dir / paper_name).write_bytes(b"test pdf")
    captured_paths: list[Path] = []

    class PartialClient:
        def extract(self, pdf_path, **kwargs):
            captured_paths.append(Path(kwargs["output_path"]))
            return {
                "manifest": {"complete": False},
                "manifest_path": str(output_dir / f"{pdf_path.stem}.manifest.json"),
            }

    report = run_validation(
        input_dir,
        output_dir,
        source_prefix="mineru_evidence_",
        client=PartialClient(),
        settings_factory=lambda: (_ for _ in ()).throw(
            AssertionError("setup must not run for partial extraction")
        ),
    )

    assert report["source_prefix"] == "mineru_evidence_"
    assert {path.name for path in captured_paths} == {
        "mineru_evidence_attention.md",
        "mineru_evidence_qlora_paper.md",
        "mineru_evidence_slm_paper.md",
    }
    assert {item["source"] for item in report["papers"]} == {
        "mineru_evidence_attention.md",
        "mineru_evidence_qlora_paper.md",
        "mineru_evidence_slm_paper.md",
    }


def test_run_validation_rejects_invalid_source_prefix(tmp_path):
    with pytest.raises(ValueError, match="source prefix"):
        run_validation(
            tmp_path / "papers",
            tmp_path / "out",
            source_prefix="bad-prefix",
        )


def test_live_service_setup_failure_is_reported(tmp_path):
    input_dir = tmp_path / "papers"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    for paper_name in PAPER_NAMES:
        (input_dir / paper_name).write_bytes(b"test pdf")

    class CompleteClient:
        def extract(self, pdf_path, **kwargs):
            return {
                "manifest": {"complete": True},
                "manifest_path": str(output_dir / f"{pdf_path.stem}.manifest.json"),
            }

    def unavailable_settings():
        raise RuntimeError("provider unavailable")

    report = run_validation(
        input_dir,
        output_dir,
        client=CompleteClient(),
        settings_factory=unavailable_settings,
    )

    assert report["overall_complete"] is False
    assert report["source_prefix"] == "mineru_"
    assert any("live service setup failed" in item for item in report["limitations"])


def test_two_source_prefixes_are_distinct_identities():
    default = _source_name("attention.pdf", "mineru_")
    custom = _source_name("attention.pdf", "mineru_evidence_")
    assert default == "mineru_attention.md"
    assert custom == "mineru_evidence_attention.md"
    assert default != custom


def test_safe_error_redacts_api_key_and_urls():
    secret = "sk-test-secret-key-value"
    message = _safe_error(
        RuntimeError(
            f"auth failed key={secret} endpoint=https://api.example.com/v1/models"
        ),
        secret,
    )
    assert secret not in message
    assert "https://api.example.com" not in message
    assert "[redacted]" in message
    assert "[url]" in message


def test_setup_failure_errors_are_secret_safe(tmp_path):
    input_dir = tmp_path / "papers"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    for paper_name in PAPER_NAMES:
        (input_dir / paper_name).write_bytes(b"test pdf")
    secret = "sk-live-setup-secret"

    class CompleteClient:
        def extract(self, pdf_path, **kwargs):
            return {
                "manifest": {"complete": True},
                "manifest_path": str(output_dir / f"{pdf_path.stem}.manifest.json"),
            }

    def leaky_settings():
        class Settings:
            OPENAI_API_KEY = secret

            def validate(self):
                raise RuntimeError(
                    f"provider rejected key={secret} at https://provider.example/v1"
                )

        return Settings()

    report = run_validation(
        input_dir,
        output_dir,
        client=CompleteClient(),
        settings_factory=leaky_settings,
    )
    serialized = str(report)
    assert secret not in serialized
    assert "https://provider.example" not in serialized
    assert any("live service setup failed" in item for item in report["limitations"])
    assert any("[redacted]" in item or "[url]" in item for item in report["limitations"])


def test_run_validation_leaves_original_pdfs_unchanged(tmp_path):
    input_dir = tmp_path / "papers"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    original_bytes = {
        name: f"pdf-bytes-{name}".encode() for name in PAPER_NAMES
    }
    for paper_name, payload in original_bytes.items():
        (input_dir / paper_name).write_bytes(payload)
    original_mtimes = {
        name: (input_dir / name).stat().st_mtime_ns for name in PAPER_NAMES
    }

    class PartialClient:
        def extract(self, pdf_path, **kwargs):
            output_path = Path(kwargs["output_path"])
            assert output_path.parent == output_dir.resolve()
            assert pdf_path.parent == input_dir.resolve()
            # Write derived output only; never touch the input PDF path.
            output_path.write_text("# derived\n", encoding="utf-8")
            return {
                "manifest": {"complete": False},
                "manifest_path": str(output_dir / f"{pdf_path.stem}.manifest.json"),
            }

    run_validation(
        input_dir,
        output_dir,
        client=PartialClient(),
        settings_factory=lambda: (_ for _ in ()).throw(
            AssertionError("setup must not run for partial extraction")
        ),
    )

    for paper_name, payload in original_bytes.items():
        pdf_path = input_dir / paper_name
        assert pdf_path.is_file()
        assert pdf_path.read_bytes() == payload
        assert pdf_path.stat().st_mtime_ns == original_mtimes[paper_name]


def _complete_client(output_dir: Path):
    class CompleteClient:
        def extract(self, pdf_path, **kwargs):
            return {
                "manifest": {"complete": True},
                "manifest_path": str(output_dir / f"{pdf_path.stem}.manifest.json"),
            }

    return CompleteClient()


def _stub_live_stack(monkeypatch, *, ingest_result_factory):
    class Settings:
        OPENAI_API_KEY = "sk-unused-in-mocks"

        def validate(self):
            return None

    class FakeProcessor:
        def __init__(self):
            self.last_report = {"parser": "mineru", "sections": 1}

        def process_mineru_markdown(self, markdown_path, manifest_path):
            return [{"metadata": {"section": "Intro"}}]

        def process(self, path):
            return [{"metadata": {"section": "Intro"}}]

    class FakeDAG:
        def verify_connection(self):
            return None

        def close(self):
            return None

        def get_visual_graph(self, locator=None):
            return {"nodes": [], "edges": []}

    class FakeDB:
        questions_collection = "questions"

        def get_section_exact(self, source, section):
            return []

        @property
        def client(self):
            return self

        def scroll(self, **kwargs):
            return [], None

        def _filter(self, source=None):
            return None

    monkeypatch.setattr(validate_mod, "DocumentProcessor", FakeProcessor)
    monkeypatch.setattr(
        validate_mod,
        "ingest_document",
        lambda path, *args, **kwargs: ingest_result_factory(Path(path).name),
    )
    return Settings, FakeDB, FakeDAG


def test_run_validation_reports_skipped_sources_without_zero_extraction_claim(
    tmp_path, monkeypatch
):
    input_dir = tmp_path / "papers"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    for paper_name in PAPER_NAMES:
        (input_dir / paper_name).write_bytes(b"test pdf")

    def skipped_result(source_name: str):
        return {
            "ingested": [],
            "skipped": [f"{source_name}::Intro"],
            "graph_relationships": {
                "candidates": 0,
                "verifier_approvals": 0,
                "retained": 0,
            },
        }

    Settings, FakeDB, FakeDAG = _stub_live_stack(
        monkeypatch, ingest_result_factory=skipped_result
    )

    report = run_validation(
        input_dir,
        output_dir,
        client=_complete_client(output_dir),
        settings_factory=Settings,
        llm_factory=lambda settings: object(),
        db_factory=lambda settings, llm: FakeDB(),
        dag_factory=lambda settings: FakeDAG(),
        evaluate_fn=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("evaluation should not run with unresolved cases")
        ),
    )

    assert all(
        item.get("ingestion_run_status") == "skipped_already_current"
        for item in report["papers"]
    )
    assert all(
        item.get("graph_relationship_counts_scope") == "not_current_run_skipped"
        for item in report["papers"]
    )
    assert all(
        "skipped compilation" in item.get("graph_extraction_note", "")
        for item in report["papers"]
    )
    assert all(
        item.get("graph_relationship_counts_scope") != "current_run"
        for item in report["papers"]
    )


def test_run_validation_bounds_retained_edge_audit_from_ingest(tmp_path, monkeypatch):
    input_dir = tmp_path / "papers"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    for paper_name in PAPER_NAMES:
        (input_dir / paper_name).write_bytes(b"test pdf")

    def compiled_result(source_name: str):
        audit = [
            {
                "source": f"S{i}",
                "relation": "RELATES_TO",
                "target": f"T{i}",
                "locator": f"{source_name}::Intro",
                "evidence_id": f"e{i}",
            }
            for i in range(15)
        ]
        return {
            "ingested": [f"{source_name}::Intro"],
            "skipped": [],
            "graph_relationships": {
                "candidates": 15,
                "verifier_approvals": 15,
                "retained": 15,
                "retained_edge_audit": audit,
            },
        }

    Settings, FakeDB, FakeDAG = _stub_live_stack(
        monkeypatch, ingest_result_factory=compiled_result
    )

    report = run_validation(
        input_dir,
        output_dir,
        client=_complete_client(output_dir),
        settings_factory=Settings,
        llm_factory=lambda settings: object(),
        db_factory=lambda settings, llm: FakeDB(),
        dag_factory=lambda settings: FakeDAG(),
        evaluate_fn=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("evaluation should not run with unresolved cases")
        ),
    )

    assert all(item.get("ingestion_run_status") == "compiled" for item in report["papers"])
    assert all(len(item.get("retained_edge_audit", [])) == 10 for item in report["papers"])


def test_print_ingest_progress_is_compact_without_secrets(capsys):
    _print_ingest_progress(
        {
            "completed": 3,
            "total": 15,
            "section": "mineru_attention.md::Method",
            "status": "compiled",
        }
    )
    _print_ingest_progress(
        {
            "completed": 3,
            "total": 15,
            "section": "mineru_attention.md::Abstract",
            "status": "up_to_date",
        }
    )
    out = capsys.readouterr().out
    assert "[ingest] 3/15 compiled mineru_attention.md::Method" in out
    assert "[ingest] 3/15 up_to_date mineru_attention.md::Abstract" in out
    assert "sk-" not in out
    assert "http" not in out
    assert "password" not in out


def test_run_validation_passes_progress_callback_and_prints_status(tmp_path, monkeypatch, capsys):
    input_dir = tmp_path / "papers"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    for paper_name in PAPER_NAMES:
        (input_dir / paper_name).write_bytes(b"test pdf")

    received_callbacks: list[object] = []

    def fake_ingest(path, *args, **kwargs):
        callback = kwargs.get("progress_callback")
        received_callbacks.append(callback)
        assert callback is not None
        source_name = Path(path).name
        callback(
            {
                "completed": 1,
                "total": 2,
                "section": f"{source_name}::Intro",
                "status": "compiled",
            }
        )
        callback(
            {
                "completed": 2,
                "total": 2,
                "section": f"{source_name}::Method",
                "status": "up_to_date",
            }
        )
        return {
            "ingested": [f"{source_name}::Intro"],
            "skipped": [f"{source_name}::Method"],
            "graph_relationships": {
                "candidates": 1,
                "verifier_approvals": 1,
                "retained": 1,
            },
        }

    Settings, FakeDB, FakeDAG = _stub_live_stack(
        monkeypatch, ingest_result_factory=lambda source_name: {}
    )
    monkeypatch.setattr(validate_mod, "ingest_document", fake_ingest)

    report = run_validation(
        input_dir,
        output_dir,
        client=_complete_client(output_dir),
        settings_factory=Settings,
        llm_factory=lambda settings: object(),
        db_factory=lambda settings, llm: FakeDB(),
        dag_factory=lambda settings: FakeDAG(),
        evaluate_fn=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("evaluation should not run with unresolved cases")
        ),
    )

    assert len(received_callbacks) == len(PAPER_NAMES)
    assert all(callback is validate_mod._print_ingest_progress for callback in received_callbacks)
    out = capsys.readouterr().out
    assert "[ingest] 1/2 compiled" in out
    assert "[ingest] 2/2 up_to_date" in out
    assert all(item.get("ingestion_status") == "complete" for item in report["papers"])
