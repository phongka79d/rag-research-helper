from main import build_ingestion_summary
from runtime.engine import (
    edges_to_mermaid,
    format_graph_context_lines,
    format_relation_line,
)


def test_ingestion_summary_keeps_counts_report_and_elapsed_time():
    summary = build_ingestion_summary(
        {
            "ingested": ["paper.pdf::Abstract", "paper.pdf::Method"],
            "skipped": ["paper.pdf::Conclusion"],
            "report": {"bibliography_omitted": True},
            "graph_relationships": {
                "candidates": 7,
                "verifier_approvals": 4,
                "retained": 3,
            },
        },
        12.5,
    )

    assert summary == {
        "compiled": 2,
        "up_to_date": 1,
        "bibliography_omitted": True,
        "candidate_relationships": 7,
        "approved_relationships": 4,
        "retained_relationships": 3,
        "elapsed_seconds": 12.5,
    }


def test_ingestion_summary_handles_a_missing_parser_report():
    summary = build_ingestion_summary(
        {"ingested": [], "skipped": []},
        0.0,
    )

    assert summary["bibliography_omitted"] is False
    assert summary["candidate_relationships"] == 0
    assert summary["approved_relationships"] == 0
    assert summary["retained_relationships"] == 0


def test_format_relation_line_uses_relation_or_label():
    assert (
        format_relation_line(
            {"source": "Matrix", "relation": "PREREQUISITE_OF", "target": "LoRA"}
        )
        == "Matrix —PREREQUISITE_OF→ LoRA"
    )
    assert (
        format_relation_line(
            {"source": "QLoRA", "label": "EXTENDS", "target": "LoRA"}
        )
        == "QLoRA —EXTENDS→ LoRA"
    )


def test_format_graph_context_lines_accepts_list_or_single_dict():
    assert format_graph_context_lines([]) == []
    assert format_graph_context_lines(None) == []
    assert format_graph_context_lines(
        [{"source": "A", "relation": "USES", "target": "B"}]
    ) == ["A —USES→ B"]
    assert format_graph_context_lines(
        {"source": "A", "label": "PART_OF", "target": "B"}
    ) == ["A —PART_OF→ B"]


def test_edges_to_mermaid_builds_graph_lr_and_escapes_special_chars():
    mermaid = edges_to_mermaid(
        {
            "nodes": [
                {"id": 'Node "A"'},
                {"id": "Lone"},
            ],
            "edges": [
                {
                    "source": 'Node "A"',
                    "label": "USES|x",
                    "target": "Target (B)",
                }
            ],
        }
    )

    assert mermaid.startswith("graph LR")
    assert 'n0["Node #quot;A#quot;"]' in mermaid
    assert "|USES#124;x|" in mermaid
    assert 'n1["Target #40;B#41;"]' in mermaid
    assert 'n2["Lone"]' in mermaid
    assert 'Node "A"' not in mermaid
    assert "USES|x" not in mermaid


def test_edges_to_mermaid_handles_empty_payload():
    assert edges_to_mermaid({}) == "graph LR"
    assert edges_to_mermaid(None) == "graph LR"
