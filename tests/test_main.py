from main import build_ingestion_summary


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
