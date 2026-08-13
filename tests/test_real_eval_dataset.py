import json
from collections import Counter
from pathlib import Path

from database.document_processor import DocumentProcessor


DATASET_PATH = Path("data/eval_real_papers.json")
ALLOWED_SOURCES = {"attention.pdf", "qlora_paper.pdf", "slm_paper.pdf"}
REQUIRED_CATEGORIES = {"factual", "conceptual", "process", "relational", "cross-paper"}


def test_real_paper_eval_dataset_is_source_scoped_and_matches_parser_sections():
    cases = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

    assert len(cases) >= 20
    assert REQUIRED_CATEGORIES.issubset({case["category"] for case in cases})
    assert all(case.get("query", "").strip() for case in cases)

    parsed_sections = {
        source: {
            section["metadata"]["section"]
            for section in DocumentProcessor().process(Path("data/papers") / source)
        }
        for source in ALLOWED_SOURCES
    }

    source_counts = Counter()
    targeted_counts = Counter()
    for case in cases:
        expected = case.get("expected", [])
        assert expected
        if case.get("target_file"):
            assert case["target_file"] in ALLOWED_SOURCES
        for item in expected:
            source = item["source"]
            assert source in ALLOWED_SOURCES
            assert item["section"] in parsed_sections[source]
            source_counts[source] += 1
        if case.get("target_file"):
            targeted_counts[case["target_file"]] += 1

    assert set(targeted_counts) == ALLOWED_SOURCES
    assert all(count >= 10 for count in targeted_counts.values())
    assert sum(case["category"] == "cross-paper" for case in cases) >= 1
    assert all(source_counts[source] >= 10 for source in ALLOWED_SOURCES)
