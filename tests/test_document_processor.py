import json
from types import SimpleNamespace

import pytest

from database.document_processor import DocumentProcessor


class FakePage:
    def __init__(self, text: str) -> None:
        self.text = text

    def extract_text(self) -> str:
        return self.text


def install_pdf(monkeypatch, pages: list[str]) -> None:
    reader = SimpleNamespace(
        is_encrypted=False,
        pages=[FakePage(page) for page in pages],
    )
    monkeypatch.setattr("database.document_processor.PdfReader", lambda _: reader)


def write_mineru_artifacts(
    tmp_path, markdown: str, *, complete: bool = True, state: str = "done"
):
    markdown_path = tmp_path / "mineru-paper.md"
    manifest_path = tmp_path / "mineru-paper.manifest.json"
    markdown_path.write_text(markdown, encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "source": "paper.pdf",
                "page_count": 4,
                "complete": complete,
                "markdown_path": str(markdown_path),
                "chunks": [
                    {
                        "page_range": "1-4",
                        "start_page": 1,
                        "end_page": 4,
                        "task_id": "task-1",
                        "state": state,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return markdown_path, manifest_path


def _long_body(seed: str) -> str:
    """Body long enough to stay its own section after stub merge."""
    return (seed + " ") * 40


def test_mineru_markdown_keeps_body_provenance_and_omits_noise(tmp_path):
    abstract_body = _long_body("A summary of the work.")
    intro_body = _long_body("The body starts here.")
    markdown_path, manifest_path = write_mineru_artifacts(
        tmp_path,
        "# Paper title\nAuthors\n"
        f"## Abstract\n{abstract_body}\n"
        f"## 1 Introduction\n{intro_body}\n"
        "## We also believe this sentence continues\nMore body.\n"
        "## References\n[1] Omit me.\n"
        f"## A Appendix\n{_long_body('Appendix retained after references.')}",
    )

    processor = DocumentProcessor()
    sections = processor.process_mineru_markdown(markdown_path, manifest_path)

    assert [section["metadata"]["section"] for section in sections] == [
        "Abstract",
        "1 Introduction",
        "A Appendix",
    ]
    assert "We also believe this sentence continues" in sections[1]["page_content"]
    assert "Omit me" not in " ".join(section["page_content"] for section in sections)
    assert "Appendix retained" in sections[2]["page_content"]
    assert sections[0]["metadata"] == {
        "source": "mineru-paper.md",
        "section": "Abstract",
        "mineru_source": "paper.pdf",
        "mineru_page_ranges": ["1-4"],
        "seq_id": 0,
    }
    assert processor.last_report == {
        "retained_section_count": 3,
        "bibliography_omitted": True,
        "mineru_source": "paper.pdf",
        "mineru_page_ranges": ["1-4"],
    }


def test_mineru_merges_stub_heading_into_next_section(tmp_path):
    real_body = _long_body("Substantial section body with enough detail.")
    markdown_path, manifest_path = write_mineru_artifacts(
        tmp_path,
        "## Abstract\nShort abstract only.\n"
        f"## 1 Introduction\n{real_body}\n"
        "## 2 Method\nTiny.\n"
        f"## 3 Results\n{real_body}",
    )

    processor = DocumentProcessor()
    sections = processor.process_mineru_markdown(markdown_path, manifest_path)

    assert [section["metadata"]["section"] for section in sections] == [
        "1 Introduction",
        "3 Results",
    ]
    assert "Abstract" in sections[0]["page_content"]
    assert "Short abstract only." in sections[0]["page_content"]
    assert "2 Method" in sections[1]["page_content"]
    assert "Tiny." in sections[1]["page_content"]
    assert sections[0]["metadata"]["seq_id"] == 0
    assert sections[1]["metadata"]["seq_id"] == 1


def test_mineru_keeps_trailing_stub_when_no_next_section(tmp_path):
    real_body = _long_body("Main body text that is not a stub.")
    markdown_path, manifest_path = write_mineru_artifacts(
        tmp_path,
        f"## 1 Introduction\n{real_body}\n"
        "## 2 Method\nOnly a few words.",
    )

    sections = DocumentProcessor().process_mineru_markdown(markdown_path, manifest_path)

    assert [section["metadata"]["section"] for section in sections] == [
        "1 Introduction",
        "2 Method",
    ]
    assert sections[1]["page_content"] == "Only a few words."


def test_mineru_omits_references_body_but_keeps_later_lettered_section(tmp_path):
    intro_body = _long_body("Introduction body retained.")
    appendix_body = _long_body("Appendix body after the bibliography.")
    markdown_path, manifest_path = write_mineru_artifacts(
        tmp_path,
        f"## 1 Introduction\n{intro_body}\n"
        "## References\n"
        "[1] First citation that must not appear.\n"
        "[2] Second citation that must not appear.\n"
        f"## A Something\n{appendix_body}",
    )

    processor = DocumentProcessor()
    sections = processor.process_mineru_markdown(markdown_path, manifest_path)

    assert [section["metadata"]["section"] for section in sections] == [
        "1 Introduction",
        "A Something",
    ]
    joined = " ".join(section["page_content"] for section in sections)
    assert "citation that must not appear" not in joined
    assert "Appendix body after the bibliography." in sections[1]["page_content"]
    assert processor.last_report["bibliography_omitted"] is True


def test_mineru_keeps_appendix_heading_after_references(tmp_path):
    intro_body = _long_body("Introduction body retained.")
    appendix_body = _long_body("Plain appendix body retained.")
    markdown_path, manifest_path = write_mineru_artifacts(
        tmp_path,
        f"## Abstract\n{_long_body('Abstract summary text.')}\n"
        f"## 1 Introduction\n{intro_body}\n"
        "## References\n[1] Bibliography only.\n"
        f"## Appendix\n{appendix_body}",
    )

    sections = DocumentProcessor().process_mineru_markdown(markdown_path, manifest_path)

    titles = [section["metadata"]["section"] for section in sections]
    assert "Appendix" in titles
    assert "Bibliography only" not in " ".join(s["page_content"] for s in sections)


def test_process_markdown_merges_stub_sections():
    real_body = _long_body("Enough markdown body text for a real section.")
    sections = DocumentProcessor().process_markdown(
        f"## Stub Heading\nBrief.\n## Real Heading\n{real_body}\n",
        "sample.md",
    )

    assert [section["metadata"]["section"] for section in sections] == ["Real Heading"]
    assert "Stub Heading" in sections[0]["page_content"]
    assert "Brief." in sections[0]["page_content"]
    assert sections[0]["metadata"]["seq_id"] == 0


@pytest.mark.parametrize(
    "complete,state,error",
    [(False, "done", "incomplete"), (True, "failed", "unfinished")],
)
def test_mineru_manifest_must_be_complete_before_parsing(
    tmp_path, complete, state, error
):
    markdown_path, manifest_path = write_mineru_artifacts(
        tmp_path, "## Abstract\nBody.", complete=complete, state=state
    )

    with pytest.raises(ValueError, match=error):
        DocumentProcessor().process_mineru_markdown(markdown_path, manifest_path)


def test_mineru_manifest_must_match_selected_markdown(tmp_path):
    markdown_path, manifest_path = write_mineru_artifacts(
        tmp_path, "## Abstract\nBody."
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["markdown_path"] = str(tmp_path / "other.md")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="does not belong"):
        DocumentProcessor().process_mineru_markdown(markdown_path, manifest_path)


def test_mineru_manifest_ranges_must_cover_the_pdf(tmp_path):
    markdown_path, manifest_path = write_mineru_artifacts(
        tmp_path, "## Abstract\nBody."
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["chunks"][0]["end_page"] = 3
    manifest["chunks"][0]["page_range"] = "1-3"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="cover the PDF"):
        DocumentProcessor().process_mineru_markdown(markdown_path, manifest_path)


def test_pdf_omits_front_matter_and_everything_from_references(monkeypatch, tmp_path):
    install_pdf(
        monkeypatch,
        [
            "Attention Is All You Need\n"
            "Ashish Vaswani\n"
            "Abstract\n"
            "A short summary.\n"
            "1 Introduction\n"
            "The body starts here.",
            "References\n"
            "[1] A citation that must not be ingested.\n"
            "Appendix\n"
            "Visualization text that must also be omitted.",
        ],
    )
    processor = DocumentProcessor()

    sections = processor.process_pdf(tmp_path / "attention.pdf")

    assert [section["metadata"]["section"] for section in sections] == [
        "Abstract",
        "1 Introduction",
    ]
    assert [section["page_content"] for section in sections] == [
        "A short summary.",
        "The body starts here.",
    ]
    assert [section["metadata"] for section in sections] == [
        {
            "source": "attention.pdf",
            "section": "Abstract",
            "page_start": 1,
            "page_end": 1,
            "seq_id": 0,
        },
        {
            "source": "attention.pdf",
            "section": "1 Introduction",
            "page_start": 1,
            "page_end": 1,
            "seq_id": 1,
        },
    ]
    assert processor.last_report == {
        "retained_section_count": 2,
        "bibliography_omitted": True,
    }


@pytest.mark.parametrize("references_heading", ["References", "8 References", "VII. References"])
def test_pdf_omits_numbered_references_headings(monkeypatch, tmp_path, references_heading):
    install_pdf(
        monkeypatch,
        [
            "Abstract\nSummary.\n1 Introduction\nBody text.",
            f"{references_heading}\n[1] Omitted bibliography.\nAppendix\nAlso omitted.",
        ],
    )
    processor = DocumentProcessor()

    sections = processor.process_pdf(tmp_path / "numbered-references.pdf")

    assert [section["metadata"]["section"] for section in sections] == [
        "Abstract",
        "1 Introduction",
    ]
    assert processor.last_report["bibliography_omitted"] is True


def test_pdf_can_start_at_first_numbered_heading_without_an_abstract(monkeypatch, tmp_path):
    install_pdf(
        monkeypatch,
        [
            "Paper title\n"
            "Author names\n"
            "1 Introduction\n"
            "Introduction text.\n"
            "2 Method\n"
            "Method text."
        ],
    )
    processor = DocumentProcessor()

    sections = processor.process_pdf(tmp_path / "numbered.pdf")

    assert [section["metadata"]["section"] for section in sections] == [
        "1 Introduction",
        "2 Method",
    ]
    assert all("Paper title" not in section["page_content"] for section in sections)
    assert processor.last_report == {
        "retained_section_count": 2,
        "bibliography_omitted": False,
    }


def test_punctuated_numbered_heading_can_start_pdf_body(monkeypatch, tmp_path):
    install_pdf(
        monkeypatch,
        [
            "Paper title\n"
            "Authors\n"
            "4 QLoRA vs. Standard Finetuning\n"
            "Comparison text."
        ],
    )

    sections = DocumentProcessor().process_pdf(tmp_path / "punctuated.pdf")

    assert [section["metadata"]["section"] for section in sections] == [
        "4 QLoRA vs. Standard Finetuning"
    ]
    assert sections[0]["page_content"] == "Comparison text."


def test_formula_fragment_stays_content_while_real_heading_splits_section(
    monkeypatch, tmp_path
):
    install_pdf(
        monkeypatch,
        [
            "Abstract\n"
            "Summary.\n"
            "3 QLoRA Finetuning\n"
            "YBF16 = XBF16 + XBF16 LBF16\n"
            "1 LBF16\n"
            "2 , (5)\n"
            "4 QLoRA vs. Standard Finetuning\n"
            "Comparison text."
        ],
    )

    sections = DocumentProcessor().process_pdf(tmp_path / "formula.pdf")

    assert [section["metadata"]["section"] for section in sections] == [
        "Abstract",
        "3 QLoRA Finetuning",
        "4 QLoRA vs. Standard Finetuning",
    ]
    assert "1 LBF16" in sections[1]["page_content"]


@pytest.mark.parametrize(
    "heading",
    [
        "2 Method",
        "2 Method:",
        "2 BERT",
        "2 GPT-4",
        "5.1 Experimental setup",
        "III Discussion",
        "III. Discussion",
    ],
)
def test_supported_compact_numbered_headings_remain_eligible(heading):
    assert DocumentProcessor._is_pdf_heading(heading)
    assert DocumentProcessor._is_pdf_body_start(heading)


@pytest.mark.parametrize(
    "line",
    [
        "1 LBF16",
        "2014 English-French dataset details continue this section.",
        "1 This is a numbered sentence.",
        "2 Is this a numbered question?",
        "3 This is a numbered exclamation!",
    ],
)
def test_formula_year_and_sentence_lines_are_not_numbered_headings(line):
    assert not DocumentProcessor._is_pdf_heading(line)
    assert not DocumentProcessor._is_pdf_body_start(line)


def test_four_digit_year_prose_remains_in_its_current_section(monkeypatch, tmp_path):
    install_pdf(
        monkeypatch,
        [
            "Abstract\n"
            "Summary.\n"
            "5 Training Data and Batching\n"
            "2014 English-French dataset details continue this section.\n"
            "The remaining training description stays together."
        ],
    )
    processor = DocumentProcessor()

    sections = processor.process_pdf(tmp_path / "training.pdf")

    assert [section["metadata"]["section"] for section in sections] == [
        "Abstract",
        "5 Training Data and Batching",
    ]
    assert sections[1]["page_content"] == (
        "2014 English-French dataset details continue this section.\n"
        "The remaining training description stays together."
    )
    assert not DocumentProcessor._is_pdf_heading(
        "2014 English-French dataset details continue this section."
    )
