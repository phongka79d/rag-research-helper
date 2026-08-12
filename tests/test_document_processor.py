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
