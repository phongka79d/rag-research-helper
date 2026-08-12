"""Best-effort PDF and Markdown section parsing for research papers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pypdf import PdfReader

MARKDOWN_HEADING = re.compile(r"^(#{1,2})\s+(.+?)\s*#*\s*$")
NUMBERED_HEADING = re.compile(
    r"^(?:(?:\d+(?:\.\d+)*)|(?:[IVXLC]+))[.)]?\s+"
    r"(?P<title>[A-Z][A-Za-z0-9 .,:;()/_-]{1,100})$"
)
FOUR_DIGIT_YEAR_PREFIX = re.compile(r"^\d{4}\b")
SECTION_PREFIX = re.compile(r"^(?:(?:\d+(?:\.\d+)*)|(?:[IVXLC]+))[.)]?\s+", re.IGNORECASE)
FORMULA_LIKE_TITLE = re.compile(r"^[A-Z]+\d+$")
COMMON_PAPER_HEADINGS = {
    "abstract",
    "introduction",
    "background",
    "related work",
    "method",
    "methods",
    "methodology",
    "experiments",
    "experimental setup",
    "results",
    "discussion",
    "conclusion",
    "conclusions",
    "limitations",
    "references",
    "appendix",
}


class DocumentProcessor:
    """Parse a paper into ordered full-section dictionaries."""

    def __init__(self) -> None:
        # Kept beside parsing so callers can report what this one parse retained.
        self.last_report: dict[str, Any] = {
            "retained_section_count": 0,
            "bibliography_omitted": False,
        }

    def process(self, file_path: str | Path) -> list[dict[str, Any]]:
        path = Path(file_path)
        suffix = path.suffix.lower()
        if suffix in {".md", ".markdown"}:
            return self.process_markdown(path.read_text(encoding="utf-8"), path.name)
        if suffix == ".pdf":
            return self.process_pdf(path)
        raise ValueError("Only PDF, Markdown, and .md files can be ingested.")

    def process_markdown(self, markdown_text: str, source: str) -> list[dict[str, Any]]:
        sections: list[dict[str, Any]] = []
        title = "Introduction"
        lines: list[str] = []

        def flush() -> None:
            content = "\n".join(lines).strip()
            if content:
                sections.append(
                    {
                        "page_content": content,
                        "metadata": {
                            "source": source,
                            "section": title,
                            "page_start": 1,
                            "page_end": 1,
                        },
                    }
                )
            lines.clear()

        for line in markdown_text.splitlines():
            match = MARKDOWN_HEADING.match(line)
            if match:
                flush()
                title = match.group(2).strip()
            else:
                lines.append(line)
        flush()
        sections = self._with_sequence_ids(sections)
        self._set_report(sections, bibliography_omitted=False)
        return sections

    def process_pdf(self, file_path: str | Path) -> list[dict[str, Any]]:
        path = Path(file_path)
        reader = PdfReader(str(path))
        if reader.is_encrypted and not reader.decrypt(""):
            raise RuntimeError(f"Cannot extract text from encrypted PDF: {path.name}")

        sections: list[dict[str, Any]] = []
        title = ""
        lines: list[str] = []
        page_start = 1
        page_end = 1
        body_started = False
        bibliography_omitted = False

        def flush() -> None:
            content = "\n".join(lines).strip()
            if content:
                sections.append(
                    {
                        "page_content": content,
                        "metadata": {
                            "source": path.name,
                            "section": title,
                            "page_start": page_start,
                            "page_end": page_end,
                        },
                    }
                )
            lines.clear()

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            for raw_line in text.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                if body_started and self._is_references_heading(line):
                    flush()
                    bibliography_omitted = True
                    break
                if not body_started:
                    if not self._is_pdf_body_start(line):
                        continue
                    body_started = True
                    title = line
                    page_start = page_number
                    page_end = page_number
                    continue
                if self._is_pdf_heading(line):
                    flush()
                    title = line
                    page_start = page_number
                    page_end = page_number
                else:
                    page_end = page_number
                    lines.append(line)
            if bibliography_omitted:
                break
        flush()
        sections = self._with_sequence_ids(sections)
        self._set_report(sections, bibliography_omitted=bibliography_omitted)
        return sections

    def _set_report(
        self, sections: list[dict[str, Any]], bibliography_omitted: bool
    ) -> None:
        self.last_report = {
            "retained_section_count": len(sections),
            "bibliography_omitted": bibliography_omitted,
        }

    @staticmethod
    def _with_sequence_ids(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for seq_id, section in enumerate(sections):
            section["metadata"]["seq_id"] = seq_id
        return sections

    @staticmethod
    def _is_pdf_heading(line: str) -> bool:
        normalized = line.lower().rstrip(":")
        if normalized in COMMON_PAPER_HEADINGS:
            return True
        return DocumentProcessor._is_numbered_heading(line)

    @staticmethod
    def _is_numbered_heading(line: str) -> bool:
        if FOUR_DIGIT_YEAR_PREFIX.match(line) or line.endswith((".", "?", "!")):
            return False
        match = NUMBERED_HEADING.match(line)
        if not match:
            return False
        title = match.group("title").rstrip(":")
        return not FORMULA_LIKE_TITLE.fullmatch(title)

    @staticmethod
    def _is_pdf_body_start(line: str) -> bool:
        normalized = line.lower().rstrip(":")
        return normalized == "abstract" or DocumentProcessor._is_numbered_heading(line)

    @staticmethod
    def _is_references_heading(line: str) -> bool:
        normalized = SECTION_PREFIX.sub("", line).lower().rstrip(":")
        return normalized == "references"
