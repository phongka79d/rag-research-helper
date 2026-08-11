"""Best-effort PDF and Markdown section parsing for research papers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pypdf import PdfReader

MARKDOWN_HEADING = re.compile(r"^(#{1,2})\s+(.+?)\s*#*\s*$")
NUMBERED_HEADING = re.compile(
    r"^(?:(?:\d+(?:\.\d+)*)|(?:[IVXLC]+))[.)]?\s+[A-Z][A-Za-z0-9 ,:;()/_-]{1,100}$"
)
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
        return self._with_sequence_ids(sections)

    def process_pdf(self, file_path: str | Path) -> list[dict[str, Any]]:
        path = Path(file_path)
        reader = PdfReader(str(path))
        if reader.is_encrypted and not reader.decrypt(""):
            raise RuntimeError(f"Cannot extract text from encrypted PDF: {path.name}")

        sections: list[dict[str, Any]] = []
        title = "Introduction"
        lines: list[str] = []
        page_start = 1
        page_end = 1

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
                if self._is_pdf_heading(line):
                    flush()
                    title = line
                    page_start = page_number
                    page_end = page_number
                else:
                    if not lines:
                        page_start = page_number
                    page_end = page_number
                    lines.append(line)
        flush()
        return self._with_sequence_ids(sections)

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
        return bool(NUMBERED_HEADING.match(line))
