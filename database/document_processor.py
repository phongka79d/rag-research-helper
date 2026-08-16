"""Best-effort PDF and Markdown section parsing for research papers."""

from __future__ import annotations

import re
import json
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
# Same threshold as thin-section graph skip in data_ingestion.
STUB_BODY_MAX_CHARS = 200


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

    def process_mineru_markdown(
        self, markdown_path: str | Path, manifest_path: str | Path
    ) -> list[dict[str, Any]]:
        """Parse only a complete, explicitly selected MinerU Markdown artifact."""
        markdown_file = Path(markdown_path).expanduser().resolve()
        manifest_file = Path(manifest_path).expanduser().resolve()
        if not markdown_file.is_file():
            raise ValueError(f"MinerU Markdown does not exist: {markdown_file}")
        if not manifest_file.is_file():
            raise ValueError(f"MinerU manifest does not exist: {manifest_file}")
        try:
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("MinerU manifest is not valid JSON") from error
        if not isinstance(manifest, dict):
            raise ValueError("MinerU manifest must be a JSON object")
        if manifest.get("complete") is not True:
            raise ValueError("MinerU manifest is incomplete; ingestion is not allowed")
        chunks = manifest.get("chunks")
        if not isinstance(chunks, list) or not chunks:
            raise ValueError("MinerU manifest has no extraction chunks")
        page_count = manifest.get("page_count")
        if isinstance(page_count, bool) or not isinstance(page_count, int) or page_count < 1:
            raise ValueError("MinerU manifest has an invalid page count")
        expected_start = 1
        if any(
            not isinstance(chunk, dict)
            or chunk.get("state") != "done"
            or not str(chunk.get("page_range", "")).strip()
            for chunk in chunks
        ):
            raise ValueError("MinerU manifest contains an unfinished extraction chunk")
        for chunk in chunks:
            start_page = chunk.get("start_page")
            end_page = chunk.get("end_page")
            page_range = str(chunk.get("page_range", "")).strip()
            range_match = re.fullmatch(r"(\d+)-(\d+)", page_range)
            if (
                isinstance(start_page, bool)
                or isinstance(end_page, bool)
                or not isinstance(start_page, int)
                or not isinstance(end_page, int)
                or range_match is None
                or int(range_match.group(1)) != start_page
                or int(range_match.group(2)) != end_page
                or start_page != expected_start
                or end_page < start_page
                or end_page > page_count
            ):
                raise ValueError("MinerU manifest page ranges do not cover the PDF")
            expected_start = end_page + 1
        if expected_start != page_count + 1:
            raise ValueError("MinerU manifest page ranges do not cover the PDF")
        recorded_markdown = manifest.get("markdown_path")
        if recorded_markdown:
            recorded_path = Path(str(recorded_markdown)).expanduser().resolve()
            if recorded_path != markdown_file:
                raise ValueError("MinerU manifest does not belong to the selected Markdown")
        original_source = str(manifest.get("source", "")).strip()
        if not original_source:
            raise ValueError("MinerU manifest does not identify the original PDF")
        page_ranges = [str(chunk["page_range"]).strip() for chunk in chunks]
        markdown = markdown_file.read_text(encoding="utf-8")
        return self._process_mineru_markdown(
            markdown,
            markdown_file.name,
            original_source,
            page_ranges,
        )

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
        sections = self._merge_stub_sections(sections)
        sections = self._with_sequence_ids(sections)
        self._set_report(sections, bibliography_omitted=False)
        return sections

    def _process_mineru_markdown(
        self,
        markdown_text: str,
        source: str,
        original_source: str,
        page_ranges: list[str],
    ) -> list[dict[str, Any]]:
        sections: list[dict[str, Any]] = []
        title = ""
        lines: list[str] = []
        body_started = False
        bibliography_omitted = False
        in_bibliography = False

        def flush() -> None:
            content = "\n".join(lines).strip()
            if body_started and content and title:
                sections.append(
                    {
                        "page_content": content,
                        "metadata": {
                            "source": source,
                            "section": title,
                            "mineru_source": original_source,
                            "mineru_page_ranges": list(page_ranges),
                        },
                    }
                )
            lines.clear()

        for raw_line in markdown_text.splitlines():
            line = raw_line.strip()
            match = MARKDOWN_HEADING.match(line)
            heading = match.group(2).strip() if match else ""
            if heading:
                if self._is_references_heading(heading) and body_started:
                    flush()
                    bibliography_omitted = True
                    in_bibliography = True
                    continue
                if in_bibliography:
                    if self._is_mineru_section_heading(heading):
                        in_bibliography = False
                        title = heading
                    continue
                if not body_started:
                    if self._is_mineru_body_start(heading):
                        body_started = True
                        title = heading
                    continue
                if self._is_mineru_section_heading(heading):
                    flush()
                    title = heading
                else:
                    # Flash occasionally promotes a prose fragment to a heading;
                    # preserve its text in the surrounding section.
                    lines.append(heading)
                continue
            if body_started and not in_bibliography:
                lines.append(line)
        flush()
        sections = self._merge_stub_sections(sections)
        sections = self._with_sequence_ids(sections)
        self._set_report(
            sections,
            bibliography_omitted=bibliography_omitted,
            mineru_source=original_source,
            mineru_page_ranges=page_ranges,
        )
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
        self,
        sections: list[dict[str, Any]],
        bibliography_omitted: bool,
        **extra: Any,
    ) -> None:
        self.last_report = {
            "retained_section_count": len(sections),
            "bibliography_omitted": bibliography_omitted,
            **extra,
        }

    @staticmethod
    def _with_sequence_ids(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for seq_id, section in enumerate(sections):
            section["metadata"]["seq_id"] = seq_id
        return sections

    @staticmethod
    def _merge_stub_sections(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Fold bodies under STUB_BODY_MAX_CHARS into the following section."""
        if not sections:
            return sections
        merged: list[dict[str, Any]] = []
        pending: list[str] = []
        last_index = len(sections) - 1
        for index, section in enumerate(sections):
            body = section["page_content"].strip()
            is_stub = len(body) < STUB_BODY_MAX_CHARS
            if is_stub and index < last_index:
                heading = str(section["metadata"].get("section", "")).strip()
                pending.append(f"{heading}\n{body}" if body else heading)
                continue
            if pending:
                prefix = "\n".join(pending)
                section = {
                    "page_content": f"{prefix}\n{section['page_content']}".strip(),
                    "metadata": dict(section["metadata"]),
                }
                pending = []
            merged.append(section)
        return merged

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
    def _is_mineru_body_start(line: str) -> bool:
        normalized = line.lower().rstrip(":")
        return normalized == "abstract" or DocumentProcessor._is_numbered_heading(line)

    @staticmethod
    def _is_mineru_section_heading(line: str) -> bool:
        normalized = line.lower().rstrip(":")
        if normalized == "references":
            return False
        return (
            normalized in COMMON_PAPER_HEADINGS
            or DocumentProcessor._is_numbered_heading(line)
            or bool(re.match(r"^[A-Z](?:\.\d+)*[.)]?\s+[A-Z]", line))
        )

    @staticmethod
    def _is_references_heading(line: str) -> bool:
        normalized = SECTION_PREFIX.sub("", line).lower().rstrip(":")
        return normalized == "references"
