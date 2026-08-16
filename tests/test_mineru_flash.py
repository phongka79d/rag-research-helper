import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pypdf import PdfWriter

from scripts.mineru_flash import (
    DEFAULT_BATCH_SIZE,
    MAX_BATCH_SIZE,
    MAX_FILE_BYTES,
    HTTPResponse,
    MinerUFlashClient,
    MinerUValidationError,
    build_page_ranges,
    build_parser,
)


def make_pdf(path: Path, pages: int) -> None:
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=72, height=72)
    with path.open("wb") as handle:
        writer.write(handle)


class FakeTransport:
    def __init__(self, *, failed_ranges=(), pending_polls=0):
        self.calls = []
        self.failed_ranges = set(failed_ranges)
        self.pending_polls = pending_polls
        self.poll_count = {}
        self.task_ranges = {}

    def request(self, method, url, *, data=None, headers=None, timeout=None):
        self.calls.append(
            {
                "method": method,
                "url": url,
                "data": data,
                "headers": headers or {},
                "timeout": timeout,
            }
        )
        if method == "POST":
            payload = json.loads(data.decode("utf-8"))
            page_range = payload["page_range"]
            task_id = f"task-{page_range}"
            self.task_ranges[task_id] = page_range
            return HTTPResponse(
                200,
                json.dumps(
                    {
                        "code": 0,
                        "data": {
                            "file_url": f"https://upload.test/{task_id}",
                            "task_id": task_id,
                        },
                    }
                ).encode(),
            )
        if method == "PUT":
            return HTTPResponse(200, b"")
        if method == "GET" and "/parse/" in url:
            task_id = url.rsplit("/", 1)[-1]
            count = self.poll_count.get(task_id, 0)
            self.poll_count[task_id] = count + 1
            page_range = self.task_ranges[task_id]
            if page_range in self.failed_ranges:
                return HTTPResponse(200, json.dumps({"data": {"state": "failed", "msg": "bad page"}}).encode())
            if count < self.pending_polls:
                return HTTPResponse(200, json.dumps({"data": {"state": "running"}}).encode())
            return HTTPResponse(
                200,
                json.dumps(
                    {"data": {"state": "done", "markdown_url": f"https://download.test/{task_id}"}}
                ).encode(),
            )
        if method == "GET":
            task_id = url.rsplit("/", 1)[-1]
            return HTTPResponse(200, f"Markdown for {self.task_ranges[task_id]}".encode())
        raise AssertionError(f"unexpected request: {method} {url}")


def test_build_page_ranges_cover_pages_once_and_respect_default_and_limit():
    assert DEFAULT_BATCH_SIZE == 10
    assert [item.value for item in build_page_ranges(26)] == ["1-10", "11-20", "21-26"]
    assert build_page_ranges(1)[0].as_dict() == {
        "page_range": "1-1",
        "start_page": 1,
        "end_page": 1,
    }
    assert build_page_ranges(40, MAX_BATCH_SIZE)[-1].value == "21-40"


@pytest.mark.parametrize("batch_size", [0, -1, MAX_BATCH_SIZE + 1])
def test_invalid_batch_size_fails_before_work(batch_size):
    with pytest.raises(MinerUValidationError, match="batch size"):
        build_page_ranges(3, batch_size)


def test_signed_upload_poll_download_and_ordered_merge(tmp_path):
    source = tmp_path / "paper.pdf"
    make_pdf(source, 21)
    transport = FakeTransport(pending_polls=1)
    result = MinerUFlashClient(
        "https://mineru.test/api/v1/agent",
        transport=transport,
        sleep=lambda _: None,
    ).extract(
        source,
        batch_size=10,
        poll_interval=0,
        timeout=10,
        language="en",
        ocr=True,
        table=False,
        formula=True,
    )

    assert result["manifest"]["complete"] is True
    assert result["markdown"] == "Markdown for 1-10\n\nMarkdown for 11-20\n\nMarkdown for 21-21"
    posts = [call for call in transport.calls if call["method"] == "POST"]
    assert [json.loads(call["data"].decode())["page_range"] for call in posts] == [
        "1-10",
        "11-20",
        "21-21",
    ]
    assert json.loads(posts[0]["data"].decode())["is_ocr"] is True
    assert json.loads(posts[0]["data"].decode())["enable_table"] is False
    puts = [call for call in transport.calls if call["method"] == "PUT"]
    assert len(puts) == 3
    assert all(call["headers"]["Content-Type"] == "" for call in puts)
    assert result["manifest"]["chunks"][1]["task_id"] == "task-11-20"


def test_partial_output_is_not_complete_and_manifest_is_written(tmp_path):
    source = tmp_path / "paper.pdf"
    make_pdf(source, 15)
    output_dir = tmp_path / "out"
    transport = FakeTransport(failed_ranges={"11-15"})
    result = MinerUFlashClient(transport=transport).extract(
        source,
        output_path=output_dir,
        poll_interval=0,
        timeout=10,
    )

    assert result["manifest"]["complete"] is False
    assert result["manifest"]["partial"] is True
    assert "Markdown for 1-10" in result["markdown"]
    assert "Markdown for 11-15" not in result["markdown"]
    assert result["manifest"]["chunks"][1]["state"] == "failed"
    assert "bad page" in result["manifest"]["chunks"][1]["error"]
    manifest_path = output_dir / "paper.manifest.json"
    markdown_path = output_dir / "paper.md"
    assert manifest_path.exists()
    assert markdown_path.read_text(encoding="utf-8") == result["markdown"]
    persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert persisted["complete"] is False
    assert persisted["markdown_path"].endswith("paper.md")


def test_timeout_is_recorded_without_leaking_signed_url(tmp_path):
    source = tmp_path / "paper.pdf"
    make_pdf(source, 1)
    transport = FakeTransport(pending_polls=100)
    current = [0.0]

    def clock():
        return current[0]

    def sleep(seconds):
        current[0] += max(seconds, 0.1)

    result = MinerUFlashClient(transport=transport, clock=clock, sleep=sleep).extract(
        source,
        poll_interval=1,
        timeout=1,
    )
    chunk = result["manifest"]["chunks"][0]
    assert result["manifest"]["complete"] is False
    assert chunk["state"] == "timeout"
    assert "timed out" in chunk["error"]
    assert "upload.test" not in json.dumps(result)


def test_size_limit_is_checked_before_pdf_reader(tmp_path, monkeypatch):
    source = tmp_path / "large.pdf"
    with source.open("wb") as handle:
        handle.truncate(MAX_FILE_BYTES + 1)
    client = MinerUFlashClient()
    monkeypatch.setattr("scripts.mineru_flash.PdfReader", SimpleNamespace)
    with pytest.raises(MinerUValidationError, match="10 MB"):
        client.extract(source)


def test_cli_exposes_requested_conservative_options():
    args = build_parser().parse_args(
        [
            "paper.pdf",
            "--output",
            "out",
            "--batch-size",
            "7",
            "--poll-interval",
            "2",
            "--timeout",
            "30",
            "--language",
            "en",
            "--ocr",
            "true",
            "--table",
            "false",
            "--formula",
            "true",
        ]
    )
    assert args.input == Path("paper.pdf")
    assert args.output == Path("out")
    assert (args.batch_size, args.poll_interval, args.timeout) == (7, 2, 30)
    assert (args.language, args.ocr, args.table, args.formula) == ("en", True, False, True)
