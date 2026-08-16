"""Opt-in local-PDF extraction through MinerU Agent Flash."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pypdf import PdfReader


DEFAULT_BASE_URL = "https://mineru.net/api/v1/agent"
DEFAULT_BATCH_SIZE = 10
MAX_BATCH_SIZE = 20
MAX_FILE_BYTES = 10 * 1024 * 1024
DEFAULT_POLL_INTERVAL = 3.0
DEFAULT_TIMEOUT = 600.0


class MinerUError(RuntimeError):
    """Base error for local validation and MinerU request failures."""


class MinerUValidationError(MinerUError, ValueError):
    """Raised before a request is submitted when local input is invalid."""


class MinerUAPIError(MinerUError):
    """Raised for HTTP, API, or task failures."""


class MinerUTimeoutError(MinerUError):
    """Raised when a task does not finish before its polling deadline."""


@dataclass(frozen=True)
class PageRange:
    start: int
    end: int

    @property
    def value(self) -> str:
        return f"{self.start}-{self.end}"

    def as_dict(self) -> dict[str, Any]:
        return {"page_range": self.value, "start_page": self.start, "end_page": self.end}


@dataclass(frozen=True)
class HTTPResponse:
    status: int
    body: bytes = b""


class UrllibTransport:
    """Small stdlib transport; injected fakes keep tests offline."""

    def request(
        self,
        method: str,
        url: str,
        *,
        data: bytes | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> HTTPResponse:
        request = Request(url, data=data, headers=dict(headers or {}), method=method)
        try:
            with urlopen(request, timeout=timeout) as response:  # nosec B310
                return HTTPResponse(int(response.status), response.read())
        except HTTPError as error:
            return HTTPResponse(int(error.code), error.read())
        except URLError as error:
            raise MinerUAPIError(f"network error: {type(error.reason).__name__}") from error


Transport = Callable[..., HTTPResponse] | Any


def validate_batch_size(batch_size: int, *, maximum: int = MAX_BATCH_SIZE) -> int:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise MinerUValidationError("batch size must be an integer")
    if not 1 <= batch_size <= maximum:
        raise MinerUValidationError(
            f"batch size must be between 1 and {maximum} pages (got {batch_size})"
        )
    return batch_size


def build_page_ranges(
    page_count: int, batch_size: int = DEFAULT_BATCH_SIZE
) -> list[PageRange]:
    """Return ordered, contiguous ranges covering pages 1..page_count once."""
    validate_batch_size(batch_size)
    if isinstance(page_count, bool) or not isinstance(page_count, int) or page_count < 1:
        raise MinerUValidationError("PDF must contain at least one page")
    return [
        PageRange(start, min(start + batch_size - 1, page_count))
        for start in range(1, page_count + 1, batch_size)
    ]


def _safe_error(error: BaseException | str) -> str:
    message = str(error) if not isinstance(error, str) else error
    message = re.sub(r"https?://[^\s)]+", "[url]", message)
    message = re.sub(
        r"(?i)(authorization|api[-_ ]?key|token|password)\s*[:=]\s*[^\s,;]+",
        r"\1=[redacted]",
        message,
    )
    return message[:500] or type(error).__name__


def _json(response: HTTPResponse, endpoint: str) -> dict[str, Any]:
    try:
        payload = json.loads(response.body.decode("utf-8")) if response.body else {}
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MinerUAPIError(f"{endpoint} returned invalid JSON") from error
    if not isinstance(payload, dict):
        raise MinerUAPIError(f"{endpoint} returned an invalid response")
    if payload.get("code", 0) not in (0, "0", 200, "200"):
        detail = payload.get("msg") or payload.get("message") or payload.get("err_msg")
        suffix = f": {_safe_error(detail)}" if detail else ""
        raise MinerUAPIError(f"{endpoint} API error {payload.get('code')}{suffix}")
    return payload


def _data(payload: dict[str, Any], endpoint: str) -> dict[str, Any]:
    data = payload.get("data")
    if not isinstance(data, dict):
        raise MinerUAPIError(f"{endpoint} response did not include data")
    return data


class MinerUFlashClient:
    """Serial signed-upload client for arbitrary local PDFs."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        *,
        transport: Transport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.transport = transport or UrllibTransport()
        self.sleep = sleep
        self.clock = clock

    def _request(
        self,
        method: str,
        url: str,
        *,
        data: bytes | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        endpoint: str,
    ) -> HTTPResponse:
        try:
            if hasattr(self.transport, "request"):
                response = self.transport.request(
                    method, url, data=data, headers=headers, timeout=timeout
                )
            else:
                response = self.transport(
                    method, url, data=data, headers=headers, timeout=timeout
                )
        except MinerUError:
            raise
        except Exception as error:
            raise MinerUAPIError(f"{endpoint} request failed: {_safe_error(error)}") from error
        if not isinstance(response, HTTPResponse):
            raise MinerUAPIError(f"{endpoint} returned an invalid response")
        if not 200 <= response.status < 300:
            detail = ""
            try:
                body = _json(response, endpoint)
                detail = str(body.get("msg") or body.get("message") or "")
            except MinerUError:
                pass
            suffix = f": {_safe_error(detail)}" if detail else ""
            raise MinerUAPIError(f"{endpoint} HTTP {response.status}{suffix}")
        return response

    def _url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    def _submit(
        self, filename: str, page_range: PageRange, options: Mapping[str, Any]
    ) -> tuple[str, str]:
        request = {"file_name": filename, "page_range": page_range.value}
        request.update({key: value for key, value in options.items() if value is not None})
        response = self._request(
            "POST",
            self._url("parse/file"),
            data=json.dumps(request).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            timeout=30,
            endpoint="submit",
        )
        data = _data(_json(response, "submit"), "submit")
        task_id, file_url = data.get("task_id"), data.get("file_url")
        if not task_id or not file_url:
            raise MinerUAPIError("submit response did not include task_id and file_url")
        return str(file_url), str(task_id)

    def _upload(self, file_url: str, content: bytes) -> None:
        self._request(
            "PUT",
            file_url,
            data=content,
            # MinerU's OSS signed URL is generated without a content type.
            # An empty value prevents urllib from injecting its default form
            # content type, which would invalidate the signed request.
            headers={"Content-Type": ""},
            timeout=60,
            endpoint="upload",
        )

    def _poll(self, task_id: str, *, poll_interval: float, timeout: float) -> tuple[str, dict[str, Any]]:
        deadline = self.clock() + timeout
        while True:
            response = self._request(
                "GET",
                self._url(f"parse/{task_id}"),
                headers={"Accept": "application/json"},
                timeout=min(30.0, timeout),
                endpoint="poll",
            )
            data = _data(_json(response, "poll"), "poll")
            state = str(data.get("state", "")).strip().lower()
            if state == "done":
                return state, data
            if state == "failed":
                raise MinerUAPIError(
                    f"task {task_id} failed: {_safe_error(data.get('err_msg') or data.get('msg') or 'unknown error')}"
                )
            if self.clock() >= deadline:
                raise MinerUTimeoutError(f"task {task_id} timed out after {timeout:g}s")
            self.sleep(min(max(poll_interval, 0.0), max(0.0, deadline - self.clock())))

    def _download(self, markdown_url: str) -> str:
        response = self._request(
            "GET",
            markdown_url,
            headers={"Accept": "text/markdown, text/plain, */*"},
            timeout=60,
            endpoint="download",
        )
        try:
            return response.body.decode("utf-8")
        except UnicodeDecodeError as error:
            raise MinerUAPIError("download returned non-UTF-8 Markdown") from error

    @staticmethod
    def _read_pdf(path: Path) -> tuple[int, bytes]:
        if not path.exists() or not path.is_file():
            raise MinerUValidationError(f"input PDF does not exist: {path}")
        if path.stat().st_size > MAX_FILE_BYTES:
            raise MinerUValidationError("input PDF exceeds the MinerU Flash 10 MB limit")
        try:
            page_count = len(PdfReader(str(path)).pages)
        except Exception as error:
            raise MinerUValidationError(
                f"unable to read PDF page count: {_safe_error(error)}"
            ) from error
        if page_count < 1:
            raise MinerUValidationError("PDF must contain at least one page")
        return page_count, path.read_bytes()

    def extract(
        self,
        input_path: str | Path,
        *,
        output_path: str | Path | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        timeout: float = DEFAULT_TIMEOUT,
        language: str | None = None,
        ocr: bool | None = None,
        table: bool | None = None,
        formula: bool | None = None,
    ) -> dict[str, Any]:
        validate_batch_size(batch_size)
        if poll_interval < 0 or timeout <= 0:
            raise MinerUValidationError("poll interval must be >= 0 and timeout must be > 0")
        path = Path(input_path).expanduser().resolve()
        page_count, content = self._read_pdf(path)
        ranges = build_page_ranges(page_count, batch_size)
        options = {
            "language": language,
            "is_ocr": ocr,
            "enable_table": table,
            "enable_formula": formula,
        }
        chunks: list[dict[str, Any]] = []
        markdown: dict[str, str] = {}
        for page_range in ranges:
            chunk = {**page_range.as_dict(), "task_id": None, "state": "pending"}
            try:
                file_url, task_id = self._submit(path.name, page_range, options)
                chunk["task_id"] = task_id
                self._upload(file_url, content)
                state, data = self._poll(
                    task_id, poll_interval=poll_interval, timeout=timeout
                )
                markdown_url = data.get("markdown_url")
                if not markdown_url:
                    raise MinerUAPIError("successful task response did not include markdown_url")
                markdown[page_range.value] = self._download(str(markdown_url))
                chunk.update({"state": state, "markdown_status": "downloaded"})
            except MinerUTimeoutError as error:
                chunk.update({"state": "timeout", "error": _safe_error(error)})
            except Exception as error:
                chunk.update({"state": "failed", "error": _safe_error(error)})
            chunks.append(chunk)

        merged = "\n\n".join(markdown[item.value] for item in ranges if item.value in markdown)
        complete = len(markdown) == len(ranges) and all(
            chunk.get("state") == "done" for chunk in chunks
        )
        manifest: dict[str, Any] = {
            "source": str(path),
            "page_count": page_count,
            "batch_size": batch_size,
            "poll_interval": poll_interval,
            "timeout": timeout,
            "options": {key: value for key, value in options.items() if value is not None},
            "complete": complete,
            "partial": bool(merged) and not complete,
            "chunks": chunks,
        }
        result: dict[str, Any] = {"markdown": merged, "manifest": manifest}
        if output_path is not None:
            markdown_path, manifest_path = _output_paths(output_path, path)
            markdown_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            markdown_path.write_text(merged, encoding="utf-8")
            manifest.update(
                {"markdown_path": str(markdown_path), "manifest_path": str(manifest_path)}
            )
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            result.update({"markdown_path": markdown_path, "manifest_path": manifest_path})
        return result


def _output_paths(output: str | Path, source: Path) -> tuple[Path, Path]:
    target = Path(output).expanduser()
    markdown_path = target if target.suffix.lower() in {".md", ".markdown"} else target / f"{source.stem}.md"
    return markdown_path, markdown_path.with_suffix(".manifest.json")


def _str_to_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected true/false")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="local PDF path")
    parser.add_argument("--output", type=Path, default=None, help="Markdown path or output directory")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--batch-size", "--batch", dest="batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--poll-interval", type=float, default=DEFAULT_POLL_INTERVAL)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="per-batch timeout in seconds")
    parser.add_argument("--language", default=None)
    for name in ("ocr", "table", "formula"):
        parser.add_argument(f"--{name}", nargs="?", const=True, type=_str_to_bool, default=None)
        parser.add_argument(f"--no-{name}", dest=name, action="store_false")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = MinerUFlashClient(args.base_url).extract(
            args.input,
            output_path=args.output,
            batch_size=args.batch_size,
            poll_interval=args.poll_interval,
            timeout=args.timeout,
            language=args.language,
            ocr=args.ocr,
            table=args.table,
            formula=args.formula,
        )
    except MinerUError as error:
        print(f"MinerU Flash extraction failed: {_safe_error(error)}", file=sys.stderr)
        return 1
    manifest = result["manifest"]
    status = "complete" if manifest["complete"] else "partial"
    print(f"MinerU Flash extraction {status}: {len(manifest['chunks'])} batch(es)")
    if result.get("markdown_path"):
        print(f"Markdown: {result['markdown_path']}")
        print(f"Manifest: {result['manifest_path']}")
    return 0 if manifest["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
