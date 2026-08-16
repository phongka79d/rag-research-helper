"""Ahead-of-time compilation of research-paper sections."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from hashlib import md5, sha256
from pathlib import Path
import json
import re
from typing import Any

from pydantic import ValidationError

from core.relations import (
    RELATION_CUE_WORDS,
    compile_local_pair_patterns,
    is_concept_endpoint as _is_concept_endpoint,
    normalize_relation,
    normalized_phrase as _normalized_phrase,
    quote_supports_relation as _quote_supports_relation,
)
from core.schemas import MAX_GRAPH_VERIFIER_CANDIDATES
from database.document_processor import DocumentProcessor


_DIRECT_WHOLE_PART_CUE = re.compile(
    r"(?:consists\s+of|is\s+composed\s+of|"
    r"is\s+made\s+up\s+of|is\s+(?:an?\s+|the\s+)?(?:component|part|layer|"
    r"module|element)\s+(?:of|in|within)|forms?\s+(?:an?\s+)?part\s+of)\b",
    re.IGNORECASE,
)
_GRAPH_REJECTION_KEYS = (
    "invalid_approval",
    "duplicate_approval",
    "invalid_endpoint",
    "invalid_evidence_id",
    "span_grounding",
    "relation_mismatch",
)
# Match the existing graph-quote bound in core.schemas (max_length=500).
_MAX_EVIDENCE_SPAN_CHARS = 500
# Bounded retained-edge audit for diagnostics only (not Neo4j persistence).
_MAX_RETAINED_EDGE_AUDIT = 10
_MAX_EVIDENCE_PREVIEW_CHARS = 120
# Sections shorter than this still persist text/plan; skip graph extract/verify.
_MIN_GRAPH_SECTION_CHARS = 200
_SENTENCE_BOUNDARY = re.compile(
    r"([.!?]+)([)\]\"'\u2019\u201d]*)(?:\s+|$)"
)
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_MD_STRUCTURAL_LINE = re.compile(
    r"^\s*(?:#{1,6}\s|[-*+]\s|\d+\.\s|\||```)"
)
# Generic noun phrase for local graph scan: optional article + short token run.
# Tokens exclude closed-class/relation-cue words so endpoints stay concise.
_LOCAL_GRAPH_CLOSED_CLASS = (
    r"is|are|was|were|be|been|being|a|an|the|of|in|within|and|or|to|for|"
    r"by|as|on|at|from|with|into|onto|one|how|between|before|after|must|need|"
    r"you|we|up|need|not|no|nor|overall|here|there|also|then|thus|"
    r"therefore|respectively"
)
_LOCAL_GRAPH_CUE_ALTERNATION = "|".join(
    re.escape(word) for word in RELATION_CUE_WORDS
)
_LOCAL_GRAPH_NP_WORD = (
    rf"(?!(?:{_LOCAL_GRAPH_CLOSED_CLASS}|{_LOCAL_GRAPH_CUE_ALTERNATION})\b)"
    r"[A-Za-z0-9][A-Za-z0-9\-]*"
)
_LOCAL_GRAPH_NP = (
    rf"(?:(?:an?|the|each|every|this|that|these|those|any|some|all)\s+)?"
    rf"({_LOCAL_GRAPH_NP_WORD}(?:\s+{_LOCAL_GRAPH_NP_WORD}){{0,5}})"
)
_LOCAL_GRAPH_OPT_ART = (
    r"(?:(?:an?|the|each|every|this|that|these|those|any|some|all)\s+)?"
)
# After consists/composed/made-up-of, allow "of" and "=" so phrases such as
# "stack of N = 6 identical layers" stay one endpoint. Still generic tokens.
_LOCAL_GRAPH_COMPOSED_PART = (
    rf"(?:(?:an?|the|each|every|this|that|these|those|any|some|all)\s+)?"
    rf"({_LOCAL_GRAPH_NP_WORD}"
    rf"(?:(?:\s+of\s+|\s*=\s*|\s+){_LOCAL_GRAPH_NP_WORD}){{0,8}})"
)


def make_parent_id(metadata: dict[str, Any]) -> str:
    """Keep section IDs stable across runs for cache checks and upserts."""
    source = metadata.get("source", "")
    section = metadata.get("section", "")
    # Ponytail: retain the original source-plus-section ID contract to avoid a data migration.
    return md5(f"{source}__{section}".encode("utf-8")).hexdigest()


def make_content_hash(text: str) -> str:
    """Identify the exact source text compiled into a parent section."""
    return sha256(text.encode("utf-8")).hexdigest()


def collect_anchor_nodes(aot: dict[str, Any]) -> list[str]:
    """Collect the AOT graph concepts stored with a parent section."""
    names: list[str] = []
    graph = aot.get("knowledge_graph", {})
    for name in aot.get("main_entities", []):
        if name:
            names.append(str(name).strip())
    for node in graph.get("nodes", []):
        if node.get("name"):
            names.append(str(node["name"]).strip())
    for edge in graph.get("edges", []):
        for endpoint in (edge.get("source"), edge.get("target")):
            if endpoint:
                names.append(str(endpoint).strip())
    return [name for name in dict.fromkeys(names) if name]


def _is_short_abbrev_period(text: str, period_idx: int) -> bool:
    """True when a period follows a short alphabetic token (e.g., e.g., Fig., Eq.)."""
    i = period_idx - 1
    if i < 0 or not text[i].isalpha():
        return False
    start = i
    while start > 0 and text[start - 1].isalpha():
        start -= 1
    return (i - start + 1) <= 3


def _trim_span_bounds(text: str, start: int, end: int) -> tuple[int, int] | None:
    """Shrink [start, end) to non-whitespace content; None if empty."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start >= end:
        return None
    return start, end


def _split_long_range(
    text: str, start: int, end: int, max_len: int = _MAX_EVIDENCE_SPAN_CHARS
) -> list[tuple[int, int]]:
    """Split a range at paragraph, newline, or whitespace before max_len."""
    chunks: list[tuple[int, int]] = []
    i = start
    while i < end:
        trimmed = _trim_span_bounds(text, i, end)
        if trimmed is None:
            break
        i, range_end = trimmed[0], end
        if range_end - i <= max_len:
            final = _trim_span_bounds(text, i, range_end)
            if final is not None:
                chunks.append(final)
            break
        window_end = i + max_len
        window = text[i:window_end]
        cut: int | None = None
        para = max(window.rfind("\r\n\r\n"), window.rfind("\n\n"))
        if para != -1:
            cut = i + para
        else:
            nl = max(window.rfind("\r\n"), window.rfind("\n"))
            if nl != -1:
                cut = i + nl
            else:
                ws = None
                for j in range(len(window) - 1, -1, -1):
                    if window[j].isspace():
                        ws = j
                        break
                if ws is not None and ws > 0:
                    cut = i + ws
                else:
                    cut = window_end
        if cut <= i:
            cut = window_end
        piece = _trim_span_bounds(text, i, cut)
        if piece is not None:
            chunks.append(piece)
        i = cut
    return chunks


def _sentence_spans_in_range(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Sentence-split text[start:end] with conservative abbreviation handling."""
    if start >= end:
        return []
    segment = text[start:end]
    results: list[tuple[int, int]] = []
    local = 0
    n = len(segment)
    while local < n and segment[local].isspace():
        local += 1
    if local >= n:
        return []
    span_local_start = local
    for match in _SENTENCE_BOUNDARY.finditer(segment, local):
        punct = match.group(1)
        punct_start = match.start(1)
        if punct.startswith(".") and _is_short_abbrev_period(segment, punct_start):
            continue
        sentence_end = match.end(2)
        piece = _trim_span_bounds(segment, span_local_start, sentence_end)
        if piece is not None:
            results.append((start + piece[0], start + piece[1]))
        next_start = match.end()
        while next_start < n and segment[next_start].isspace():
            next_start += 1
        span_local_start = next_start
        if span_local_start >= n:
            return results
    if span_local_start < n:
        piece = _trim_span_bounds(segment, span_local_start, n)
        if piece is not None:
            results.append((start + piece[0], start + piece[1]))
    return results


def _prose_spans(text: str) -> list[tuple[int, int]]:
    """Split on paragraph breaks, then sentences within each paragraph."""
    results: list[tuple[int, int]] = []
    idx = 0
    for match in _PARAGRAPH_BREAK.finditer(text):
        results.extend(_sentence_spans_in_range(text, idx, match.start()))
        idx = match.end()
    results.extend(_sentence_spans_in_range(text, idx, len(text)))
    return results


def _is_markdown_heavy(text: str) -> bool:
    """True when many non-empty lines look like Markdown structure."""
    non_empty = [line for line in text.splitlines() if line.strip()]
    if len(non_empty) < 2:
        return False
    structural = sum(1 for line in non_empty if _MD_STRUCTURAL_LINE.match(line))
    return structural >= 2 and structural * 2 >= len(non_empty)


def _line_spans(text: str) -> list[tuple[int, int]]:
    """Split on newlines (including \\r\\n), keeping non-whitespace line content."""
    spans: list[tuple[int, int]] = []
    i = 0
    n = len(text)
    while i < n:
        j = i
        while j < n and text[j] not in "\r\n":
            j += 1
        piece = _trim_span_bounds(text, i, j)
        if piece is not None:
            spans.append(piece)
        if j < n and text[j] == "\r":
            j += 1
            if j < n and text[j] == "\n":
                j += 1
        elif j < n and text[j] == "\n":
            j += 1
        i = j
    return spans


def build_evidence_spans(section_text: str) -> list[dict]:
    """Derive stable, source-preserving evidence spans from section text.

    Spans use generic sentence, paragraph, or Markdown structural boundaries.
    Each span text is a verbatim slice of section_text at [start:end).
    """
    if not isinstance(section_text, str) or not section_text.strip():
        return []

    raw_ranges = (
        _line_spans(section_text)
        if _is_markdown_heavy(section_text)
        else _prose_spans(section_text)
    )
    spans: list[dict] = []
    for start, end in raw_ranges:
        for chunk_start, chunk_end in _split_long_range(section_text, start, end):
            spans.append(
                {
                    "id": f"e{len(spans)}",
                    "start": chunk_start,
                    "end": chunk_end,
                    "text": section_text[chunk_start:chunk_end],
                }
            )
    return spans


def _is_grounded_name(name: Any, normalized_section: str) -> bool:
    phrase = _normalized_phrase(name)
    return bool(phrase) and f" {phrase} " in f" {normalized_section} "


def _local_graph_pair_patterns() -> list[tuple[re.Pattern[str], str, int, int]]:
    """Compile the shared wording table against local noun-phrase macros."""
    return compile_local_pair_patterns(
        noun_phrase=_LOCAL_GRAPH_NP,
        optional_article=_LOCAL_GRAPH_OPT_ART,
        composed_part=_LOCAL_GRAPH_COMPOSED_PART,
    )


_LOCAL_GRAPH_PAIR_PATTERNS = _local_graph_pair_patterns()


def _scan_span_for_relation_pairs(span_text: str) -> list[tuple[str, str, str]]:
    """Extract grounded (source, relation, target) pairs from one evidence text."""
    if not isinstance(span_text, str) or not span_text.strip():
        return []
    normalized_span = _normalized_phrase(span_text)
    pairs: list[tuple[str, str, str]] = []
    for pattern, relation, source_group, target_group in _LOCAL_GRAPH_PAIR_PATTERNS:
        for match in pattern.finditer(span_text):
            source = match.group(source_group).strip()
            target = match.group(target_group).strip()
            if (
                not source
                or not target
                or _normalized_phrase(source) == _normalized_phrase(target)
            ):
                continue
            if not _is_concept_endpoint(source) or not _is_concept_endpoint(target):
                continue
            if not _is_grounded_name(source, normalized_span):
                continue
            if not _is_grounded_name(target, normalized_span):
                continue
            if not _quote_supports_relation(span_text, source, relation, target):
                continue
            pairs.append((source, relation, target))
    return pairs


def _is_int_offset(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def propose_local_graph_candidates(
    section_text: str,
    evidence_spans: list[dict] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Return (candidates, extra_spans).

    candidates: {source, relation, target, evidence_id}
    extra_spans: window spans that must be added to the section span list
                 so the local gate can resolve evidence_id.
    """
    if not isinstance(section_text, str) or not section_text.strip():
        return [], []
    if evidence_spans is None:
        evidence_spans = build_evidence_spans(section_text)
    if not isinstance(evidence_spans, list):
        return [], []

    candidates: list[dict] = []
    seen: set[tuple[str, str, str, str]] = set()
    window_spans: list[dict] = []

    def add_from_text(span_text: str, evidence_id: str) -> bool:
        added = False
        for source, relation, target in _scan_span_for_relation_pairs(span_text):
            key = (
                _normalized_phrase(source),
                relation,
                _normalized_phrase(target),
                evidence_id,
            )
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                {
                    "source": source,
                    "relation": relation,
                    "target": target,
                    "evidence_id": evidence_id,
                }
            )
            added = True
        return added

    for span in evidence_spans:
        if not isinstance(span, dict):
            continue
        span_id = str(span.get("id", "") or "").strip()
        span_text = span.get("text")
        if not span_id or not isinstance(span_text, str) or not span_text.strip():
            continue
        add_from_text(span_text, span_id)

    for index in range(len(evidence_spans) - 1):
        left = evidence_spans[index]
        right = evidence_spans[index + 1]
        if not isinstance(left, dict) or not isinstance(right, dict):
            continue
        left_id = str(left.get("id", "") or "").strip()
        right_id = str(right.get("id", "") or "").strip()
        left_start, left_end = left.get("start"), left.get("end")
        right_start, right_end = right.get("start"), right.get("end")
        if (
            not left_id
            or not right_id
            or not _is_int_offset(left_start)
            or not _is_int_offset(left_end)
            or not _is_int_offset(right_start)
            or not _is_int_offset(right_end)
            or left_start < 0
            or right_end > len(section_text)
            or left_end > right_start
            or left_start >= left_end
            or right_start >= right_end
        ):
            continue
        gap = section_text[left_end:right_start]
        if _PARAGRAPH_BREAK.search(gap):
            continue
        window_start = left_start
        window_end = right_end
        window_text = section_text[window_start:window_end]
        window_id = f"{left_id}+{right_id}"
        if not add_from_text(window_text, window_id):
            continue
        window_spans.append(
            {
                "id": window_id,
                "start": window_start,
                "end": window_end,
                "text": window_text,
            }
        )

    bounded = candidates[:MAX_GRAPH_VERIFIER_CANDIDATES]
    kept_ids = {item["evidence_id"] for item in bounded}
    extra_spans = [span for span in window_spans if span["id"] in kept_ids]
    return bounded, extra_spans


def _grounded_names(values: Any, normalized_section: str) -> list[str]:
    """Keep unique model-proposed names only when this section contains them."""
    if not isinstance(values, list):
        return []
    names: list[str] = []
    seen: set[str] = set()
    for value in values:
        name = str(value).strip()
        normalized_name = _normalized_phrase(name)
        if (
            normalized_name
            and normalized_name not in seen
            and _is_grounded_name(name, normalized_section)
        ):
            names.append(name)
            seen.add(normalized_name)
    return names


def filter_aot_to_section(aot: dict[str, Any], section_text: str) -> dict[str, Any]:
    """Remove AOT concepts that cannot be evidenced in the current section."""
    normalized_section = _normalized_phrase(section_text)
    main_entities = _grounded_names(aot.get("main_entities", []), normalized_section)

    roadmap: list[dict[str, Any]] = []
    for raw_step in aot.get("learning_roadmap", []):
        if not isinstance(raw_step, dict):
            continue
        step = dict(raw_step)
        step["concepts"] = _grounded_names(
            step.get("concepts", []), normalized_section
        )
        if str(step.get("title", "")).strip() and str(
            step.get("content_focus", "")
        ).strip():
            roadmap.append(step)

    raw_graph = aot.get("knowledge_graph", {})
    graph = raw_graph if isinstance(raw_graph, dict) else {}
    nodes: list[dict[str, Any]] = []
    for raw_node in graph.get("nodes", []):
        if not isinstance(raw_node, dict):
            continue
        name = str(raw_node.get("name", "")).strip()
        if _is_grounded_name(name, normalized_section):
            nodes.append({**raw_node, "name": name})

    edges: list[dict[str, Any]] = []
    for raw_edge in graph.get("edges", []):
        if not isinstance(raw_edge, dict):
            continue
        source = str(raw_edge.get("source", "")).strip()
        target = str(raw_edge.get("target", "")).strip()
        if _is_grounded_name(source, normalized_section) and _is_grounded_name(
            target, normalized_section
        ):
            edges.append({**raw_edge, "source": source, "target": target})

    return {
        "main_entities": main_entities,
        "learning_roadmap": roadmap,
        "knowledge_graph": {"nodes": nodes, "edges": edges},
    }


def _evidence_span_lookup(evidence_spans: list[dict]) -> dict[str, dict]:
    """Map span id -> span dict for O(1) resolution during the local gate."""
    lookup: dict[str, dict] = {}
    for span in evidence_spans:
        if not isinstance(span, dict):
            continue
        span_id = span.get("id")
        if span_id is None:
            continue
        key = str(span_id).strip()
        if key:
            lookup[key] = span
    return lookup


def _span_is_verbatim_slice(section_text: str, span: dict) -> bool:
    """True when span text equals section_text[start:end] with in-range offsets."""
    start = span.get("start")
    end = span.get("end")
    text = span.get("text")
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, int)
        or not isinstance(end, int)
        or not isinstance(text, str)
        or start < 0
        or end > len(section_text)
        or start >= end
    ):
        return False
    return section_text[start:end] == text


def _evidence_id_is_window(evidence_id: str) -> bool:
    """Adjacent-window ids join two span ids with '+' (e.g. e0+e1)."""
    return "+" in evidence_id


def _candidate_triple_key(candidate: dict[str, Any]) -> tuple[str, str, str] | None:
    """Normalized (source, relation, target) for union dedup; None if unusable."""
    source = str(candidate.get("source", "")).strip()
    target = str(candidate.get("target", "")).strip()
    relation = normalize_relation(candidate.get("relation"))
    if not source or not target or not relation:
        return None
    return (_normalized_phrase(source), relation, _normalized_phrase(target))


def _candidate_evidence_preference(candidate: dict[str, Any], is_local: bool) -> int:
    """Higher score wins on triple collision; local single-span is best."""
    evidence_id = str(candidate.get("evidence_id", "") or "").strip()
    is_window = _evidence_id_is_window(evidence_id)
    if is_local and not is_window:
        return 3
    if is_local:
        return 2
    if not is_window:
        return 1
    return 0


def _union_graph_candidates(
    local_candidates: list[dict],
    model_edges: Any,
) -> tuple[list[dict[str, Any]], set[int]]:
    """Merge local and model edges; prefer local single-span on triple ties.

    Returns (capped candidates, indexes that came from the local scan).
    """
    if not isinstance(model_edges, list):
        model_edges = []
    chosen: dict[tuple[str, str, str], tuple[dict[str, Any], bool, int]] = {}
    order_keys: list[tuple[str, str, str]] = []

    def consider(raw: Any, is_local: bool) -> None:
        if not isinstance(raw, dict):
            return
        key = _candidate_triple_key(raw)
        if key is None:
            return
        candidate = {
            "source": str(raw.get("source", "")).strip(),
            "relation": normalize_relation(raw.get("relation")),
            "target": str(raw.get("target", "")).strip(),
            "evidence_id": str(raw.get("evidence_id", "") or "").strip(),
        }
        pref = _candidate_evidence_preference(candidate, is_local)
        existing = chosen.get(key)
        if existing is None:
            chosen[key] = (candidate, is_local, pref)
            order_keys.append(key)
            return
        if pref > existing[2]:
            chosen[key] = (candidate, is_local, pref)

    for item in local_candidates:
        consider(item, True)
    for item in model_edges:
        consider(item, False)

    merged: list[dict[str, Any]] = []
    local_indexes: set[int] = set()
    for key in order_keys:
        if len(merged) >= MAX_GRAPH_VERIFIER_CANDIDATES:
            break
        candidate, is_local, _pref = chosen[key]
        if is_local:
            local_indexes.add(len(merged))
        merged.append(candidate)
    return merged, local_indexes


def _approved_graph_edges(
    section_text: str,
    candidates: list[dict[str, Any]],
    approvals: Any,
    rejection_counts: dict[str, int] | None = None,
    evidence_spans: list[dict] | None = None,
    audit_sink: list[dict[str, Any]] | None = None,
    locator: str | None = None,
) -> list[dict[str, Any]]:
    """Keep only candidate edges approved against resolved source evidence spans.

    At most one edge per normalized (source, relation, target) is retained.
    Single-span evidence ids are preferred over adjacent-window ids.
    """
    def reject(reason: str) -> None:
        if rejection_counts is not None and reason in _GRAPH_REJECTION_KEYS:
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

    if not isinstance(approvals, list):
        return []
    if evidence_spans is None:
        evidence_spans = build_evidence_spans(section_text)
    span_by_id = _evidence_span_lookup(evidence_spans)
    normalized_section = _normalized_phrase(section_text)
    # key -> (edge, evidence_id, span_text) for single-triple dedup per section.
    accepted_by_triple: dict[tuple[str, str, str], tuple[dict[str, Any], str, str]] = {}
    index_counts: dict[int, int] = {}
    for approval in approvals:
        if not isinstance(approval, dict):
            reject("invalid_approval")
            continue
        index = approval.get("index")
        if (
            not isinstance(index, bool)
            and isinstance(index, int)
            and 0 <= index < len(candidates)
        ):
            index_counts[index] = index_counts.get(index, 0) + 1

    for approval in approvals:
        if not isinstance(approval, dict):
            continue
        index = approval.get("index")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= len(candidates)
            or index_counts.get(index) != 1
        ):
            reject(
                "duplicate_approval"
                if isinstance(index, int)
                and not isinstance(index, bool)
                and index_counts.get(index, 0) > 1
                else "invalid_approval"
            )
            continue

        candidate = candidates[index]
        source = str(candidate.get("source", "")).strip()
        target = str(candidate.get("target", "")).strip()
        relation = normalize_relation(candidate.get("relation"))
        if not relation:
            reject("relation_mismatch")
            continue
        normalized_source = _normalized_phrase(source)
        normalized_target = _normalized_phrase(target)

        evidence_id = candidate.get("evidence_id")
        if not isinstance(evidence_id, str) or not evidence_id.strip():
            reject("invalid_evidence_id")
            continue
        resolved_evidence_id = evidence_id.strip()
        span = span_by_id.get(resolved_evidence_id)
        if span is None:
            reject("invalid_evidence_id")
            continue
        span_text = span.get("text")
        if not isinstance(span_text, str) or not span_text.strip():
            reject("invalid_evidence_id")
            continue
        if not _span_is_verbatim_slice(section_text, span):
            reject("span_grounding")
            continue

        if (
            not normalized_source
            or normalized_source == normalized_target
            or not _is_concept_endpoint(source)
            or not _is_concept_endpoint(target)
            or not _is_grounded_name(source, normalized_section)
            or not _is_grounded_name(target, normalized_section)
        ):
            reject("invalid_endpoint")
            continue

        normalized_span = _normalized_phrase(span_text)
        if not _is_grounded_name(source, normalized_span) or not _is_grounded_name(
            target, normalized_span
        ):
            reject("span_grounding")
            continue

        if not _quote_supports_relation(span_text, source, relation, target):
            reject("relation_mismatch")
            continue

        triple_key = (normalized_source, relation, normalized_target)
        edge = {"source": source, "relation": relation, "target": target}
        existing = accepted_by_triple.get(triple_key)
        if existing is not None:
            _existing_edge, existing_id, _existing_span = existing
            # Prefer single-span evidence over adjacent-window (e0+e1) ids.
            if _evidence_id_is_window(existing_id) and not _evidence_id_is_window(
                resolved_evidence_id
            ):
                accepted_by_triple[triple_key] = (edge, resolved_evidence_id, span_text)
            # else keep the existing (single-span wins over a later window)
            continue
        accepted_by_triple[triple_key] = (edge, resolved_evidence_id, span_text)

    accepted: list[dict[str, Any]] = []
    for edge, resolved_evidence_id, span_text in accepted_by_triple.values():
        accepted.append(edge)
        if (
            audit_sink is not None
            and len(audit_sink) < _MAX_RETAINED_EDGE_AUDIT
        ):
            audit_sink.append(
                {
                    "source": edge["source"],
                    "relation": edge["relation"],
                    "target": edge["target"],
                    "locator": locator or "",
                    "evidence_id": resolved_evidence_id,
                    "evidence_preview": span_text[:_MAX_EVIDENCE_PREVIEW_CHARS],
                }
            )
    return accepted


def _verifier_approval_count(candidates: list[dict[str, Any]], approvals: Any) -> int:
    """Count unique valid verifier indexes before the local evidence gate."""
    if not isinstance(approvals, list):
        return 0
    indexes: set[int] = set()
    duplicates: set[int] = set()
    for approval in approvals:
        if not isinstance(approval, dict):
            continue
        index = approval.get("index")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < len(candidates)
        ):
            continue
        if index in indexes:
            duplicates.add(index)
        else:
            indexes.add(index)
    return len(indexes - duplicates)


def _has_direct_whole_part_cue(section_text: str) -> bool:
    """Limit recovery calls to explicit whole-part phrasing, never generic usage."""
    normalized = _normalized_phrase(section_text)
    return bool(_DIRECT_WHOLE_PART_CUE.search(normalized))


def _recovery_candidates(
    section_text: str,
    raw_edges: Any,
    evidence_spans: list[dict] | None = None,
) -> list[dict[str, str]]:
    """Keep grounded PART_OF fallback candidates anchored to a resolved evidence span."""
    if not isinstance(raw_edges, list):
        return []
    if evidence_spans is None:
        evidence_spans = build_evidence_spans(section_text)
    span_by_id = _evidence_span_lookup(evidence_spans)
    normalized_section = _normalized_phrase(section_text)
    candidates: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for raw_edge in raw_edges[:MAX_GRAPH_VERIFIER_CANDIDATES]:
        if not isinstance(raw_edge, dict):
            continue
        source = str(raw_edge.get("source", "")).strip()
        relation = str(raw_edge.get("relation", "")).strip().upper()
        target = str(raw_edge.get("target", "")).strip()
        evidence_id = str(raw_edge.get("evidence_id", "") or "").strip()
        key = (_normalized_phrase(source), relation, _normalized_phrase(target))
        span = span_by_id.get(evidence_id)
        if (
            not source
            or not target
            or not evidence_id
            or span is None
            or relation != "PART_OF"
            or key in seen
            or key[0] == key[2]
            or not _is_grounded_name(source, normalized_section)
            or not _is_grounded_name(target, normalized_section)
        ):
            continue
        span_text = span.get("text")
        if not isinstance(span_text, str) or not span_text.strip():
            continue
        if not _span_is_verbatim_slice(section_text, span):
            continue
        normalized_span = _normalized_phrase(span_text)
        if not _is_grounded_name(source, normalized_span) or not _is_grounded_name(
            target, normalized_span
        ):
            continue
        if not _quote_supports_relation(span_text, source, relation, target):
            continue
        seen.add(key)
        candidates.append(
            {
                "source": source,
                "relation": relation,
                "target": target,
                "evidence_id": evidence_id,
            }
        )
    return candidates


def _append_paper_nodes(existing_nodes: list[str], aot: dict[str, Any]) -> None:
    """Reuse only terms accepted from earlier sections of this same paper."""
    known = {_normalized_phrase(name) for name in existing_nodes}
    for name in collect_anchor_nodes(aot):
        normalized_name = _normalized_phrase(name)
        if normalized_name and normalized_name not in known:
            existing_nodes.append(name)
            known.add(normalized_name)


def _payload_dict(payload: Any, label: str) -> dict[str, Any]:
    """Return a plain response mapping before the plan/graph merge.

    The production LLM service returns ``model_dump()`` mappings, while small
    test doubles may return a Pydantic response model directly.  Keeping this
    conversion at the ingestion boundary lets the two independently validated
    responses share the existing filtering and persistence path.
    """
    if isinstance(payload, dict):
        return payload
    model_dump = getattr(payload, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, dict):
            return dumped
    raise TypeError(f"{label} extraction returned a non-object payload.")


def _merge_plan_and_graph(plan_payload: Any, graph_payload: Any) -> dict[str, Any]:
    """Combine separately validated plan and graph responses into the AOT shape."""
    plan = _payload_dict(plan_payload, "plan")
    graph = _payload_dict(graph_payload, "graph")
    if not isinstance(plan.get("main_entities", []), list) or not isinstance(
        plan.get("learning_roadmap", []), list
    ):
        raise TypeError("plan extraction returned an invalid response shape.")
    # Accept the merged key as a small compatibility convenience for custom
    # providers, while the split LLM boundary normally returns nodes/edges.
    graph = graph.get("knowledge_graph", graph)
    if not isinstance(graph, dict):
        raise TypeError("graph extraction returned an invalid knowledge graph.")
    if not isinstance(graph.get("nodes", []), list) or not isinstance(
        graph.get("edges", []), list
    ):
        raise TypeError("graph extraction returned an invalid response shape.")
    return {
        "main_entities": plan.get("main_entities", []),
        "learning_roadmap": plan.get("learning_roadmap", []),
        "knowledge_graph": {
            "nodes": graph.get("nodes", []),
            "edges": graph.get("edges", []),
        },
    }


def _aot_from_plan_and_local_candidates(
    plan_payload: Any, local_candidates: list[dict]
) -> dict[str, Any]:
    """Build AOT when local span candidates replace graph extraction."""
    plan = _payload_dict(plan_payload, "plan")
    if not isinstance(plan.get("main_entities", []), list) or not isinstance(
        plan.get("learning_roadmap", []), list
    ):
        raise TypeError("plan extraction returned an invalid response shape.")
    names: list[str] = []
    seen: set[str] = set()

    def add_name(value: Any) -> None:
        name = str(value).strip()
        key = _normalized_phrase(name)
        if name and key not in seen:
            names.append(name)
            seen.add(key)

    for value in plan.get("main_entities", []):
        add_name(value)
    for edge in local_candidates:
        if not isinstance(edge, dict):
            continue
        add_name(edge.get("source", ""))
        add_name(edge.get("target", ""))
    return {
        "main_entities": plan.get("main_entities", []),
        "learning_roadmap": plan.get("learning_roadmap", []),
        "knowledge_graph": {
            "nodes": [{"name": name} for name in names],
            "edges": list(local_candidates),
        },
    }


def _report_progress(
    callback: Callable[[dict[str, Any]], None] | None,
    completed: int,
    total: int,
    section: str,
    status: str,
) -> None:
    if callback is not None:
        callback(
            {
                "completed": completed,
                "total": total,
                "section": section,
                "status": status,
            }
        )


def ingest_document(
    file_path: str | Path,
    db: Any,
    llm: Any,
    dag: Any,
    processor: DocumentProcessor | None = None,
    force_reingest: bool = False,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    mineru_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compile each source section into graph, roadmap, parent, and HyDE children."""
    document_processor = processor or DocumentProcessor()
    if mineru_manifest_path is None:
        sections = document_processor.process(file_path)
    else:
        mineru_processor = getattr(document_processor, "process_mineru_markdown", None)
        if not callable(mineru_processor):
            raise ValueError("The selected processor does not support MinerU manifests")
        sections = mineru_processor(file_path, mineru_manifest_path)
    report = dict(getattr(document_processor, "last_report", {}))
    report.setdefault("retained_section_count", len(sections))
    report.setdefault("bibliography_omitted", False)
    if not sections:
        raise ValueError("No retained paper body sections were found; nothing was replaced.")

    # Ponytail: do not leak global graph concepts into a different paper's extraction.
    existing_nodes: list[str] = []
    ingested: list[str] = []
    skipped: list[str] = []
    graph_relationships = {
        "candidates": 0,
        "verifier_approvals": 0,
        "retained": 0,
    }
    local_rejections = {key: 0 for key in _GRAPH_REJECTION_KEYS}
    retained_edge_audit: list[dict[str, Any]] = []
    total = len(sections)
    completed = 0
    source = str(sections[0]["metadata"].get("source", ""))
    previous_sections = (
        db.get_section_exact(source, "") if force_reingest and source else []
    )
    current_parent_ids = {
        make_parent_id(dict(section["metadata"])) for section in sections
    }

    for section in sections:
        full_text = section["page_content"]
        metadata = dict(section["metadata"])
        parent_id = make_parent_id(metadata)
        content_hash = make_content_hash(full_text)
        label = f"{metadata['source']}::{metadata['section']}"

        if db.section_exists(parent_id, content_hash) and not force_reingest:
            skipped.append(label)
            completed += 1
            _report_progress(progress_callback, completed, total, label, "up_to_date")
            continue

        replacement_started = False
        try:
            _report_progress(progress_callback, completed, total, label, "compiling")
            # The split service boundary lets roadmap/plan work stay on the
            # configured text model while graph work is routed independently.
            # Keep the legacy combined call for lightweight providers and test
            # doubles that have not adopted the new optional methods yet.
            plan_extractor = getattr(llm, "extract_section_plan", None)
            graph_extractor = getattr(llm, "extract_section_graph", None)
            use_split_extraction = callable(plan_extractor) and callable(graph_extractor)
            # Generic body-length gate: heading stubs skip graph work entirely.
            thin_section = len(full_text.strip()) < _MIN_GRAPH_SECTION_CHARS
            if thin_section:
                evidence_spans = []
                local_candidates: list[dict] = []
                extra_spans: list[dict] = []
                gate_spans: list[dict] = []
                # Plan (or combined plan+graph stripped) + questions only.
                if callable(plan_extractor):
                    extraction_mode = "thin_plan"
                else:
                    extraction_mode = "thin_combined"
            else:
                # Build spans once per compiled section; reuse for extraction, verify, recovery.
                evidence_spans = build_evidence_spans(full_text)
                local_candidates, extra_spans = propose_local_graph_candidates(
                    full_text, evidence_spans
                )
                gate_spans = list(evidence_spans) + list(extra_spans)
                # Non-thin sections always extract a model graph; local scan is
                # unioned later and does not skip extract_section_graph.
                if use_split_extraction:
                    extraction_mode = "split"
                else:
                    extraction_mode = "combined"
            worker_count = 3 if extraction_mode == "split" else 2
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                if extraction_mode == "thin_plan":
                    plan_future = executor.submit(
                        plan_extractor,
                        full_text,
                        existing_nodes=list(existing_nodes),
                    )
                elif extraction_mode == "split":
                    plan_future = executor.submit(
                        plan_extractor,
                        full_text,
                        existing_nodes=list(existing_nodes),
                    )
                    graph_future = executor.submit(
                        graph_extractor,
                        full_text,
                        existing_nodes=list(existing_nodes),
                        evidence_spans=evidence_spans,
                    )
                else:
                    aot_future = executor.submit(
                        llm.extract_section_plan_and_graph,
                        full_text,
                        existing_nodes=list(existing_nodes),
                    )
                questions_future = executor.submit(
                    llm.generate_hypothetical_questions,
                    full_text,
                    num_questions=5,
                    section_title=metadata["section"],
                )
                if extraction_mode == "thin_plan":
                    aot = _aot_from_plan_and_local_candidates(plan_future.result(), [])
                elif extraction_mode == "split":
                    # Plan and questions are independent of graph; an unusable
                    # graph payload must not abort the rest of this section or
                    # later sections of the same document.
                    plan_payload = plan_future.result()
                    try:
                        aot = _merge_plan_and_graph(
                            plan_payload, graph_future.result()
                        )
                    except (ValidationError, TypeError, RuntimeError):
                        aot = _aot_from_plan_and_local_candidates(plan_payload, [])
                else:
                    aot = aot_future.result()
                    if thin_section:
                        # Combined path still returns a graph; drop it for thin bodies.
                        aot = {
                            "main_entities": aot.get("main_entities", [])
                            if isinstance(aot, dict)
                            else [],
                            "learning_roadmap": aot.get("learning_roadmap", [])
                            if isinstance(aot, dict)
                            else [],
                            "knowledge_graph": {"nodes": [], "edges": []},
                        }
                aot = filter_aot_to_section(aot, full_text)
                graph = aot.get("knowledge_graph", {})
                if not isinstance(graph, dict):
                    graph = {}
                    aot["knowledge_graph"] = graph
                if thin_section:
                    # No local scan, graph extract, or verifier for short bodies.
                    graph["edges"] = []
                    graph["nodes"] = []
                else:
                    # Union local scan + model edges; prefer local single-span
                    # on the same normalized triple; cap at the verifier bound.
                    candidate_edges, local_indexes = _union_graph_candidates(
                        local_candidates, graph.get("edges", [])
                    )
                    graph_relationships["candidates"] += len(candidate_edges)
                    # Local matches already passed the wording table; auto-approve
                    # only those indexes. Verify remaining model candidates and
                    # map approvals back onto the merged list.
                    approvals: list[Any] = [
                        {"index": index} for index in sorted(local_indexes)
                    ]
                    model_index_list = [
                        index
                        for index in range(len(candidate_edges))
                        if index not in local_indexes
                    ]
                    if model_index_list:
                        model_only = [
                            candidate_edges[index] for index in model_index_list
                        ]
                        model_approvals = llm.verify_graph_edges(
                            full_text,
                            model_only,
                            evidence_spans=gate_spans,
                        )
                        if isinstance(model_approvals, list):
                            for approval in model_approvals:
                                if not isinstance(approval, dict):
                                    approvals.append(approval)
                                    continue
                                raw_index = approval.get("index")
                                if (
                                    not isinstance(raw_index, bool)
                                    and isinstance(raw_index, int)
                                    and 0 <= raw_index < len(model_index_list)
                                ):
                                    approvals.append(
                                        {"index": model_index_list[raw_index]}
                                    )
                                else:
                                    approvals.append(approval)
                    graph_relationships["verifier_approvals"] += (
                        _verifier_approval_count(candidate_edges, approvals)
                    )
                    graph["edges"] = _approved_graph_edges(
                        full_text,
                        candidate_edges,
                        approvals,
                        local_rejections,
                        evidence_spans=gate_spans,
                        audit_sink=retained_edge_audit,
                        locator=label,
                    )
                    graph_relationships["retained"] += len(graph["edges"])
                questions = questions_future.result()
            nodes = graph.get("nodes", [])
            edges = graph.get("edges", [])

            replacement_started = True
            db.delete_parent(parent_id)
            dag.remove_source_locator(metadata)

            _append_paper_nodes(existing_nodes, aot)

            dag.save_knowledge_graph(
                nodes=nodes,
                edges=edges,
                source=metadata,
                main_entities=aot.get("main_entities", []),
            )

            roadmap_steps = [
                {**step_data, "seq_id": seq_id}
                for seq_id, step_data in enumerate(aot.get("learning_roadmap", []))
            ]

            parent_metadata = {
                **metadata,
                "content_hash": content_hash,
                "main_entities": aot.get("main_entities", []),
                "anchor_nodes": collect_anchor_nodes(aot),
            }
            compiled_writer = getattr(db, "upsert_compiled_section", None)
            if callable(compiled_writer):
                compiled_writer(
                    roadmap_steps,
                    full_text,
                    metadata,
                    parent_metadata,
                    parent_id,
                    questions,
                    metadata["source"],
                )
            else:
                db.upsert_curriculum_section(
                    roadmap_steps,
                    full_text,
                    metadata,
                    parent_metadata,
                    parent_id,
                )
                db.upsert_questions(questions, parent_id, metadata["source"])
            ingested.append(label)
            completed += 1
            _report_progress(progress_callback, completed, total, label, "compiled")
        except Exception:
            if replacement_started:
                # Ponytail: cross-store replacement is intentionally best-effort, not atomic.
                try:
                    db.delete_parent(parent_id)
                except Exception:
                    pass
                try:
                    dag.remove_source_locator(metadata)
                except Exception:
                    pass
            raise

    if force_reingest:
        for previous_section in previous_sections:
            previous_metadata = dict(previous_section.get("metadata", {}))
            previous_parent_id = str(
                previous_metadata.get("parent_id") or make_parent_id(previous_metadata)
            )
            if (
                previous_metadata.get("source") == source
                and previous_parent_id not in current_parent_ids
            ):
                dag.remove_source_locator(previous_metadata)
                db.delete_parent(previous_parent_id)

    if any(local_rejections.values()):
        graph_relationships["local_rejections"] = local_rejections
    if retained_edge_audit:
        graph_relationships["retained_edge_audit"] = retained_edge_audit[
            :_MAX_RETAINED_EDGE_AUDIT
        ]
    return {
        "ingested": ingested,
        "skipped": skipped,
        "report": report,
        "graph_relationships": graph_relationships,
    }


MINERU_OUTPUT_DIR = Path("data/mineru")


def sibling_complete_mineru_manifest(markdown_path: str | Path) -> Path | None:
    """Return a sibling complete MinerU manifest when the Markdown is that extract."""
    path = Path(markdown_path)
    candidate = path.with_name(f"{path.stem}.manifest.json")
    if not candidate.is_file():
        return None
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("complete") is not True:
        return None
    return candidate


def compile_uploaded_document(
    file_path: str | Path,
    db: Any,
    llm: Any,
    dag: Any,
    *,
    force_reingest: bool = False,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    mineru_client: Any | None = None,
    mineru_output_dir: str | Path = MINERU_OUTPUT_DIR,
    processor: DocumentProcessor | None = None,
) -> dict[str, Any]:
    """Run the app ingest path: PDF through MinerU, then the existing AOT pipeline."""
    path = Path(file_path).expanduser()
    suffix = path.suffix.lower()
    if suffix in {".md", ".markdown"}:
        return ingest_document(
            path,
            db,
            llm,
            dag,
            processor=processor,
            force_reingest=force_reingest,
            progress_callback=progress_callback,
            mineru_manifest_path=sibling_complete_mineru_manifest(path),
        )
    if suffix != ".pdf":
        raise ValueError("Only PDF and Markdown files can be ingested.")

    client = mineru_client
    if client is None:
        from scripts.mineru_flash import MinerUFlashClient

        client = MinerUFlashClient()
    _report_progress(progress_callback, 0, 1, path.name, "extracting")
    output_path = Path(mineru_output_dir).expanduser() / f"{path.stem}.md"
    extraction = client.extract(path, output_path=output_path, batch_size=10, language="en")
    manifest = extraction.get("manifest") if isinstance(extraction, dict) else None
    if not isinstance(manifest, dict) or manifest.get("complete") is not True:
        raise ValueError(
            "MinerU extraction is incomplete; AOT compilation did not start."
        )
    markdown_path = extraction.get("markdown_path") or output_path
    manifest_path = extraction.get("manifest_path")
    if manifest_path is None:
        raise ValueError("MinerU extraction did not write a manifest.")
    _report_progress(progress_callback, 1, 1, path.name, "extracted")
    return ingest_document(
        markdown_path,
        db,
        llm,
        dag,
        processor=processor,
        force_reingest=force_reingest,
        progress_callback=progress_callback,
        mineru_manifest_path=manifest_path,
    )
