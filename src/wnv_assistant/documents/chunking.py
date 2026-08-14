"""Chunks extracted page text into DocumentChunk records.

Recursive character splitting (paragraph -> sentence -> raw character)
with a calibrated target size and overlap, plus a table-aware row-based
strategy for markdown tables. See Task 3's design note in the
implementation plan for why this replaced an earlier
whole-page-as-one-chunk design.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

_TARGET_CHUNK_CHARS = 1200  # ~300 tokens
_CHUNK_OVERLAP_CHARS = 150  # ~12.5%
_BOUNDARY_CONTEXT_CHARS = 300  # adjacent-page slice folded in before chunking

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class DocumentChunk:
    """One chunk of extracted document text, ready to embed and store."""

    chunk_id: str
    document_name: str
    page_number: int
    chunk_type: str  # "body" | "table"
    text: str
    is_template_page: bool


def make_chunk_id(
    document_name: str, page_number: int, chunk_type: str, part: int = 0
) -> str:
    """Hash of document name + page + chunk type (+ part, for split chunks)."""
    raw = f"{document_name}:{page_number}:{chunk_type}:{part}"
    return hashlib.sha256(raw.encode()).hexdigest()


def chunk_page(
    document_name: str,
    page_number: int,
    text: str,
    is_template: bool,
    *,
    prev_page_text: str = "",
    next_page_text: str = "",
) -> list[DocumentChunk]:
    """Chunk one page's extracted text.

    A small trailing slice of the previous page and leading slice of the
    next page are folded onto this page's text before chunking -- not
    stored as separate chunks, just context that keeps a chunk near a
    page boundary from being embedded as an isolated, possibly
    mid-sentence fragment.
    """
    prefix = prev_page_text[-_BOUNDARY_CONTEXT_CHARS:] if prev_page_text else ""
    suffix = next_page_text[:_BOUNDARY_CONTEXT_CHARS] if next_page_text else ""
    context_text = "\n\n".join(p for p in [prefix, text, suffix] if p)

    chunks: list[DocumentChunk] = []
    part = 0
    for chunk_type, content in _split_into_segments(context_text):
        if not content.strip():
            continue
        pieces = (
            _chunk_table(content, _TARGET_CHUNK_CHARS)
            if chunk_type == "table"
            else _split_text_recursive(content, _TARGET_CHUNK_CHARS, _CHUNK_OVERLAP_CHARS)
        )
        for piece in pieces:
            chunks.append(
                DocumentChunk(
                    chunk_id=make_chunk_id(document_name, page_number, chunk_type, part),
                    document_name=document_name,
                    page_number=page_number,
                    chunk_type=chunk_type,
                    text=piece,
                    is_template_page=is_template,
                )
            )
            part += 1
    return chunks


def _split_text_recursive(text: str, target: int, overlap: int) -> list[str]:
    """Greedily pack paragraphs into ~target-sized chunks with overlap.

    Falls back to sentence-level splitting for any single paragraph that
    alone exceeds target, then to raw character splitting as a last
    resort -- the standard recursive character splitting hierarchy.
    """
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    pieces: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= target:
            pieces.append(paragraph)
        else:
            pieces.extend(_split_oversized_paragraph(paragraph, target))

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + 2 + len(piece) > target:
            chunks.append(current)
            tail = current[-overlap:] if len(current) > overlap else current
            current = f"{tail}\n\n{piece}"
        else:
            current = f"{current}\n\n{piece}" if current else piece
    if current:
        chunks.append(current)
    return chunks


def _split_oversized_paragraph(paragraph: str, target: int) -> list[str]:
    """Split a paragraph too big for one chunk: sentences, then characters."""
    sentences = [s for s in _SENTENCE_SPLIT.split(paragraph) if s.strip()]
    if len(sentences) > 1:
        pieces: list[str] = []
        current = ""
        for sentence in sentences:
            if current and len(current) + 1 + len(sentence) > target:
                pieces.append(current)
                current = sentence
            else:
                current = f"{current} {sentence}" if current else sentence
        if current:
            pieces.append(current)
        return pieces
    # A single sentence still too big -- fall back to raw character slices.
    return [paragraph[i : i + target] for i in range(0, len(paragraph), target)]


def _is_separator_row(line: str) -> bool:
    """A markdown table separator like `| --- | --- |` -- only |, -, :, spaces."""
    stripped = line.strip().replace("|", "").replace("-", "").replace(" ", "").replace(":", "")
    return len(stripped) == 0 and "|" in line and "-" in line


def _has_separator(lines: list[str]) -> bool:
    return any(_is_separator_row(line) for line in lines)


def _split_into_segments(text: str) -> list[tuple[str, str]]:
    """Split text into ("body" | "table", content) segments.

    A markdown table is detected by consecutive lines starting with '|'
    that contain a real separator row -- avoids false positives from a
    line that happens to start with '|' but isn't part of a table.
    """
    segments: list[tuple[str, str]] = []
    current_lines: list[str] = []
    pipe_buffer: list[str] = []

    def flush_pipe_buffer() -> None:
        if not pipe_buffer:
            return
        if _has_separator(pipe_buffer):
            if current_lines:
                segments.append(("body", "\n".join(current_lines)))
                current_lines.clear()
            segments.append(("table", "\n".join(pipe_buffer)))
        else:
            current_lines.extend(pipe_buffer)
        pipe_buffer.clear()

    for line in text.split("\n"):
        if line.strip().startswith("|"):
            pipe_buffer.append(line)
        else:
            flush_pipe_buffer()
            current_lines.append(line)
    flush_pipe_buffer()
    if current_lines:
        segments.append(("body", "\n".join(current_lines)))
    return segments


def _chunk_table(table_text: str, target: int) -> list[str]:
    """Split a markdown table by rows, repeating the header in each chunk."""
    lines = [line for line in table_text.strip().split("\n") if line.strip()]
    header_lines: list[str] = []
    data_lines: list[str] = []
    found_separator = False
    for line in lines:
        if not found_separator:
            header_lines.append(line)
            if _is_separator_row(line):
                found_separator = True
        else:
            data_lines.append(line)
    if not found_separator:
        return [table_text.strip()]

    header = "\n".join(header_lines)
    chunks: list[str] = []
    current_rows: list[str] = []
    current_len = len(header)
    for row in data_lines:
        if current_rows and current_len + len(row) + 1 > target:
            chunks.append(header + "\n" + "\n".join(current_rows))
            current_rows = []
            current_len = len(header)
        current_rows.append(row)
        current_len += len(row) + 1
    if current_rows:
        chunks.append(header + "\n" + "\n".join(current_rows))
    return chunks
