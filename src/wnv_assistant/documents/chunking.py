"""Chunks extracted page text into DocumentChunk records.

Recursive character splitting (paragraph -> sentence -> raw character)
with a calibrated target size and overlap. See the design doc's §1 step 3
for the full reasoning, including why tables aren't given dedicated
chunking treatment and why the template flag is computed per resulting
chunk rather than passed in from the caller.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from .template_flag import is_template_page

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
    chunk_type: str  # always "body" for now -- see design doc §2
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
    *,
    prev_page_text: str = "",
    next_page_text: str = "",
) -> list[DocumentChunk]:
    """Chunk one page's extracted text.

    A small trailing slice of the previous page and leading slice of the
    next page are folded onto this page's text before chunking -- not
    stored as separate chunks, just context that keeps a chunk near a
    page boundary from being embedded as an isolated, possibly
    mid-sentence fragment. The template flag is computed against each
    resulting chunk's own actual text (which can include folded-in
    neighbor-page context), not the originating page's raw text alone --
    see the module docstring for why.
    """
    prefix = prev_page_text[-_BOUNDARY_CONTEXT_CHARS:] if prev_page_text else ""
    suffix = next_page_text[:_BOUNDARY_CONTEXT_CHARS] if next_page_text else ""
    context_text = "\n\n".join(p for p in [prefix, text, suffix] if p)

    pieces = _split_text_recursive(
        context_text, _TARGET_CHUNK_CHARS, _CHUNK_OVERLAP_CHARS
    )
    chunks: list[DocumentChunk] = []
    for part, piece in enumerate(pieces):
        chunks.append(
            DocumentChunk(
                chunk_id=make_chunk_id(document_name, page_number, "body", part),
                document_name=document_name,
                page_number=page_number,
                chunk_type="body",
                text=piece,
                is_template_page=is_template_page(piece),
            )
        )
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
