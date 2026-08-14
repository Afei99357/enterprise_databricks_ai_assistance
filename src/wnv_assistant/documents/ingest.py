"""Per-document ingestion: extract, flag, chunk, embed.

Spark-independent -- returns plain data. The Auto Loader job wires this
into a stream and writes results to document_chunks (see storage.py).
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import pdfplumber

from .chunking import DocumentChunk, chunk_page
from .embedding import EmbeddingClient
from .extraction import extract_page_text


@dataclass(frozen=True)
class EmbeddedChunk:
    """A DocumentChunk with its embedding vector attached."""

    chunk: DocumentChunk
    embedding: list[float]


def extract_document_chunks(document_name: str, pdf_bytes: bytes) -> list[DocumentChunk]:
    """Extract and chunk every page of a PDF. No embedding call.

    Pages are extracted first, then chunked with access to their
    neighbors' text -- chunk_page folds a small slice of the adjacent
    page's text in as boundary context and computes the template flag
    per resulting chunk (see chunking.py's module docstring for why).
    """
    pages: list[tuple[int, str]] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            pages.append((page_number, extract_page_text(page)))

    chunks: list[DocumentChunk] = []
    for i, (page_number, text) in enumerate(pages):
        prev_text = pages[i - 1][1] if i > 0 else ""
        next_text = pages[i + 1][1] if i < len(pages) - 1 else ""
        chunks.extend(
            chunk_page(
                document_name,
                page_number,
                text,
                prev_page_text=prev_text,
                next_page_text=next_text,
            )
        )
    return chunks


def embed_document_chunks(
    chunks: list[DocumentChunk], embedding_client: EmbeddingClient
) -> list[EmbeddedChunk]:
    """Embed every chunk's text in one batch call, preserving chunk order."""
    if not chunks:
        return []
    vectors = embedding_client.embed([c.text for c in chunks])
    return [
        EmbeddedChunk(chunk=c, embedding=v)
        for c, v in zip(chunks, vectors, strict=True)
    ]
