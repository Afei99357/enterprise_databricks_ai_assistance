"""Tests for document_chunks row conversion."""

from __future__ import annotations

from datetime import datetime, timezone

from wnv_assistant.documents.chunking import DocumentChunk
from wnv_assistant.documents.ingest import EmbeddedChunk
from wnv_assistant.documents.storage import chunks_to_rows


class TestChunksToRows:
    def test_converts_fields_correctly(self) -> None:
        chunk = DocumentChunk(
            chunk_id="abc123",
            document_name="toolkit.pdf",
            page_number=5,
            chunk_type="body",
            text="Some extracted text.",
            is_template_page=False,
        )
        embedded = EmbeddedChunk(chunk=chunk, embedding=[0.1, 0.2, 0.3])

        rows = chunks_to_rows([embedded])

        assert len(rows) == 1
        row = rows[0]
        assert row["chunk_id"] == "abc123"
        assert row["document_name"] == "toolkit.pdf"
        assert row["page_number"] == 5
        assert row["chunk_type"] == "body"
        assert row["text"] == "Some extracted text."
        assert row["embedding"] == [0.1, 0.2, 0.3]
        assert row["is_template_page"] is False
        assert isinstance(row["ingested_at"], datetime)
        assert row["ingested_at"].tzinfo == timezone.utc

    def test_empty_list_returns_empty_list(self) -> None:
        assert chunks_to_rows([]) == []
