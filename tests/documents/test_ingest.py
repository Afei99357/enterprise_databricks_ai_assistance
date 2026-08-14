"""Tests for per-document ingestion orchestration."""

from __future__ import annotations

import io
from unittest.mock import MagicMock

from reportlab.pdfgen import canvas

from wnv_assistant.documents.ingest import (
    EmbeddedChunk,
    embed_document_chunks,
    extract_document_chunks,
)


def _make_two_page_pdf(page1_text: str, page2_text: str) -> bytes:
    """Build a two-page PDF. Newlines in a page's text become drawn lines."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    for i, page_text in enumerate((page1_text, page2_text)):
        if i:
            c.showPage()
        for line_number, line in enumerate(page_text.split("\n")):
            c.drawString(72, 720 - line_number * 14, line)
    c.save()
    return buf.getvalue()


class TestExtractDocumentChunks:
    def test_one_chunk_per_page(self) -> None:
        pdf_bytes = _make_two_page_pdf("Page one text.", "Page two text.")
        chunks = extract_document_chunks("toolkit.pdf", pdf_bytes)

        assert len(chunks) == 2
        assert chunks[0].page_number == 1
        assert chunks[0].document_name == "toolkit.pdf"
        assert "Page one text." in chunks[0].text
        assert chunks[1].page_number == 2
        assert "Page two text." in chunks[1].text

    def test_template_page_is_flagged(self) -> None:
        # Page 2's marker sits past the 300-char boundary-context window, so
        # it is not folded into page 1's chunk. Without that padding page 1's
        # chunk would legitimately contain the marker and be flagged too --
        # the flag now describes each chunk's own text, not its source page.
        filler = "\n".join(
            f"Filler line {i} of ordinary narrative content." for i in range(10)
        )
        pdf_bytes = _make_two_page_pdf(
            "Normal narrative text.", f"{filler}\nCall [INSERT PHONE] for help."
        )
        chunks = extract_document_chunks("toolkit.pdf", pdf_bytes)

        assert chunks[0].is_template_page is False
        assert chunks[1].is_template_page is True

    def test_neighbor_page_marker_flags_the_chunk_that_contains_it(self) -> None:
        """A short template page bleeds its marker into its neighbor's chunk.

        That chunk really does contain the marker, so flagging it True is
        correct -- this is the false negative the per-chunk flag fixes.
        """
        pdf_bytes = _make_two_page_pdf(
            "Normal narrative text.", "Call [INSERT PHONE] for help."
        )
        chunks = extract_document_chunks("toolkit.pdf", pdf_bytes)

        assert "[INSERT PHONE]" in chunks[0].text
        assert chunks[0].is_template_page is True
        assert chunks[1].is_template_page is True


class TestEmbedDocumentChunks:
    def test_attaches_embeddings_in_order(self) -> None:
        pdf_bytes = _make_two_page_pdf("First page.", "Second page.")
        chunks = extract_document_chunks("toolkit.pdf", pdf_bytes)

        mock_client = MagicMock()
        mock_client.embed.return_value = [[0.1, 0.2], [0.3, 0.4]]

        embedded = embed_document_chunks(chunks, mock_client)

        assert len(embedded) == 2
        assert all(isinstance(e, EmbeddedChunk) for e in embedded)
        assert embedded[0].chunk == chunks[0]
        assert embedded[0].embedding == [0.1, 0.2]
        assert embedded[1].embedding == [0.3, 0.4]
        mock_client.embed.assert_called_once_with(
            [chunks[0].text, chunks[1].text]
        )

    def test_empty_chunk_list_skips_embedding_call(self) -> None:
        mock_client = MagicMock()
        result = embed_document_chunks([], mock_client)
        assert result == []
        mock_client.embed.assert_not_called()
