"""Tests for page chunking."""

from __future__ import annotations

from wnv_assistant.documents.chunking import (
    DocumentChunk,
    chunk_page,
    make_chunk_id,
)


class TestMakeChunkId:
    def test_same_inputs_produce_same_id(self) -> None:
        a = make_chunk_id("toolkit.pdf", 5, "body")
        b = make_chunk_id("toolkit.pdf", 5, "body")
        assert a == b

    def test_different_page_produces_different_id(self) -> None:
        a = make_chunk_id("toolkit.pdf", 5, "body")
        b = make_chunk_id("toolkit.pdf", 6, "body")
        assert a != b

    def test_different_part_produces_different_id(self) -> None:
        a = make_chunk_id("toolkit.pdf", 5, "body", part=0)
        b = make_chunk_id("toolkit.pdf", 5, "body", part=1)
        assert a != b


class TestChunkPage:
    def test_short_page_produces_one_chunk(self) -> None:
        chunks = chunk_page("toolkit.pdf", 5, "Some page text.", is_template=False)
        assert len(chunks) == 1
        chunk = chunks[0]
        assert isinstance(chunk, DocumentChunk)
        assert chunk.document_name == "toolkit.pdf"
        assert chunk.page_number == 5
        assert chunk.chunk_type == "body"
        assert chunk.text == "Some page text."
        assert chunk.is_template_page is False

    def test_template_flag_is_carried_through(self) -> None:
        chunks = chunk_page("toolkit.pdf", 1, "[INSERT NAME]", is_template=True)
        assert chunks[0].is_template_page is True

    def test_oversized_page_splits_into_multiple_chunks(self) -> None:
        paragraph = "A" * 800
        text = "\n\n".join([paragraph] * 5)  # 4000+ chars, well over target
        chunks = chunk_page("report.pdf", 2, text, is_template=False)
        assert len(chunks) > 1
        assert all(c.chunk_type == "body" for c in chunks)
        assert all(len(c.text) <= 1200 + 150 for c in chunks)  # target + overlap slack
        assert len({c.chunk_id for c in chunks}) == len(chunks)

    def test_short_page_stays_one_chunk(self) -> None:
        text = "A" * 1000  # under _TARGET_CHUNK_CHARS
        chunks = chunk_page("report.pdf", 3, text, is_template=False)
        assert len(chunks) == 1

    def test_boundary_context_is_folded_into_first_chunk(self) -> None:
        chunks = chunk_page(
            "report.pdf",
            4,
            "This page's own text.",
            is_template=False,
            prev_page_text="Trailing context from the previous page.",
        )
        assert "Trailing context from the previous page." in chunks[0].text
        assert "This page's own text." in chunks[0].text

    def test_boundary_context_is_folded_into_last_chunk(self) -> None:
        chunks = chunk_page(
            "report.pdf",
            4,
            "This page's own text.",
            is_template=False,
            next_page_text="Leading context from the next page.",
        )
        assert "Leading context from the next page." in chunks[-1].text

    def test_markdown_table_splits_by_row_with_repeated_header(self) -> None:
        header = "| County | Cases |\n| --- | --- |"
        # 150 rows -> ~2,800 chars total, safely over the 1,200-char
        # target (60 rows was tried and measured at only ~1,090 chars --
        # not enough margin to actually force a split).
        rows = [f"| County{i} | {i} |" for i in range(150)]
        table_text = header + "\n" + "\n".join(rows)
        chunks = chunk_page("surveillance.pdf", 7, table_text, is_template=False)
        assert len(chunks) > 1
        assert all(c.chunk_type == "table" for c in chunks)
        # Every chunk repeats the header -- self-contained on its own.
        assert all("| County | Cases |" in c.text for c in chunks)

    def test_small_table_stays_one_chunk(self) -> None:
        table_text = "| County | Cases |\n| --- | --- |\n| Cook | 12 |"
        chunks = chunk_page("surveillance.pdf", 8, table_text, is_template=False)
        assert len(chunks) == 1
        assert chunks[0].chunk_type == "table"

    def test_mixed_text_and_table_produces_both_chunk_types(self) -> None:
        text = (
            "Some narrative text before the table.\n\n"
            "| County | Cases |\n| --- | --- |\n| Cook | 12 |\n\n"
            "Some narrative text after the table."
        )
        chunks = chunk_page("mixed.pdf", 9, text, is_template=False)
        types = {c.chunk_type for c in chunks}
        assert types == {"body", "table"}
