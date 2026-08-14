"""Tests for page chunking."""

from __future__ import annotations

from wnv_assistant.documents.chunking import (
    DocumentChunk,
    chunk_page,
    make_chunk_id,
)

# target (1200) + overlap (150) + 2 for the "\n\n" that joins the carried-over
# overlap tail to the next piece -- see _split_text_recursive.
_MAX_CHUNK_CHARS = 1200 + 150 + 2


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
        chunks = chunk_page("toolkit.pdf", 5, "Some page text.")
        assert len(chunks) == 1
        chunk = chunks[0]
        assert isinstance(chunk, DocumentChunk)
        assert chunk.document_name == "toolkit.pdf"
        assert chunk.page_number == 5
        assert chunk.chunk_type == "body"
        assert chunk.text == "Some page text."
        assert chunk.is_template_page is False

    def test_oversized_page_splits_into_multiple_chunks(self) -> None:
        paragraph = "A" * 800
        text = "\n\n".join([paragraph] * 5)  # 4000+ chars, well over target
        chunks = chunk_page("report.pdf", 2, text)
        assert len(chunks) > 1
        assert all(c.chunk_type == "body" for c in chunks)
        assert all(len(c.text) <= _MAX_CHUNK_CHARS for c in chunks)
        assert len({c.chunk_id for c in chunks}) == len(chunks)

    def test_unbroken_run_respects_the_size_ceiling(self) -> None:
        """One giant unsplittable "sentence" exercises the widest chunk case.

        Character-slicing yields target-sized pieces, and each packed
        chunk after the first is `overlap` tail + "\\n\\n" + a full
        target-sized piece -- so target + overlap + 2 is the true ceiling.
        """
        chunks = chunk_page("d.pdf", 1, "A" * 3000)
        assert len(chunks) > 1
        assert max(len(c.text) for c in chunks) <= _MAX_CHUNK_CHARS

    def test_short_page_stays_one_chunk(self) -> None:
        text = "A" * 1000  # under _TARGET_CHUNK_CHARS
        chunks = chunk_page("report.pdf", 3, text)
        assert len(chunks) == 1

    def test_boundary_context_is_folded_into_first_chunk(self) -> None:
        chunks = chunk_page(
            "report.pdf",
            4,
            "This page's own text.",
            prev_page_text="Trailing context from the previous page.",
        )
        assert "Trailing context from the previous page." in chunks[0].text
        assert "This page's own text." in chunks[0].text

    def test_boundary_context_is_folded_into_last_chunk(self) -> None:
        chunks = chunk_page(
            "report.pdf",
            4,
            "This page's own text.",
            next_page_text="Leading context from the next page.",
        )
        assert "Leading context from the next page." in chunks[-1].text


class TestTemplateFlag:
    """The flag is computed per resulting chunk, not passed in by the caller."""

    def test_marker_in_the_page_text_flags_the_chunk(self) -> None:
        chunks = chunk_page("toolkit.pdf", 1, "[INSERT NAME]")
        assert chunks[0].is_template_page is True

    def test_plain_text_is_not_flagged(self) -> None:
        chunks = chunk_page("toolkit.pdf", 1, "Ordinary narrative text.")
        assert chunks[0].is_template_page is False

    def test_marker_folded_in_from_the_next_page_flags_the_chunk(self) -> None:
        """The chunk's own text contains the marker, so it must be flagged.

        This page's raw text has no marker -- the old page-level flag
        computed by the caller would have said False, a false negative.
        """
        chunks = chunk_page(
            "toolkit.pdf",
            1,
            "Ordinary narrative text on this page.",
            next_page_text="Call [INSERT PHONE] for help.",
        )
        assert len(chunks) == 1
        assert "[INSERT PHONE]" in chunks[0].text
        assert chunks[0].is_template_page is True

    def test_marker_beyond_the_boundary_window_does_not_flag_the_chunk(self) -> None:
        """A marker past the 300-char fold-in window never reaches the chunk."""
        next_page_text = ("B" * 400) + " Call [INSERT PHONE] for help."
        chunks = chunk_page(
            "toolkit.pdf",
            1,
            "Ordinary narrative text on this page.",
            next_page_text=next_page_text,
        )
        assert all("[INSERT PHONE]" not in c.text for c in chunks)
        assert all(c.is_template_page is False for c in chunks)

    def test_flag_is_computed_per_chunk_not_per_page(self) -> None:
        """An oversized page gets an accurate flag on each of its chunks."""
        plain = "A" * 1100
        marked = "Please call [INSERT PHONE]. " + ("B" * 1100)
        chunks = chunk_page("toolkit.pdf", 1, f"{plain}\n\n{marked}")

        assert len(chunks) > 1
        flags = [c.is_template_page for c in chunks]
        assert any(flags), "the chunk carrying the marker must be flagged"
        assert not all(flags), "chunks without the marker must not be flagged"
        for chunk in chunks:
            assert chunk.is_template_page is ("[INSERT PHONE]" in chunk.text)
