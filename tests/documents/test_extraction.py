"""Tests for pdfplumber-based page extraction."""

from __future__ import annotations

import io

import pdfplumber
import pytest

from wnv_assistant.documents.extraction import extract_page_text


def _make_pdf_bytes(text: str) -> bytes:
    """Build a minimal one-page PDF containing exactly `text`, for tests."""
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.drawString(72, 720, text)
    c.save()
    return buf.getvalue()


def _make_pdf_with_table(table_data: list[list[str]], text: str | None = None) -> bytes:
    """Build a PDF with a real drawn table (and optional text) via platypus.

    Args:
        table_data: List of rows, each row is a list of strings (cells).
        text: Optional plain text to appear before the table.

    Returns:
        PDF bytes.
    """
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Table, TableStyle

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, topMargin=36, bottomMargin=36)
    story = []

    if text:
        styles = getSampleStyleSheet()
        story.append(Paragraph(text, styles["Normal"]))

    table = Table(table_data)
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ]
        )
    )
    story.append(table)
    doc.build(story)
    return buf.getvalue()


@pytest.fixture
def simple_page():
    """A one-page PDF with a single line of plain text, opened for the test."""
    pdf_bytes = _make_pdf_bytes("Hello, WNV surveillance.")
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        yield pdf.pages[0]


@pytest.fixture
def table_page():
    """A one-page PDF with a real drawn 2x2 table."""
    table_data = [
        ["Species", "Count"],
        ["WNV", "42"],
    ]
    pdf_bytes = _make_pdf_with_table(table_data)
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        yield pdf.pages[0]


class TestExtractPageText:
    def test_extracts_plain_text(self, simple_page) -> None:
        result = extract_page_text(simple_page)
        assert "Hello, WNV surveillance." in result

    def test_empty_page_returns_empty_string(self) -> None:
        pdf_bytes = _make_pdf_bytes("")
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            result = extract_page_text(pdf.pages[0])
        assert result == ""

    def test_table_cell_content_appears_in_plain_text(self, table_page) -> None:
        """A table's cells are ordinary visible text to extract_text().

        Tables get no special treatment here (no markdown rendering) --
        this asserts only that their content isn't *lost*, which is the
        whole reason the markdown-rendering step was safe to drop.
        """
        result = extract_page_text(table_page)

        assert "Species" in result
        assert "Count" in result
        assert "WNV" in result
        assert "42" in result
