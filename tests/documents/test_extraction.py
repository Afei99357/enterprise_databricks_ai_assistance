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


def _make_pdf_with_table(
    table_data: list[list[str]], text: str | None = None
) -> bytes:
    """Build a PDF with a table (and optional text) using reportlab.platypus.

    Args:
        table_data: List of rows, each row is a list of strings (cells).
        text: Optional plain text to appear before the table.

    Returns:
        PDF bytes.
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
    from reportlab.lib.styles import getSampleStyleSheet

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, topMargin=36, bottomMargin=36)
    story = []

    if text:
        styles = getSampleStyleSheet()
        story.append(Paragraph(text, styles['Normal']))

    table = Table(table_data)
    table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
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
    """A one-page PDF with a simple 2x2 table."""
    table_data = [
        ['Name', 'Value'],
        ['Test', '123'],
    ]
    pdf_bytes = _make_pdf_with_table(table_data)
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        yield pdf.pages[0]


@pytest.fixture
def text_and_table_page():
    """A one-page PDF with text followed by a table."""
    table_data = [
        ['Species', 'Count'],
        ['WNV', '42'],
        ['Other', '8'],
    ]
    pdf_bytes = _make_pdf_with_table(
        table_data,
        text="Surveillance data for Q4:"
    )
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

    def test_extracts_table_as_markdown(self, table_page) -> None:
        """Test that tables are extracted and rendered as markdown."""
        result = extract_page_text(table_page)

        # Verify markdown table structure
        assert "| Name | Value |" in result
        assert "| --- | --- |" in result
        assert "| Test | 123 |" in result

    def test_extracts_table_with_multiple_rows(self, text_and_table_page) -> None:
        """Test table extraction with multiple data rows."""
        result = extract_page_text(text_and_table_page)

        # Verify text is present
        assert "Surveillance data for Q4" in result

        # Verify table structure
        assert "| Species | Count |" in result
        assert "| --- | --- |" in result
        assert "| WNV | 42 |" in result
        assert "| Other | 8 |" in result

    def test_text_and_table_are_separated(self, text_and_table_page) -> None:
        """Test that text and table output are separated by double newline."""
        result = extract_page_text(text_and_table_page)

        # The result should have both text and markdown table
        assert "Surveillance data for Q4" in result
        assert "| Species | Count |" in result

        # Text and markdown table should be separated by double newline
        assert "\n\n" in result

        # Extract parts by splitting on double newline
        parts = result.split("\n\n")
        assert len(parts) >= 2

        # One part contains the intro text, another contains the markdown table
        has_intro_text = any("Surveillance data for Q4" in part for part in parts)
        has_markdown_table = any("| Species | Count |" in part for part in parts)

        assert has_intro_text, "Result should contain the intro text"
        assert has_markdown_table, "Result should contain the markdown table"
