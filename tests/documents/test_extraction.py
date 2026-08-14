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


@pytest.fixture
def simple_page():
    """A one-page PDF with a single line of plain text, opened for the test."""
    pdf_bytes = _make_pdf_bytes("Hello, WNV surveillance.")
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
