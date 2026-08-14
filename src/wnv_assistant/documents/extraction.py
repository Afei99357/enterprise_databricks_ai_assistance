"""Extracts text from PDF pages using pdfplumber."""

from __future__ import annotations

import pdfplumber


def extract_page_text(page: pdfplumber.page.Page) -> str:
    """Extract a page's plain text.

    Tables are not given special treatment -- pdfplumber's extract_text()
    already includes a table's cell content as part of the page's plain
    text (it has no concept of "table" vs "prose", it just reads visible
    text), so a page containing a table needs no additional handling
    here. An earlier version of this function also rendered detected
    tables as a separate markdown block and appended it -- that caused
    every table's content to be stored twice (see the design doc's Open
    Questions for why this was reverted).
    """
    return (page.extract_text() or "").strip()
