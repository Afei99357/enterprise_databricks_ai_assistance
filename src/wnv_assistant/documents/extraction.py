"""Extracts text and tables from PDF pages using pdfplumber."""

from __future__ import annotations

import pdfplumber


def extract_page_text(page: pdfplumber.page.Page) -> str:
    """Extract a page's text and any detected tables as one markdown string.

    Tables are rendered as markdown tables and appended after the page's
    plain text -- keeps the output shape uniform (one markdown string per
    page) rather than a separate structured-table representation.
    """
    text = (page.extract_text() or "").strip()
    tables = page.extract_tables()
    table_blocks = [block for t in tables if t and (block := _table_to_markdown(t))]
    parts = [p for p in [text, *table_blocks] if p]
    return "\n\n".join(parts)


def _table_to_markdown(table: list[list[str | None]]) -> str:
    """Render a pdfplumber-extracted table as a markdown table."""
    rows = [[cell or "" for cell in row] for row in table]
    header, *body = rows
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)
