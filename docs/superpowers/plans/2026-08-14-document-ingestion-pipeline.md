# Document Ingestion Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ingest PDF documents from a Unity Catalog Volume into a searchable
`document_chunks` Delta table — extract text/tables per page with
`pdfplumber`, flag template pages, chunk, embed, and store — with no
vision/OCR step (see `docs/superpowers/specs/2026-07-31-rag-document-layer-design.md`
for why).

**Architecture:** A new `wnv_assistant.documents` package holds all
Spark-independent logic (extraction, template-flagging, chunking, embedding)
as plain, unit-testable functions. A Databricks notebook
(`notebooks/ingest_documents.py`) wires those functions into an Auto
Loader-triggered batch stream that writes to Delta — the only part of this
feature that isn't unit-tested directly, matching how this project's
existing DLT pipelines and Model Serving notebooks are also
infrastructure, not unit-tested code.

**Tech Stack:** `pdfplumber` (PDF text/table extraction), PySpark Structured
Streaming with Auto Loader (`cloudFiles`) in `foreachBatch` batch-trigger
mode, Databricks Foundation Model API (`databricks-gte-large-en`
embeddings), Delta Lake, Databricks Asset Bundles.

## Global Constraints

- Python `>=3.11` (from `pyproject.toml`).
- New runtime dependency `pdfplumber` pinned to `0.11.10` everywhere it's
  referenced (dev deps, job notebook's `%pip install`, job resource YAML's
  `environment` spec) — this project pins exact versions, not minimums, for
  anything that gets installed at runtime (see the vision-endpoint
  notebooks' own commentary on why: an unpinned dependency once silently
  resolved to an incompatible newer release).
- Follow existing module conventions exactly: `from __future__ import
  annotations` at the top of every new module, `@dataclass(frozen=True)`
  for data models, a `*_from_env()` factory function for anything reading
  `WNV_DATABRICKS_*` env vars (matches `executor_from_env`,
  `client_from_env`).
- Test layout mirrors source layout: `tests/documents/` for
  `src/wnv_assistant/documents/`.
- Real-document tests follow the existing `real_data` marker convention
  (gated on `WNV_RUN_REAL_DATA_TESTS=1`, matching
  `tests/data/test_real_contract.py`) — never require Databricks
  connectivity for a `real_data` test.

---

## File Structure

- `pyproject.toml` — add `pdfplumber==0.11.10` to the `dev` dependency group.
- `src/wnv_assistant/documents/__init__.py` — new package, empty.
- `src/wnv_assistant/documents/extraction.py` — pdfplumber-based per-page
  text/table extraction. One function: `extract_page_text`.
- `src/wnv_assistant/documents/template_flag.py` — deterministic
  `[INSERT...]`-style regex check. One function: `is_template_page`.
- `src/wnv_assistant/documents/chunking.py` — `DocumentChunk` dataclass,
  `make_chunk_id`, `chunk_page` (recursive paragraph -> sentence ->
  character splitting at a calibrated target size, table-aware row
  splitting with repeated headers, adjacent-page boundary context folded
  in before chunking — see Task 3's design note for why this replaced an
  earlier whole-page-as-one-chunk approach).
- `src/wnv_assistant/documents/embedding.py` — `EmbeddingClient` (mirrors
  `DatabricksLLMClient`'s shape) and `embed_from_env`.
- `src/wnv_assistant/documents/ingest.py` — `EmbeddedChunk` dataclass,
  `extract_document_chunks` (per-document, Spark-independent), and
  `embed_document_chunks`. Ties extraction + template-flag + chunking +
  embedding together.
- `src/wnv_assistant/documents/storage.py` — `chunks_to_rows` (pure,
  unit-tested) and `write_chunks` (needs a live `SparkSession`, exercised
  only by the job notebook — same pattern as `Executor._execute_spark`).
- `resources/sql/create_document_chunks_table.sql` — bootstrap DDL for
  `document_chunks`, mirroring `resources/sql/create_assistant_tables.sql`.
- `notebooks/ingest_documents.py` — the Databricks Job's notebook task:
  Auto Loader stream + `foreachBatch` wiring.
- `resources/document_ingestion.yml` — the Databricks Job resource,
  `schedule.pause_status: PAUSED` by default.
- `tests/documents/__init__.py`, `tests/documents/test_extraction.py`,
  `tests/documents/test_template_flag.py`, `tests/documents/test_chunking.py`,
  `tests/documents/test_embedding.py`, `tests/documents/test_ingest.py`,
  `tests/documents/test_storage.py`.

---

### Task 1: `pdfplumber` extraction (`extraction.py`)

**Files:**
- Modify: `pyproject.toml` (add `pdfplumber==0.11.10` to `dev` deps)
- Create: `src/wnv_assistant/documents/__init__.py`
- Create: `src/wnv_assistant/documents/extraction.py`
- Test: `tests/documents/__init__.py`, `tests/documents/test_extraction.py`

**Interfaces:**
- Produces: `extract_page_text(page: pdfplumber.page.Page) -> str` — a
  single markdown string combining the page's plain text and any detected
  tables (rendered as markdown tables, appended after the text). Later
  tasks (`ingest.py`) call this per page.

- [ ] **Step 1: Add the dependency**

Edit `pyproject.toml`'s `[dependency-groups]` block:

```toml
[dependency-groups]
dev = [
    "pytest>=8.0,<9",
    "ruff>=0.12,<1",
    "mlflow>=2.20.0",
    "pandas>=1.5.0",
    "pdfplumber==0.11.10",
]
```

Run: `uv sync`
Expected: installs `pdfplumber` and its dependencies into `.venv`.

- [ ] **Step 2: Create the package and write the failing test**

Create `src/wnv_assistant/documents/__init__.py` (empty file).
Create `tests/documents/__init__.py` (empty file).

Create `tests/documents/test_extraction.py`:

```python
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
```

`reportlab` is needed only to *build* test fixtures (not a runtime
dependency of the extraction code itself) — add it alongside `pdfplumber`
in the same `pyproject.toml` edit from Step 1:

```toml
    "pdfplumber==0.11.10",
    "reportlab==4.2.5",
```

Run: `uv sync` again to pick up `reportlab`.

- [ ] **Step 3: Run the test to verify it fails**

Run: `uv run pytest tests/documents/test_extraction.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'wnv_assistant.documents.extraction'`

- [ ] **Step 4: Write the implementation**

Create `src/wnv_assistant/documents/extraction.py`:

```python
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
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest tests/documents/test_extraction.py -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/wnv_assistant/documents/__init__.py \
  src/wnv_assistant/documents/extraction.py \
  tests/documents/__init__.py tests/documents/test_extraction.py
git commit -m "Add pdfplumber page extraction (documents.extraction)"
```

---

### Task 2: Deterministic template-page flag (`template_flag.py`)

**Files:**
- Create: `src/wnv_assistant/documents/template_flag.py`
- Test: `tests/documents/test_template_flag.py`

**Interfaces:**
- Produces: `is_template_page(text: str) -> bool`. Consumed by `ingest.py`
  (Task 5) per page.

- [ ] **Step 1: Write the failing test**

Create `tests/documents/test_template_flag.py`:

```python
"""Tests for deterministic template-page detection."""

from __future__ import annotations

from wnv_assistant.documents.template_flag import is_template_page


class TestIsTemplatePage:
    def test_flags_bracketed_insert_marker(self) -> None:
        text = "Contact us at [INSERT PHONE NUMBER] for more information."
        assert is_template_page(text) is True

    def test_flags_multiple_markers_case_insensitively(self) -> None:
        text = "[insert county name] reported cases in [Insert Month]."
        assert is_template_page(text) is True

    def test_does_not_flag_normal_narrative_text(self) -> None:
        text = "Cook County reported 12 positive mosquito pools in July."
        assert is_template_page(text) is False

    def test_does_not_flag_unrelated_bracket_usage(self) -> None:
        text = "See the appendix [1] for full case counts."
        assert is_template_page(text) is False
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/documents/test_template_flag.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

Create `src/wnv_assistant/documents/template_flag.py`:

```python
"""Deterministic template-page detection.

Regexes the extracted text for `[INSERT ...]`-style bracket placeholders
rather than trusting any upstream source's self-reported flag -- decouples
correctness from whatever produced the text.
"""

from __future__ import annotations

import re

_TEMPLATE_PATTERN = re.compile(r"\[INSERT[^\]]*\]", re.IGNORECASE)


def is_template_page(text: str) -> bool:
    """True if the text contains a genuine fill-in-the-blank template marker."""
    return bool(_TEMPLATE_PATTERN.search(text))
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/documents/test_template_flag.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/wnv_assistant/documents/template_flag.py tests/documents/test_template_flag.py
git commit -m "Add deterministic template-page detection (documents.template_flag)"
```

---

### Task 3: Chunking (`chunking.py`)

**Design (revised from real-world research + comparison against prior
projects — see conversation history dated 2026-08-14):** recursive
character/token splitting is the industry-standard default for RAG
(LangChain's own default splitter), and for research-paper-style
documents specifically, a direct benchmark across 50 real academic papers
found it *beats* semantic (embedding-similarity) chunking outright (69%
vs. 54% accuracy) — so this uses recursive splitting, not semantic
chunking. Concretely:

- **Recursive fallback hierarchy: paragraph → sentence → raw character.**
  Paragraphs (`\n\n`-separated) are packed greedily into
  `_TARGET_CHUNK_CHARS`-sized chunks. A paragraph that alone exceeds the
  target gets split by sentence; a single sentence that still exceeds it
  falls back to raw character slicing. This is the standard "recursive
  character splitting" pattern.
- **Calibrated target size + overlap.** `_TARGET_CHUNK_CHARS = 1200`
  (~300 tokens) and `_CHUNK_OVERLAP_CHARS = 150` (~12.5%), both within
  the commonly-cited 200–500 token / 10–20% overlap range for general
  text — not the whole-page-sized chunks the first draft of this task
  used.
- **Boundary context, not full cross-page merging.** A real risk with
  per-page chunking: a paragraph split by a page break becomes two
  separate, independently-embedded fragments, and a fragment can rank too
  low in similarity search to ever get selected (an isolated half-sentence
  embeds worse than a complete thought). The fix here is a small, fixed
  slice of the *adjacent* page's text (`_BOUNDARY_CONTEXT_CHARS = 300`)
  folded onto the start/end of a page's text *before* chunking — not a
  new chunk, not full-document concatenation, no paragraph-continuation
  detection needed. This keeps chunks still attributable to a single
  `page_number` (no `start_page`/`end_page` schema change needed) while
  meaningfully reducing the isolated-fragment risk.
- **Tables get a separate row-based strategy, not paragraph splitting.**
  Detected via consecutive `|`-prefixed lines with a real separator row.
  Split by row if oversized, with the header + separator row *repeated in
  every resulting chunk* — a table chunk is always self-contained and
  interpretable on its own, never a headerless fragment of rows.
  `chunk_type` is now `"body"` or `"table"` (previously always `"body"`).

**Files:**
- Create: `src/wnv_assistant/documents/chunking.py`
- Test: `tests/documents/test_chunking.py`

**Interfaces:**
- Produces: `DocumentChunk` (frozen dataclass: `chunk_id: str,
  document_name: str, page_number: int, chunk_type: str, text: str,
  is_template_page: bool`), `make_chunk_id(document_name: str, page_number:
  int, chunk_type: str, part: int = 0) -> str`, `chunk_page(document_name:
  str, page_number: int, text: str, is_template: bool, *, prev_page_text:
  str = "", next_page_text: str = "") -> list[DocumentChunk]`. Consumed by
  `ingest.py` (Task 5) and `storage.py` (Task 6).

- [ ] **Step 1: Write the failing test**

Create `tests/documents/test_chunking.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/documents/test_chunking.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

Create `src/wnv_assistant/documents/chunking.py`:

```python
"""Chunks extracted page text into DocumentChunk records.

Recursive character splitting (paragraph -> sentence -> raw character)
with a calibrated target size and overlap, plus a table-aware row-based
strategy for markdown tables. See Task 3's design note in the
implementation plan for why this replaced an earlier
whole-page-as-one-chunk design.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

_TARGET_CHUNK_CHARS = 1200  # ~300 tokens
_CHUNK_OVERLAP_CHARS = 150  # ~12.5%
_BOUNDARY_CONTEXT_CHARS = 300  # adjacent-page slice folded in before chunking

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class DocumentChunk:
    """One chunk of extracted document text, ready to embed and store."""

    chunk_id: str
    document_name: str
    page_number: int
    chunk_type: str  # "body" | "table"
    text: str
    is_template_page: bool


def make_chunk_id(
    document_name: str, page_number: int, chunk_type: str, part: int = 0
) -> str:
    """Hash of document name + page + chunk type (+ part, for split chunks)."""
    raw = f"{document_name}:{page_number}:{chunk_type}:{part}"
    return hashlib.sha256(raw.encode()).hexdigest()


def chunk_page(
    document_name: str,
    page_number: int,
    text: str,
    is_template: bool,
    *,
    prev_page_text: str = "",
    next_page_text: str = "",
) -> list[DocumentChunk]:
    """Chunk one page's extracted text.

    A small trailing slice of the previous page and leading slice of the
    next page are folded onto this page's text before chunking -- not
    stored as separate chunks, just context that keeps a chunk near a
    page boundary from being embedded as an isolated, possibly
    mid-sentence fragment.
    """
    prefix = prev_page_text[-_BOUNDARY_CONTEXT_CHARS:] if prev_page_text else ""
    suffix = next_page_text[:_BOUNDARY_CONTEXT_CHARS] if next_page_text else ""
    context_text = "\n\n".join(p for p in [prefix, text, suffix] if p)

    chunks: list[DocumentChunk] = []
    part = 0
    for chunk_type, content in _split_into_segments(context_text):
        if not content.strip():
            continue
        pieces = (
            _chunk_table(content, _TARGET_CHUNK_CHARS)
            if chunk_type == "table"
            else _split_text_recursive(content, _TARGET_CHUNK_CHARS, _CHUNK_OVERLAP_CHARS)
        )
        for piece in pieces:
            chunks.append(
                DocumentChunk(
                    chunk_id=make_chunk_id(document_name, page_number, chunk_type, part),
                    document_name=document_name,
                    page_number=page_number,
                    chunk_type=chunk_type,
                    text=piece,
                    is_template_page=is_template,
                )
            )
            part += 1
    return chunks


def _split_text_recursive(text: str, target: int, overlap: int) -> list[str]:
    """Greedily pack paragraphs into ~target-sized chunks with overlap.

    Falls back to sentence-level splitting for any single paragraph that
    alone exceeds target, then to raw character splitting as a last
    resort -- the standard recursive character splitting hierarchy.
    """
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    pieces: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= target:
            pieces.append(paragraph)
        else:
            pieces.extend(_split_oversized_paragraph(paragraph, target))

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + 2 + len(piece) > target:
            chunks.append(current)
            tail = current[-overlap:] if len(current) > overlap else current
            current = f"{tail}\n\n{piece}"
        else:
            current = f"{current}\n\n{piece}" if current else piece
    if current:
        chunks.append(current)
    return chunks


def _split_oversized_paragraph(paragraph: str, target: int) -> list[str]:
    """Split a paragraph too big for one chunk: sentences, then characters."""
    sentences = [s for s in _SENTENCE_SPLIT.split(paragraph) if s.strip()]
    if len(sentences) > 1:
        pieces: list[str] = []
        current = ""
        for sentence in sentences:
            if current and len(current) + 1 + len(sentence) > target:
                pieces.append(current)
                current = sentence
            else:
                current = f"{current} {sentence}" if current else sentence
        if current:
            pieces.append(current)
        return pieces
    # A single sentence still too big -- fall back to raw character slices.
    return [paragraph[i : i + target] for i in range(0, len(paragraph), target)]


def _is_separator_row(line: str) -> bool:
    """A markdown table separator like `| --- | --- |` -- only |, -, :, spaces."""
    stripped = line.strip().replace("|", "").replace("-", "").replace(" ", "").replace(":", "")
    return len(stripped) == 0 and "|" in line and "-" in line


def _has_separator(lines: list[str]) -> bool:
    return any(_is_separator_row(line) for line in lines)


def _split_into_segments(text: str) -> list[tuple[str, str]]:
    """Split text into ("body" | "table", content) segments.

    A markdown table is detected by consecutive lines starting with '|'
    that contain a real separator row -- avoids false positives from a
    line that happens to start with '|' but isn't part of a table.
    """
    segments: list[tuple[str, str]] = []
    current_lines: list[str] = []
    pipe_buffer: list[str] = []

    def flush_pipe_buffer() -> None:
        if not pipe_buffer:
            return
        if _has_separator(pipe_buffer):
            if current_lines:
                segments.append(("body", "\n".join(current_lines)))
                current_lines.clear()
            segments.append(("table", "\n".join(pipe_buffer)))
        else:
            current_lines.extend(pipe_buffer)
        pipe_buffer.clear()

    for line in text.split("\n"):
        if line.strip().startswith("|"):
            pipe_buffer.append(line)
        else:
            flush_pipe_buffer()
            current_lines.append(line)
    flush_pipe_buffer()
    if current_lines:
        segments.append(("body", "\n".join(current_lines)))
    return segments


def _chunk_table(table_text: str, target: int) -> list[str]:
    """Split a markdown table by rows, repeating the header in each chunk."""
    lines = [line for line in table_text.strip().split("\n") if line.strip()]
    header_lines: list[str] = []
    data_lines: list[str] = []
    found_separator = False
    for line in lines:
        if not found_separator:
            header_lines.append(line)
            if _is_separator_row(line):
                found_separator = True
        else:
            data_lines.append(line)
    if not found_separator:
        return [table_text.strip()]

    header = "\n".join(header_lines)
    chunks: list[str] = []
    current_rows: list[str] = []
    current_len = len(header)
    for row in data_lines:
        if current_rows and current_len + len(row) + 1 > target:
            chunks.append(header + "\n" + "\n".join(current_rows))
            current_rows = []
            current_len = len(header)
        current_rows.append(row)
        current_len += len(row) + 1
    if current_rows:
        chunks.append(header + "\n" + "\n".join(current_rows))
    return chunks
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/documents/test_chunking.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add src/wnv_assistant/documents/chunking.py tests/documents/test_chunking.py
git commit -m "Add recursive/structural chunking with table-aware row splitting"
```

---

### Task 4: Embedding client (`embedding.py`)

**Files:**
- Create: `src/wnv_assistant/documents/embedding.py`
- Test: `tests/documents/test_embedding.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `EmbeddingClient(host: str, token: str, endpoint: str =
  "databricks-gte-large-en", max_retries: int = 3, retry_delay: float =
  1.0)` with method `embed(texts: list[str]) -> list[list[float]]`, and
  `embed_from_env() -> EmbeddingClient`. Consumed by `ingest.py` (Task 5).

- [ ] **Step 1: Write the failing test**

Create `tests/documents/test_embedding.py`. Note:
`tests/llm/test_client.py` (the closest existing analog, testing
`DatabricksLLMClient`) has no unit-level HTTP mock to mirror — it only has
an `integration`-marked test requiring real Databricks credentials. The
mock below (`unittest.mock.patch` on `urllib.request.urlopen`) is a new,
standalone pattern for this codebase, not a match to an existing one.

```python
"""Tests for the Databricks embeddings client."""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from wnv_assistant.documents.embedding import EmbeddingClient, embed_from_env


class TestEmbeddingClient:
    def test_embed_returns_vectors_in_order(self) -> None:
        client = EmbeddingClient(host="https://test.databricks.com", token="tok")
        response_body = json.dumps(
            {"data": [{"embedding": [0.1, 0.2]}, {"embedding": [0.3, 0.4]}]}
        ).encode()

        mock_resp = MagicMock()
        mock_resp.read.return_value = response_body
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = client.embed(["first text", "second text"])

        assert result == [[0.1, 0.2], [0.3, 0.4]]

    def test_url_strips_scheme_and_trailing_slash(self) -> None:
        client = EmbeddingClient(host="https://test.databricks.com/", token="tok")
        assert client._url == (
            "https://test.databricks.com/serving-endpoints/"
            "databricks-gte-large-en/invocations"
        )


class TestEmbedFromEnv:
    def test_raises_without_host_and_token(self) -> None:
        for key in ("WNV_DATABRICKS_HOST", "WNV_DATABRICKS_TOKEN"):
            os.environ.pop(key, None)
        with pytest.raises(RuntimeError, match="WNV_DATABRICKS_HOST"):
            embed_from_env()

    def test_uses_env_values(self) -> None:
        os.environ["WNV_DATABRICKS_HOST"] = "https://test.databricks.com"
        os.environ["WNV_DATABRICKS_TOKEN"] = "tok"
        try:
            client = embed_from_env()
            assert client.host == "test.databricks.com"
            assert client.endpoint == "databricks-gte-large-en"
        finally:
            os.environ.pop("WNV_DATABRICKS_HOST", None)
            os.environ.pop("WNV_DATABRICKS_TOKEN", None)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/documents/test_embedding.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

Create `src/wnv_assistant/documents/embedding.py`:

```python
"""Databricks Foundation Model API embedding client."""

from __future__ import annotations

import json
import time
from urllib import error, request


class EmbeddingClient:
    """Calls a Databricks Model Serving embeddings endpoint.

    Uses the OpenAI-compatible embeddings format.
    """

    def __init__(
        self,
        *,
        host: str,
        token: str,
        endpoint: str = "databricks-gte-large-en",
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        self.host = host.removeprefix("https://").removesuffix("/")
        self.token = token
        self.endpoint = endpoint
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    @property
    def _url(self) -> str:
        return f"https://{self.host}/serving-endpoints/{self.endpoint}/invocations"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, preserving input order."""
        payload = {"input": texts}
        body = json.dumps(payload).encode()

        for attempt in range(self.max_retries):
            try:
                req = request.Request(
                    self._url, data=body, headers=self._headers(), method="POST"
                )
                with request.urlopen(req, timeout=60) as resp:
                    result = json.loads(resp.read().decode())
                    return [row["embedding"] for row in result["data"]]
            except error.HTTPError as e:
                if e.code in (429, 500, 502, 503):
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay * (2**attempt))
                        continue
                raise RuntimeError(
                    f"Embedding API error {e.code}: {e.read().decode()[:200]}"
                ) from e
            except Exception:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2**attempt))
                    continue
                raise

        return []


def embed_from_env() -> EmbeddingClient:
    """Create an EmbeddingClient from environment variables.

    Reads WNV_DATABRICKS_HOST, WNV_DATABRICKS_TOKEN, WNV_EMBEDDING_ENDPOINT.
    """
    import os

    host = os.environ.get("WNV_DATABRICKS_HOST", "")
    token = os.environ.get("WNV_DATABRICKS_TOKEN", "")
    endpoint = os.environ.get("WNV_EMBEDDING_ENDPOINT", "databricks-gte-large-en")

    if not host or not token:
        raise RuntimeError("Set WNV_DATABRICKS_HOST and WNV_DATABRICKS_TOKEN")

    return EmbeddingClient(host=host, token=token, endpoint=endpoint)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/documents/test_embedding.py -v`
Expected: PASS (4 tests). If the mocking approach doesn't match this
project's actual convention in `tests/llm/test_client.py`, adjust the test
(not the implementation) to match.

- [ ] **Step 5: Commit**

```bash
git add src/wnv_assistant/documents/embedding.py tests/documents/test_embedding.py
git commit -m "Add Databricks embeddings client (documents.embedding)"
```

---

### Task 5: Per-document orchestration (`ingest.py`)

**Files:**
- Create: `src/wnv_assistant/documents/ingest.py`
- Test: `tests/documents/test_ingest.py`

**Interfaces:**
- Consumes: `extract_page_text` (Task 1), `is_template_page` (Task 2),
  `DocumentChunk`/`chunk_page` (Task 3), `EmbeddingClient` (Task 4).
- Produces: `EmbeddedChunk` (frozen dataclass: `chunk: DocumentChunk,
  embedding: list[float]`), `extract_document_chunks(document_name: str,
  pdf_bytes: bytes) -> list[DocumentChunk]`, `embed_document_chunks(chunks:
  list[DocumentChunk], embedding_client: EmbeddingClient) ->
  list[EmbeddedChunk]`. Consumed by `storage.py` (Task 6) and the job
  notebook (Task 7).

- [ ] **Step 1: Write the failing test**

Create `tests/documents/test_ingest.py`:

```python
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
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.drawString(72, 720, page1_text)
    c.showPage()
    c.drawString(72, 720, page2_text)
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
        pdf_bytes = _make_two_page_pdf(
            "Normal narrative text.", "Call [INSERT PHONE] for help."
        )
        chunks = extract_document_chunks("toolkit.pdf", pdf_bytes)

        assert chunks[0].is_template_page is False
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
```

Add `reportlab` usage here too (already a dev dependency from Task 1).

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/documents/test_ingest.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

Create `src/wnv_assistant/documents/ingest.py`:

```python
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
from .template_flag import is_template_page


@dataclass(frozen=True)
class EmbeddedChunk:
    """A DocumentChunk with its embedding vector attached."""

    chunk: DocumentChunk
    embedding: list[float]


def extract_document_chunks(document_name: str, pdf_bytes: bytes) -> list[DocumentChunk]:
    """Extract, flag, and chunk every page of a PDF. No embedding call.

    Pages are extracted and flagged first, then chunked with access to
    their neighbors' text -- chunk_page folds a small slice of the
    adjacent page's text in as boundary context (see chunking.py's design
    note on why). The template flag is still computed from each page's
    own raw text, before any boundary context is folded in.
    """
    pages: list[tuple[int, str, bool]] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = extract_page_text(page)
            pages.append((page_number, text, is_template_page(text)))

    chunks: list[DocumentChunk] = []
    for i, (page_number, text, flagged) in enumerate(pages):
        prev_text = pages[i - 1][1] if i > 0 else ""
        next_text = pages[i + 1][1] if i < len(pages) - 1 else ""
        chunks.extend(
            chunk_page(
                document_name,
                page_number,
                text,
                flagged,
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/documents/test_ingest.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/wnv_assistant/documents/ingest.py tests/documents/test_ingest.py
git commit -m "Add per-document ingestion orchestration (documents.ingest)"
```

---

### Task 6: Storage (`storage.py`) and `document_chunks` bootstrap DDL

**Files:**
- Create: `src/wnv_assistant/documents/storage.py`
- Create: `resources/sql/create_document_chunks_table.sql`
- Test: `tests/documents/test_storage.py`

**Interfaces:**
- Consumes: `EmbeddedChunk` (Task 5).
- Produces: `chunks_to_rows(chunks: list[EmbeddedChunk]) -> list[dict]`
  (pure, unit-tested), `write_chunks(spark, chunks: list[EmbeddedChunk],
  catalog: str, schema: str) -> None` (needs a live `SparkSession`, not
  unit-tested here — same as `Executor._execute_spark`). Consumed by the
  job notebook (Task 7).

- [ ] **Step 1: Write the failing test**

Create `tests/documents/test_storage.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/documents/test_storage.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

Create `src/wnv_assistant/documents/storage.py`:

```python
"""Writes embedded document chunks to the document_chunks Delta table."""

from __future__ import annotations

from datetime import datetime, timezone

from .ingest import EmbeddedChunk


def chunks_to_rows(chunks: list[EmbeddedChunk]) -> list[dict]:
    """Convert embedded chunks to document_chunks table row dicts."""
    now = datetime.now(timezone.utc)
    return [
        {
            "chunk_id": ec.chunk.chunk_id,
            "document_name": ec.chunk.document_name,
            "page_number": ec.chunk.page_number,
            "chunk_type": ec.chunk.chunk_type,
            "text": ec.chunk.text,
            "embedding": ec.embedding,
            "is_template_page": ec.chunk.is_template_page,
            "ingested_at": now,
        }
        for ec in chunks
    ]


def write_chunks(spark, chunks: list[EmbeddedChunk], catalog: str, schema: str) -> None:
    """Append embedded chunks to `<catalog>.<schema>.document_chunks`.

    Requires a live SparkSession (the job notebook's `spark` global) --
    not exercised by unit tests, same as Executor._execute_spark.
    """
    rows = chunks_to_rows(chunks)
    if not rows:
        return
    df = spark.createDataFrame(rows)
    catalog_escaped = catalog.replace("`", "``")
    schema_escaped = schema.replace("`", "``")
    df.write.mode("append").saveAsTable(
        f"`{catalog_escaped}`.`{schema_escaped}`.document_chunks"
    )
```

Create `resources/sql/create_document_chunks_table.sql`:

```sql
-- Bootstrap: Create the document_chunks table for the RAG document layer.
-- Run this once via Databricks SQL or a notebook before running the
-- ingestion job (notebooks/ingest_documents.py). Table is created
-- idempotently (CREATE IF NOT EXISTS).

CREATE TABLE IF NOT EXISTS eliao.wnv_demo.document_chunks (
    chunk_id STRING NOT NULL,
    document_name STRING NOT NULL,
    page_number INT NOT NULL,
    chunk_type STRING NOT NULL,
    text STRING NOT NULL,
    embedding ARRAY<FLOAT> NOT NULL,
    is_template_page BOOLEAN NOT NULL,
    ingested_at TIMESTAMP NOT NULL
)
USING DELTA
CLUSTER BY (document_name, page_number);
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/documents/test_storage.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/wnv_assistant/documents/storage.py \
  resources/sql/create_document_chunks_table.sql \
  tests/documents/test_storage.py
git commit -m "Add document_chunks storage (row conversion + Spark write) and bootstrap DDL"
```

---

### Task 7: Auto Loader ingestion job (notebook + bundle resource)

**Files:**
- Create: `notebooks/ingest_documents.py`
- Create: `resources/document_ingestion.yml`

**Interfaces:**
- Consumes: `embed_from_env` (Task 4), `extract_document_chunks` +
  `embed_document_chunks` (Task 5), `write_chunks` (Task 6).
- Produces: nothing consumed by later tasks in this plan — this is the
  infrastructure entrypoint. Sets `WNV_VISION_ENDPOINT`-equivalent config
  (none needed here, no vision step) and populates `document_chunks` for
  Plan B (retrieval) to read.

This task is infrastructure wiring, not unit-testable code (same as
`resources/pipeline.yml`'s DLT SQL files, or the vision-serving
notebooks) — no TDD steps; write both files directly, verify with `python
-c "import ast; ast.parse(open('notebooks/ingest_documents.py').read())"`
that the notebook's Python is syntactically valid (the `%pip`/`# MAGIC`
lines are Databricks-notebook-only syntax and will not literally execute
outside a Databricks notebook context — this check only confirms the rest
of the cells parse).

- [ ] **Step 1: Write the job notebook**

Create `notebooks/ingest_documents.py`:

```python
# Databricks notebook source
# MAGIC %md
# MAGIC # Ingest documents into document_chunks
# MAGIC
# MAGIC Auto Loader-triggered batch ingestion: reads new PDFs from a Unity
# MAGIC Catalog Volume, extracts + chunks + embeds each page via
# MAGIC `wnv_assistant.documents`, and appends results to `document_chunks`.
# MAGIC Auto Loader's own checkpoint tracks which files have already been
# MAGIC processed, so re-running only picks up genuinely new files. No
# MAGIC vision/OCR step -- see
# MAGIC `docs/superpowers/specs/2026-07-31-rag-document-layer-design.md`
# MAGIC for why this project uses pdfplumber only.
# MAGIC
# MAGIC Run `resources/sql/create_document_chunks_table.sql` once before the
# MAGIC first run of this notebook.

# COMMAND ----------

%pip install "pdfplumber==0.11.10"
dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "eliao")
dbutils.widgets.text("schema", "wnv_demo")
dbutils.widgets.text("volume_path", "/Volumes/eliao/wnv_demo/documents/")
dbutils.widgets.text(
    "checkpoint_path",
    "/Volumes/eliao/wnv_demo/documents_checkpoints/ingest_documents/",
)

catalog = dbutils.widgets.get("catalog").strip()
schema = dbutils.widgets.get("schema").strip()
volume_path = dbutils.widgets.get("volume_path").strip()
checkpoint_path = dbutils.widgets.get("checkpoint_path").strip()

print(f"Catalog:    {catalog}")
print(f"Schema:     {schema}")
print(f"Volume:     {volume_path}")
print(f"Checkpoint: {checkpoint_path}")

# COMMAND ----------

from wnv_assistant.documents.embedding import embed_from_env
from wnv_assistant.documents.ingest import embed_document_chunks, extract_document_chunks
from wnv_assistant.documents.storage import write_chunks

embedding_client = embed_from_env()


def process_batch(batch_df, batch_id: int) -> None:
    """Extract, chunk, embed, and store every new PDF in this micro-batch."""
    for row in batch_df.collect():
        document_name = row["path"].rsplit("/", 1)[-1]
        pdf_bytes = bytes(row["content"])
        chunks = extract_document_chunks(document_name, pdf_bytes)
        embedded = embed_document_chunks(chunks, embedding_client)
        write_chunks(spark, embedded, catalog, schema)
        print(f"Ingested {document_name}: {len(embedded)} chunks")

# COMMAND ----------

stream = (
    spark.readStream.format("cloudFiles")
    .option("cloudFiles.format", "binaryFile")
    .option("pathGlobFilter", "*.pdf")
    .load(volume_path)
)

query = (
    stream.writeStream.foreachBatch(process_batch)
    .option("checkpointLocation", checkpoint_path)
    .trigger(availableNow=True)
    .start()
)
query.awaitTermination()
print("Ingestion run complete.")
```

- [ ] **Step 2: Write the bundle resource**

Create `resources/document_ingestion.yml`:

```yaml
resources:
  jobs:
    ingest_documents:
      name: "[${bundle.target}] WNV Document Ingestion"
      tasks:
        - task_key: ingest
          notebook_task:
            notebook_path: ../notebooks/ingest_documents.py
          environment_key: default
      environments:
        - environment_key: default
          spec:
            client: "1"
            dependencies:
              - pdfplumber==0.11.10
      schedule:
        quartz_cron_expression: "0 0 6 * * ?"
        timezone_id: "America/Chicago"
        pause_status: PAUSED
      tags:
        project: wnv-assistant
        layer: documents
```

`pause_status: PAUSED` matches the design doc's decision — deploying the
bundle never causes this job to start running on its own; trigger it
manually or explicitly unpause the schedule later.

- [ ] **Step 3: Verify the notebook's non-magic cells parse**

Run:

```bash
python3 -c "
import re
src = open('notebooks/ingest_documents.py').read()
# Strip Databricks-only lines (%pip, dbutils magic comments are valid
# Python already; only the leading '# MAGIC' comment lines and the bare
# '%pip install ...' line are notebook-only syntax).
lines = [l for l in src.splitlines() if not l.startswith('%pip')]
compile('\n'.join(lines), 'notebooks/ingest_documents.py', 'exec')
print('OK')
"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add notebooks/ingest_documents.py resources/document_ingestion.yml
git commit -m "Add Auto Loader document ingestion job (notebook + bundle resource)"
```

---

## Self-Review Notes

**Spec coverage** (against `docs/superpowers/specs/2026-07-31-rag-document-layer-design.md`):
- §1 Ingestion pipeline: Auto Loader-triggered, paused-by-default Job — Task 7.
  `pdfplumber.extract_text()` + `extract_tables()` — Task 1. Template-flag
  regex — Task 2. Embedding call — Task 4/5. Store to `document_chunks` —
  Task 6. Sparse/scanned pages not specially handled — no code branch
  exists for this in `extract_document_chunks`, matching the spec's
  explicit decision to accept the gap rather than detect it.
  **Superseded**: the spec's original "one chunk per page, paragraph-split
  above ~4,500 chars" chunking approach was revised (2026-08-14, after
  comparing prior projects and researching current RAG chunking practice)
  to recursive character splitting (paragraph -> sentence -> character)
  at a calibrated ~1,200-char target with overlap, plus table-aware
  row-splitting and adjacent-page boundary context — see Task 3's design
  note. The design doc itself needs a matching update to §1 before this
  plan's chunking approach and the spec fully agree again.
- §2 Storage: `document_chunks` schema — Task 6's DDL matches the spec's
  table exactly. `chunk_type` is now actively `"body"` or `"table"`
  (previously always `"body"`, with `chunk_type` kept only for "future
  extensibility" per the spec's original note) — a real behavior change
  from the spec as originally written, not just an implementation detail.
- §3 Retrieval, §4 Integration, §5 Testing (eval suite): out of scope for
  this plan — covered by the follow-up "Plan B" (retriever, `DocumentTool`,
  orchestrator wiring), which depends on this plan's `document_chunks`
  table existing first.

**Placeholder scan:** none found — every step has real, complete code.

**Type consistency:** `DocumentChunk` (Task 3) is used identically in
Tasks 5, 6; `EmbeddedChunk` (Task 5) is used identically in Task 6 and the
job notebook (Task 7); `EmbeddingClient.embed` signature (Task 4) matches
its call site in `ingest.py` (Task 5); `chunk_page`'s new
`prev_page_text`/`next_page_text` keyword arguments (Task 3) match how
`ingest.py`'s `extract_document_chunks` (Task 5) calls it.

**Follow-up required:** the design doc
(`docs/superpowers/specs/2026-07-31-rag-document-layer-design.md`) §1
still describes the old one-chunk-per-page approach and needs updating to
match this plan before both documents agree.
