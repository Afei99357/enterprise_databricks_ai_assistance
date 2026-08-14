# RAG Document Layer — Design

Date: 2026-07-31 (revised 2026-08-14)
Branch under discussion: `milestone/m6-agent-architecture`

## Context

The assistant currently answers structured-data questions via
`TextToSQLTool` against the Gold Delta table. The `DOCUMENT` and `MIXED`
routes already exist in `agent/routing.py` and `agent/orchestrator.py`, but
both are stubs returning "Document retrieval is not yet available." The
project's stated purpose is demonstrating a pattern for combining structured
surveillance data with unstructured source documents (CDC toolkits, research
papers, internal reports) — this design wires up the missing half.

**Revision note (2026-08-14).** The original version of this design used
vision-model OCR (Qwen3-VL) for every page, based on a spike test showing it
matched source documents essentially exactly. Actually standing up a custom
GPU-backed Model Serving endpoint for it proved far more costly than
expected (a long real debugging process — MLflow schema/signature quirks,
request-shape surprises specific to Databricks' serving layer, GPU memory
sizing), and the *deployed*, resource-constrained endpoint turned out to be
measurably less accurate than free plain-text extraction: a real
side-by-side check against `pdfplumber.extract_text()` ground truth found
`pdfplumber` 100% correct on a test page where the OCR endpoint had a real
(if subtle) transcription error. Given most real documents have genuine
embedded text layers, this design now uses `pdfplumber` for every page and
drops vision-OCR from this project's scope entirely. The OCR work (a
working, fully-debugged Qwen3-VL serving endpoint; the spike-test harness
and findings; three real test documents) was moved to a separate project,
`/home/eric/Projects/ocr-document-extraction`, to revisit later if a real
need for it shows up (scanned pages, diagram-heavy pages — see Open
Questions).

**Validation before design.** Before committing to an extraction approach, a
spike test compared vision-model OCR quality across five configurations
(Gemma 3 12B, Llama 4 Maverick, Claude Sonnet 4.5 at 150 and 300 DPI, and a
locally-run open-weight model, Qwen3-VL 4B) against the actual text of two
real source documents, diffed line-by-line against ground truth. This
informed the OCR investigation (now spun out — see above) but the concrete
findings are still relevant context:

- Every Databricks-hosted model tested (Gemma, Maverick, Claude) made real
  content errors on a dense table page — dropped rows, cross-row content
  bleeding, garbled sentences — even at 300 DPI.
- Qwen3-VL 4B, run locally at full resolution, matched source documents
  essentially exactly across all test pages.
- Testing against a second and third real document (an academic paper, a
  slide deck) surfaced two further risks that still shape this design: some
  tables can span more than one page, and some pages carry their real
  content in vector-drawn diagrams (flowcharts, cladograms) that
  `pdfplumber` extracts only as a bag of disconnected labels, not the
  diagram's actual structure/relationships — a real, accepted gap (see Open
  Questions).

## 1. Ingestion pipeline

**Decision.** A Databricks Job, defined with `schedule.pause_status: PAUSED`
by default so deploying the bundle never causes it to start running on its
own — it can be triggered manually, or the schedule can be explicitly
unpaused later. The job is triggered by an Auto Loader (`cloudFiles`) stream
watching a Unity Catalog Volume (e.g. `/Volumes/.../documents/`) for new
PDFs; Auto Loader's own checkpoint tracks which files have already been
processed, so no custom manifest/tracking logic is needed regardless of how
many documents eventually land in the Volume.

Per new document, per page:

1. **Extract** — `pdfplumber.extract_text()` for the page's text, and
   `page.extract_tables()` for any detected tables, rendered as markdown
   tables and appended to the page's text rather than left as separate
   structured data — keeps the chunk shape uniform (plain markdown per
   chunk) and avoids a second chunk type for what's still fundamentally the
   same page of content.
2. **Template-flag deterministically** — regex the extracted text for
   `[INSERT`-style bracket patterns to flag genuine fill-in-the-blank
   template pages. Same check as before, now against `pdfplumber` output
   instead of OCR output — the regex itself doesn't change.
3. **Chunk** — one chunk per page (`chunk_type = body`). Paragraph-aware
   splitting only applies if a page's extracted text exceeds a length
   threshold well beyond what real documents tested so far have produced
   (~4,500 characters was the longest page seen in earlier testing).
4. **Embed** — call a text-embedding endpoint (Databricks Foundation Model
   API, `databricks-gte-large-en`) per chunk.
5. **Store** — write chunks + embeddings + metadata to `document_chunks`
   (§2).

A page with little or no extractable text (a scanned image, a diagram-only
page) is not specially detected or handled — whatever `pdfplumber` returns,
even if sparse or empty, is what gets ingested. This is an accepted gap, not
an oversight — see Open Questions.

## 2. Storage

**Decision.** One new Delta table, `document_chunks`, alongside the existing
structured Gold table:

| column | type | notes |
|---|---|---|
| `chunk_id` | STRING (PK) | hash of document name + page + chunk type |
| `document_name` | STRING | source filename |
| `page_number` | INT | 1-indexed |
| `chunk_type` | STRING | `body` (only type for now — see Open Questions) |
| `text` | STRING | extracted markdown for this chunk |
| `embedding` | ARRAY&lt;FLOAT&gt; | text-embedding vector |
| `is_template_page` | BOOLEAN | from the deterministic regex check |
| `ingested_at` | TIMESTAMP | |

`chunk_type` keeps room for a future `figure_caption` (or similar) type if
diagram handling is ever added back — dropped from active use for now, not
removed from the schema shape.

## 3. Retrieval

**Decision.** In-process brute-force cosine similarity over `document_chunks`
to start, implemented behind a small `Retriever` interface so it can be
swapped for Databricks Vector Search later without touching the rest of the
pipeline — not needed at current or near-term scale (a handful of
documents, at most a few hundred chunks), but the seam is cheap to leave
open.

Query flow:

1. **Search** — embed the question, rank all chunks by cosine similarity,
   take the top-k.
2. **Expand** — for each matched chunk, also pull in its immediately
   adjacent page's chunk(s). This is the default fix for content that's
   split across a page boundary (a table's title on page N, its rows on
   page N+1) — it's automatic and doesn't depend on the adjacent chunk
   having scored well on its own.
3. **Agentic follow-up loop** — capped at 3 iterations. After the initial
   search + expand, the loop can take one of:
   - `lookup_page(n)` — fetch a specific page's chunk(s) directly. Triggered
     when the model recognizes an internal cross-reference ("see page 12")
     that isn't physically adjacent to what was already retrieved, or when
     a table visibly looks cut off at a page boundary in the assembled
     context (no deterministic signal for this without OCR — relies on the
     model noticing, same as any other follow-up trigger).
   - `search(refined_query)` — re-run search with a refined query when the
     initial results look insufficient.
   - `answer` — synthesize the final answer from everything gathered so far.
4. Whatever context is assembled feeds the same answer-synthesis pattern
   already used for analytics (`synthesize_answer` in `agent/orchestrator.py`).

Hybrid (semantic + keyword) search, a cross-encoder reranker, and HyDE-style
query rewriting were considered and explicitly deferred — not needed at
current document count, and can be added inside the `Retriever` interface
later without changing its callers.

## 4. Integration

**Decision.** A `DocumentTool`, mirroring the existing `TextToSQLTool`
pattern (`analytics/text_to_sql.py`) rather than inventing a new shape:

```python
@dataclass(frozen=True)
class DocumentResult:
    answer: str
    chunks: list[dict]       # retrieved + expanded chunks used
    citations: list[dict]    # {"document_name": ..., "page_number": ...}
    warnings: list[str]

class DocumentTool:
    def answer(self, question: str, conversation_context: str = "") -> DocumentResult:
        ...
```

`agent/orchestrator.py`'s `_build_response` gets real `DOCUMENT` and `MIXED`
branches in place of the current "not yet available" stubs, following the
same shape as the existing `ANALYTICS` branch: call the tool, build a
`ToolResult` (already has `citations: list[dict]` — unused today, populated
by this work), synthesize the answer, attach warnings only for genuine
failure cases (not "not implemented" placeholders).

## 5. Testing

**Decision, following the project's existing pytest marker conventions**
(`real_data`, `integration`, `eval`):

- Unit tests, no external dependencies: `pdfplumber` extraction against
  real fixture PDFs, the deterministic template-flag regex, retrieval's
  adjacent-page expansion, cosine-similarity ranking.
- `integration`-marked tests: real calls to the embedding endpoint, gated on
  `WNV_DATABRICKS_HOST`/`WNV_DATABRICKS_TOKEN` being set, matching the
  existing convention.
- `eval`-marked tests: extend `tests/evals/questions.yml` with `DOCUMENT` and
  `MIXED` questions against the real ingested documents, following the
  existing eval harness (routing accuracy + answer-shape checks, not
  exhaustive answer-correctness grading — same intentional scope decision
  as the analytics eval suite).

## Open questions / explicitly deferred

- **No vision-OCR in this project.** Pages with no extractable text layer
  (scanned images) or whose real content is a diagram are not specially
  handled — `pdfplumber` returns whatever sparse/empty text it can, and
  that's what gets ingested. A working, debugged vision-OCR endpoint and
  the reasoning for when it would actually be worth using (no text layer at
  all, or a real diagram) live in `/home/eric/Projects/ocr-document-extraction`,
  for a future project if this gap proves to matter in practice.
- **Multi-page table stitching at ingestion time** — a table spanning more
  than 2 pages is not merged into a single chunk, and unlike the earlier
  OCR-based version of this design, there's no deterministic
  continuation-flag signal at all (that came from the model, not from
  `pdfplumber`). The agentic retrieval loop's `lookup_page` action is
  relied on instead, on a best-effort basis. Revisit if this proves
  insufficient in practice.
- **Vector Search migration** — the `Retriever` interface exists specifically
  so this is a swap, not a rewrite, when/if the corpus grows enough to need
  a managed ANN index.
- **Hybrid search, reranking, HyDE** — deferred; same seam.
