# RAG Document Layer — Design

Date: 2026-07-31
Branch under discussion: `milestone/m6-agent-architecture`

## Context

The assistant currently answers structured-data questions via
`TextToSQLTool` against the Gold Delta table. The `DOCUMENT` and `MIXED`
routes already exist in `agent/routing.py` and `agent/orchestrator.py`, but
both are stubs returning "Document retrieval is not yet available." The
project's stated purpose is demonstrating a pattern for combining structured
surveillance data with unstructured source documents (CDC toolkits, research
papers, internal reports) — this design wires up the missing half.

**Validation before design.** Before committing to an extraction approach, a
spike test compared vision-model OCR quality across five configurations
(Gemma 3 12B, Llama 4 Maverick, Claude Sonnet 4.5 at 150 and 300 DPI, and a
locally-run open-weight model, Qwen3-VL 4B) against the actual text of two
real source documents, diffed line-by-line against ground truth. Findings
that shaped this design:

- Every Databricks-hosted model tested (Gemma, Maverick, Claude) made real
  content errors on a dense table page — dropped rows, cross-row content
  bleeding, garbled sentences — even at 300 DPI.
- Qwen3-VL 4B, run locally, matched source documents essentially exactly
  across all test pages, including correctly distinguishing genuine
  fill-in-the-blank template pages from normal narrative pages (every
  hosted model false-flagged a normal page as a template).
- A GPU is sufficient to serve a 4B vision model, and since ingestion only
  needs to run on a schedule or manual trigger (not as an always-on
  endpoint), GPU cost is not a real constraint at this scale. (Initially
  assumed T4/`GPU_SMALL` would be enough — corrected during implementation
  to A10/`GPU_MEDIUM`, since Databricks documents that as the default tier
  for general inference and uses A10 in their own worked example for a
  *smaller* vision-language model; vision models need more headroom than a
  same-size text model because page-image inputs decode into a lot of
  vision tokens, consuming KV-cache memory beyond just the model weights.)
- Given Qwen3-VL's validated accuracy and the low GPU cost, the design uses
  vision-OCR for every page rather than a text-layer-extraction-first
  approach — simpler, and the earlier concern (vision-OCR is less reliable
  than text extraction) does not hold for this specific model.
- Testing against a second and third real document (an academic paper, a
  slide deck) surfaced two further risks that shaped chunking: tables can
  span more than one page, and some pages carry their real content in
  vector-drawn diagrams (flowcharts, cladograms) that have no meaningful
  text-extraction equivalent — a bag of disconnected labels — even though
  the model can see and describe the diagram directly from the page image.

## 1. Model serving

**Decision.** Deploy Qwen3-VL 4B as a custom, GPU-backed (A10, `GPU_MEDIUM`) Databricks
Model Serving endpoint. The ingestion pipeline calls it by a configurable
endpoint name (`WNV_VISION_ENDPOINT`), the same pattern already used for
`WNV_LLM_ENDPOINT` — swapping models later is a config change, not a code
change.

Standing up this endpoint (packaging the model for Databricks Custom Model
Serving, GPU compute config) is a deployment prerequisite for this pipeline
but is infrastructure work, not pipeline logic — out of scope for the
implementation plan that follows this design. The pipeline is written
against the endpoint's contract (an OpenAI-style chat completions API
accepting image input, as validated in the spike test), not against Qwen3-VL
specifically, so it is not blocked on that deployment being finished first;
it can be developed and unit-tested against a fake/mocked endpoint.

## 2. Ingestion pipeline

**Decision.** A Databricks Job, defined with `schedule.pause_status: PAUSED`
by default so deploying the bundle never causes it to start running on its
own — it can be triggered manually, or the schedule can be explicitly
unpaused later. The job is triggered by an Auto Loader (`cloudFiles`) stream
watching a Unity Catalog Volume (e.g. `/Volumes/.../documents/`) for new
PDFs; Auto Loader's own checkpoint tracks which files have already been
processed, so no custom manifest/tracking logic is needed regardless of how
many documents eventually land in the Volume.

Per new document, per page:

1. **Render** the page to a 300 DPI PNG. (300 DPI is the validated floor —
   150 DPI measurably lost content in the spike test that 300 DPI recovered.)
2. **Extract** — send the image to the Qwen3-VL endpoint with the validated
   extraction prompt, which instructs the model to:
   - transcribe all text as clean markdown, tables as markdown tables,
   - keep a colored callout/sidebar box as a distinct block, not interleaved
     into adjacent body text,
   - flag a genuine fill-in-the-blank template page,
   - flag when a table continues from the previous page or onto the next
     page (`TABLE_CONTINUES_FROM_PREV` / `TABLE_CONTINUES_TO_NEXT`) — this
     signal is consumed later by the retrieval loop (§4), not by ingestion;
     multi-page tables are **not** stitched into a single chunk at ingestion
     time (explicitly deferred — see Open Questions),
   - if the page contains a diagram (flowchart, tree, pathway diagram),
     describe its content and relationships under a distinct `Figure:`
     label, separate from body text — the same structural pattern already
     validated for callout boxes (kept as a distinct block, not interleaved
     into body paragraphs), extended to cover diagrams. This specific
     labeling instruction was not itself part of the spike test (the spike
     confirmed the model *can* accurately describe a diagram when asked
     directly, not that it reliably applies a distinct label unprompted for
     every diagram-bearing page) — it's a low-risk extension of a validated
     pattern, not independently validated. If diagram descriptions bleed
     into body text instead of staying under the `Figure:` label during
     implementation, the chunker falls back to treating the whole page as
     one `body` chunk (no `figure_caption` chunk created) rather than
     guessing at a split.
3. **Template-flag deterministically** — regex the extracted markdown for
   `[INSERT`-style bracket patterns rather than trusting the model's
   self-reported flag. This is cheap and decouples correctness from which
   model happens to be behind the endpoint in the future; Qwen3-VL got this
   right unassisted in testing, but a future model swap should not be able
   to silently regress it.
4. **Chunk** — one chunk per page by default (`chunk_type = body`). If the
   extracted markdown contains a `Figure:`-labeled block (per step 2), that
   block becomes an additional `chunk_type = figure_caption` chunk,
   alongside (not instead of) the page's body-text chunk.
   Paragraph-aware splitting only applies if a page's extracted text
   exceeds a length threshold well beyond what real documents tested so far
   have produced (~4,500 characters was the longest page seen).
5. **Embed** — call a text-embedding endpoint (Databricks Foundation Model
   API, `databricks-gte-large-en`) per chunk.
6. **Store** — write chunks + embeddings + metadata to `document_chunks`
   (§3).

## 3. Storage

**Decision.** One new Delta table, `document_chunks`, alongside the existing
structured Gold table:

| column | type | notes |
|---|---|---|
| `chunk_id` | STRING (PK) | hash of document name + page + chunk type |
| `document_name` | STRING | source filename |
| `page_number` | INT | 1-indexed |
| `chunk_type` | STRING | `body` \| `figure_caption` |
| `text` | STRING | extracted markdown for this chunk |
| `embedding` | ARRAY&lt;FLOAT&gt; | text-embedding vector |
| `is_template_page` | BOOLEAN | from the deterministic regex check |
| `table_continues_to_next` | BOOLEAN | from the model's continuation flag |
| `table_continues_from_prev` | BOOLEAN | from the model's continuation flag |
| `ingested_at` | TIMESTAMP | |

## 4. Retrieval

**Decision.** In-process brute-force cosine similarity over `document_chunks`
to start, implemented behind a small `Retriever` interface so it can be
swapped for Databricks Vector Search later without touching the rest of the
pipeline — not needed at current or near-term scale (a handful of
documents, at most a few hundred chunks), but the seam is cheap to leave
open.

Query flow:

1. **Search** — embed the question, rank all chunks by cosine similarity,
   take the top-k.
2. **Expand** — for each matched chunk, also pull in its same-page sibling
   chunks (e.g. a page's `figure_caption` chunk rides along with its `body`
   chunk, and vice versa) and its immediately adjacent page's chunk(s). This
   is the default fix for content that's split across a page boundary (a
   table's title on page N, its rows on page N+1) — it's automatic and
   doesn't depend on the adjacent chunk having scored well on its own.
3. **Agentic follow-up loop** — capped at 3 iterations. After the initial
   search + expand, the loop can take one of:
   - `lookup_page(n)` — fetch a specific page's chunk(s) directly. Triggered
     when the assembled context contains a `table_continues_to_next` /
     `table_continues_from_prev` flag pointing beyond the already-expanded
     adjacent page (the case from §2 that ingestion deliberately doesn't
     solve — a table spanning more than 2 pages), or when the model
     recognizes an internal cross-reference ("see page 12") that isn't
     physically adjacent to what was already retrieved.
   - `search(refined_query)` — re-run search with a refined query when the
     initial results look insufficient.
   - `answer` — synthesize the final answer from everything gathered so far.
4. Whatever context is assembled feeds the same answer-synthesis pattern
   already used for analytics (`synthesize_answer` in `agent/orchestrator.py`).

Hybrid (semantic + keyword) search, a cross-encoder reranker, and HyDE-style
query rewriting were considered and explicitly deferred — not needed at
current document count, and can be added inside the `Retriever` interface
later without changing its callers.

## 5. Integration

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

## 6. Testing

**Decision, following the project's existing pytest marker conventions**
(`real_data`, `integration`, `eval`):

- Unit tests, no external dependencies: chunking logic, the deterministic
  template-flag regex, parsing a `Figure:`-labeled block out of extracted
  markdown into its own chunk, retrieval's same-page/adjacent-page
  expansion, cosine-similarity ranking.
- `integration`-marked tests: real calls to the Qwen3-VL endpoint and the
  embedding endpoint, gated on `WNV_DATABRICKS_HOST`/`WNV_DATABRICKS_TOKEN`
  being set, matching the existing convention.
- `eval`-marked tests: extend `tests/evals/questions.yml` with `DOCUMENT` and
  `MIXED` questions against the real ingested documents, following the
  existing eval harness (routing accuracy + answer-shape checks, not
  exhaustive answer-correctness grading — same intentional scope decision
  as the analytics eval suite).

## Open questions / explicitly deferred

- **Multi-page table stitching at ingestion time** — a table spanning more
  than 2 pages is not merged into a single chunk. The agentic retrieval
  loop's `lookup_page` action is relied on instead. Revisit if this proves
  insufficient in practice.
- **Vector Search migration** — the `Retriever` interface exists specifically
  so this is a swap, not a rewrite, when/if the corpus grows enough to need
  a managed ANN index.
- **Hybrid search, reranking, HyDE** — deferred; same seam.
