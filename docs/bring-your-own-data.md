# Bring Your Own Structured Data

This assistant's analytics tool is schema-driven, not WNV-specific. This
project uses West Nile virus surveillance data because it was on hand for
the demo — the pattern is meant to generalize to any single-table (or
small-number-of-tables) structured dataset on Databricks.

To point this assistant at different data, replace these pieces:

## 1. The semantic layer: `src/wnv_assistant/analytics/schema.py`

This is the only place the text-to-SQL tool learns what tables and columns
exist. Replace `GOLD_TABLE` (a `TableDescription`) with one or more
`TableDescription` entries describing your own table(s):

```python
YOUR_TABLE = TableDescription(
    name="your_table_name",
    description="One row per <grain>. Contains <what>.",
    columns=[
        ColumnDescription("your_column", "TYPE", "Plain-English meaning"),
        ...
    ],
    max_rows=5000,
)

ALLOWED_TABLES: list[TableDescription] = [YOUR_TABLE]
```

Every column needs a plain-English `description` — this text goes directly
into the LLM's system prompt (`format_schema_for_prompt()`), and is what
lets the LLM generate correct SQL without being told your business
semantics ahead of time. If a column's meaning is non-obvious (e.g. "NULL
means not reported, not zero" for this project), write that in the
description — it's the only place that knowledge is captured for the LLM.

## 2. The validator allowlist: `src/wnv_assistant/analytics/validator.py`

This derives its allowlist from `ALLOWED_TABLES`/`ALLOWED_COLUMNS` in
`schema.py`, so it updates automatically once you edit `schema.py` — no
changes needed here unless you want different safety rules (e.g. a
different max row limit policy, or allowing joins across multiple tables).

## 3. The executor's catalog/schema: `src/wnv_assistant/analytics/executor.py`

`executor_from_env()` reads `WNV_DATABRICKS_CATALOG`/`WNV_DATABRICKS_SCHEMA`
(defaulting to `eliao`/`wnv_demo` if unset) to know which Unity Catalog
catalog/schema your table lives in. Set these two environment variables to
point at your own catalog/schema; no code change needed if you're staying
on Databricks Unity Catalog with a single schema.

## 4. The eval questions: `tests/evals/questions.yml`

Replace the WNV-specific questions with representative questions against
your own schema, following the same `question` / `expected_route` /
`expected_sql_keywords` shape. `tests/evals/run_eval.py` itself needs no
changes — the harness is schema-agnostic.

## What stays the same

- `agent/orchestrator.py`, `agent/routing.py` — the routing/orchestration
  policy doesn't know or care what table it's querying.
- `llm/client.py`, `llm/databricks_client.py` — the LLM transport layer.
- `analytics/text_to_sql.py` — the SQL-generation flow (prompt assembly,
  JSON-envelope parsing, validation, execution) is schema-agnostic; only
  the schema text injected into the prompt changes.
- `conversation/` — conversation memory works off `AnalyticsQuerySpec`
  (metric/aggregation/dimensions/filters), which is a generic shape, not
  tied to WNV's specific column names.

## What doesn't generalize yet

- `serving.py`'s `WNVAssistantModel` name and its `_handle_request` /
  `AgentResponse` shape are specific to this project's response contract.
  Renaming the class is cosmetic; the response shape (`answer`, `route`,
  `tool_results`, `warnings`, `insufficient_evidence`) is a reasonable
  starting contract for other structured-data assistants too, but hasn't
  been validated against a second real dataset yet.
- Document/RAG retrieval (the `DOCUMENT`/`MIXED` routes) is not implemented
  in this codebase yet — see the design doc at
  `docs/superpowers/specs/2026-07-29-agent-architecture-improvements-design.md`
  for current status.
