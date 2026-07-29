# Agent Architecture Improvements — Design

Date: 2026-07-29
Branch under discussion: `milestone/m5-conversation`

## Context

`plan/PROJECT_PLAN.md` is out of date and no longer treated as the source of
truth for sequencing. This design covers four specific implementation issues
found while reviewing the current branch, discussed and decided one at a time.
The project's actual goal (clarified during this discussion): demonstrate a
reusable pattern — "if another company has structured data and wants to
deploy a similar assistant, how would they do it" — not to build a
WNV-optimized tool. That goal shaped the decisions below, particularly #1.

## 1. Analytics reliability (text-to-SQL)

**Current state.** `TextToSQLTool` (`src/wnv_assistant/analytics/text_to_sql.py`)
generates freeform SQL from natural language via an LLM, validated against an
allowlist (`analytics/validator.py`, `analytics/schema.py`) before execution.
No eval suite exists (`tests/evals/` is an empty directory — this was planned
as milestone M6 in the old plan but never built).

**Decision.** Do **not** replace freeform SQL generation with a fixed
template/parameterized query library. A template library would be schema-
specific and undermine the "bring your own data" story — the current
schema-driven design (`schema.py` feeds table/column descriptions into the
prompt) already generalizes reasonably well to a different table.

Instead:

- Build the eval suite: `tests/evals/questions.yml` (~15-20 representative
  questions against the real Gold schema) + `tests/evals/run_eval.py`
  (loads questions, runs them through the orchestrator, scores routing
  accuracy, SQL correctness, answer correctness, and safety). This is
  itself schema-agnostic methodology — the harness works for any customer's
  schema, only `questions.yml` changes.
- Document the "bring your own schema" story explicitly: what a new
  deployment replaces (`schema.py`'s `TableDescription`/`ColumnDescription`,
  the single-Gold-table assumption, the validator allowlist) to point this
  assistant at different structured data.

## 2. Conversation memory state

**Current state.** `conversation/models.py` already defines
`AnalyticsQuerySpec` — "structured query intent emitted by the LLM alongside
SQL... the primary source for conversation state extraction" — with a
`to_analytics_state()` converter. It is fully unused: `grep` shows it
appears only in its own unit test (`tests/conversation/test_models.py`).
The orchestrator's real state capture, `_extract_state()`
(`agent/orchestrator.py`), instead scrapes `county`/`year` values out of the
*returned result rows*. This fails whenever a query filters on a field it
doesn't also return — e.g. "top 5 counties by mosquito activity in 2022"
returns `county` and a count, never `year`, so the year filter used to
produce the query is lost and a follow-up like "what about 2021?" has
nothing to anchor on.

**Decision.** Wire up the existing `AnalyticsQuerySpec` end-to-end instead
of inventing a new mechanism:

- The LLM emits both the SQL and a structured `AnalyticsQuerySpec`
  (metric, aggregation, dimensions, filters, order_by, limit) in the same
  response when answering an analytics question.
- `orchestrator._extract_state()` is replaced with
  `query_spec.to_analytics_state(result_context)` — state reflects what was
  *asked*, not what happened to come back in the result columns.

## 3. LLM client abstraction + routing layer

**Current state, found while discussing this topic:**

- `llm/client.py` (meant to be the provider-neutral interface) imports
  `_keyword_route` from `llm/databricks_client.py` (the concrete
  Databricks implementation) as a fallback for `classify_route()`. The
  abstract interface has a hard dependency on one concrete provider.
- Routing policy is duplicated and partially dead:
  - `orchestrator.py` defines `ROUTING_SYSTEM_PROMPT` — confirmed via grep
    to be **unused anywhere**.
  - The routing prompt that actually runs is a second, separately written
    copy inside `databricks_client.py`'s `classify_route()`.
  - Two separate keyword-fallback implementations exist
    (`orchestrator._keyword_route` returning `RouteDecision`,
    `databricks_client._keyword_route` returning a plain tuple).
- `Route.MIXED` is defined in the enum (`conversation/models.py`) but is
  **not reachable**: neither classifier prompt ever offers it as an output
  option, and `Orchestrator._execute()` / `_build_response()` have no
  branch that handles it. This blocks the intended design (one orchestrator
  agent deciding analytics vs. document vs. both) from ever being reachable
  once RAG is built.

**Decision.** Restructure so routing/prompt *policy* is owned in one place,
and the LLM client is a thin, swappable transport:

- Shrink `LLMClient` (`llm/client.py`) to a single primitive:
  `chat(messages, *, max_tokens, temperature) -> str`. Remove
  `classify_route`, `generate_sql`, `synthesize_answer` as client interface
  methods.
- Move the routing prompt, the SQL-generation prompt, and the
  answer-synthesis prompt to orchestration-level code (`agent/orchestrator.py`
  and `analytics/text_to_sql.py`), which calls `llm_client.chat(...)`
  directly. Concrete clients (`databricks_client.py`, and any future
  provider client) implement only `chat()`.
- Consolidate to one routing prompt and one keyword-fallback function,
  living with the orchestrator. Delete the dead `ROUTING_SYSTEM_PROMPT`
  duplication and the duplicate `_keyword_route` in `databricks_client.py`.
- Add `MIXED` as a real classifiable category in the consolidated routing
  prompt/enum, even though the dual-tool execution path
  (`_execute()`/`_build_response()` handling `MIXED` by calling both tools)
  is only meaningfully exercised once RAG exists. The classification layer
  should not need another rework when that happens.

**Sequencing note.** This topic and #2 both touch `text_to_sql.py` — the
client shrink (this topic) happens first, then `text_to_sql.py` is updated
once to both call the new `chat()`-only client and emit the structured
`AnalyticsQuerySpec` (topic #2) in the same pass, rather than rewriting the
file twice.

## 4. Bundle-managed model serving

**Current state.** `resources/serving.yml` was deleted in commit `95ee7ad`
("remove unsupported serving.yml") rather than fixed. The original file
(recovered via `git show 5870f06:resources/serving.yml`) mixed three
incompatible Databricks Model Serving concepts in one config:
`chat_model_configuration` (pay-per-token foundation model serving),
`external_model` (proxying to a model hosted outside Databricks), and what
was actually needed — serving a Unity-Catalog-registered custom pyfunc
model, which uses a different block (`served_entities` with
`entity_name`/`entity_version`). Because of this, model deployment today is
entirely manual: `notebooks/register_model.ipynb` registers a model
version, and `notebooks/test_registered_model.ipynb` calls
`mlflow.pyfunc.load_model(...).predict(...)` directly in a notebook. No live
Model Serving REST endpoint has actually been stood up or tested — M4's own
exit criterion ("POST to serving endpoint returns a valid AgentResponse")
is not yet met.

**Decision.** Rewrite `resources/serving.yml` with the correct schema for a
Unity Catalog custom model:

```yaml
resources:
  serving_endpoints:
    wnv_assistant:
      name: "wnv-assistant-${bundle.target}"
      config:
        served_entities:
          - entity_name: "${var.model_name}"
            entity_version: "${var.model_version}"
            workload_size: "Small"
            scale_to_zero_enabled: true
            environment_vars:
              WNV_DATABRICKS_WAREHOUSE_ID: "${var.warehouse_id}"
              WNV_LLM_ENDPOINT: "${var.llm_endpoint}"
              WNV_DATABRICKS_TOKEN: "{{secrets/${var.secret_scope}/token}}"

variables:
  model_name:
    default: "eliao.wnv_demo.wnv_assistant"
  model_version:
    default: "1"
  warehouse_id:
    default: ""
  llm_endpoint:
    default: "databricks-gpt-oss-20b"
  secret_scope:
    default: "wnv_assistant"
```

`WNV_DATABRICKS_HOST` is supplied automatically by Model Serving at runtime
and does not need to be set here. The token must come from a Databricks
secret scope (`secret_scope` variable), never a literal value, consistent
with how the test notebook already avoids storing tokens in model
artifacts.

## Out of scope for this design

- RAG/document retrieval implementation (still a separate go/no-go decision,
  not reopened here).
- Rewriting `plan/PROJECT_PLAN.md` itself (acknowledged as outdated; not
  part of this design).
