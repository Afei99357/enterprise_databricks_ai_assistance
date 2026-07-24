# Enterprise AI Assistant on Databricks — WNV Demonstration Plan

## 1. Project goal

Build a finished demonstration assistant for West Nile virus (WNV) surveillance in Illinois. The assistant must answer:

- structured questions from county/month surveillance and weather data;
- document questions from authoritative WNV guidance;
- mixed questions that require both data analysis and cited guidance.

The MVP will use Databricks AI/BI Genie for structured questions and a retrieval-augmented generation (RAG) path for documents. A routing agent will decide which path to use.

```text
User question
      |
Routing agent
      |
      +-- STRUCTURED ----> Genie Agent ----> Gold Delta table
      |
      +-- DOCUMENT ------> RAG ------------> Vector index + source documents
      |
      +-- MIXED ---------> Genie + RAG ----> Combined, cited answer
      |
      +-- OUT_OF_SCOPE --> Safe response / clarification
```

Genie—not a custom collection of hard-coded SQL queries—is the primary structured-data interface. Direct LLM-generated SQL is deferred unless Genie cannot meet a demonstrated requirement.

## 2. Current status

| Work item | Status | Evidence/result |
|---|---|---|
| Local Python project and dependency management with `uv` | Complete | `pyproject.toml`, `uv.lock`, package and tests exist |
| Profile and validate the three source CSV files | Complete | Duplicate keys, metadata completeness and conflicts tested |
| Connect to the Databricks workspace | Complete | OAuth profile `wnv-demo` configured |
| Create governed landing and table locations | Complete | `eliao.wnv_demo` schema and `eliao.wnv_demo.landing` volume |
| Upload immutable source files | Complete | Bird, horse and mosquito CSV files uploaded |
| Create deployable Databricks bundle | Complete | `databricks.yml` and pipeline resource deployed |
| Bronze ingestion with Auto Loader | Complete | Three streaming tables with managed checkpoints |
| Silver cleaning and aggregation | Complete | Three materialized views with one row per county/month/activity |
| Gold analytical layer | Complete | Unified county-month WNV/weather materialized view |
| Validate Gold grain and conflicts | Complete | 2,261 unique county-month rows; no key/weather conflicts |
| Configure and test Genie Agent | **Next** | Not started |
| Build document ingestion and RAG | Planned | Source PDF is local; not uploaded or indexed |
| Build routing/orchestration agent | Planned | Design agreed; not implemented |
| MLflow tracing and evaluation | Planned | Not implemented |
| User interface | Planned | Not implemented |
| CI/CD workflow | Planned | Bundle exists; automated workflow not implemented |

## 3. Scope and design decisions

### In scope for the MVP

- Illinois non-human WNV surveillance data from 2002–2022.
- Bird, horse and mosquito activity aggregated by county and month.
- Weather variables already present in the collected CSV files.
- Databricks Lakeflow Spark Declarative Pipelines for Bronze, Silver and Gold.
- AI/BI Genie Agent for natural-language structured analysis.
- RAG over selected authoritative WNV documents with citations.
- A small routing layer with four outcomes: `STRUCTURED`, `DOCUMENT`, `MIXED`, and `OUT_OF_SCOPE`.
- Modular tests, MLflow evaluation, a simple app, and CI/CD.

### Deferred unless needed

- A second weather REST API ingestion pipeline. The existing files already contain weather values and are sufficient for the first demonstration.
- A custom text-to-SQL agent or hard-coded query library.
- Forecasting, causal claims, real-time alerts, and clinical diagnosis.
- Multiple environments/catalogs. The current demonstration stays entirely inside the user-approved `eliao` catalog.

### Safety and interpretation

- The assistant must describe relationships as associations, not causation.
- Missing values must not be treated as zero.
- It must not provide medical diagnoses.
- Public-health guidance answers must cite retrieved documents.
- Structured answers must identify relevant filters, dates, geography, and metrics.

## 4. Data and Databricks resources

### Local source files

```text
data/
├── non_human_data_2002_2017_monthly_aggregated_by_county.csv
├── bird_illinois_2002_2022_monthly_aggregated_by_county.csv
├── horse_illinois_2002_2022_monthly_aggregated_by_county.csv
├── mosquito_illinois_2002_2022_monthly_aggregated_by_county.csv
└── documents/
    └── WNV-Outbreak-Communications-Toolkit-2025_508c.pdf
```

The three activity-specific CSV files are the inputs to the implemented pipeline. The combined CSV can remain as a comparison/reference file and is not required as a fourth pipeline input.

### Current Databricks objects

```text
Catalog: eliao
Schema:  eliao.wnv_demo
Volume:  eliao.wnv_demo.landing

/Volumes/eliao/wnv_demo/landing/non_human/
├── bird/bird_illinois_2002_2022_20260724.csv
├── horse/horse_illinois_2002_2022_20260724.csv
└── mosquito/mosquito_illinois_2002_2022_20260724.csv
```

The dated filenames make each landing event immutable. Future files can be added to the corresponding activity directory. Auto Loader will process only files not already recorded in its managed checkpoint.

## 5. Implemented structured-data pipeline

### Phase 1 — Local foundation and data contracts: complete

- [x] Create a `uv`-managed Python project.
- [x] Lock dependencies in `uv.lock`.
- [x] Add reusable validation code under `src/wnv_assistant/`.
- [x] Add unit and real-data contract tests.
- [x] Run Ruff, pytest and package build checks.

Validated source profile:

| Activity | Raw rows | Unique county-month rows | Duplicate source rows | Incomplete metadata rows | Conflicting duplicates |
|---|---:|---:|---:|---:|---:|
| Bird | 15,127 | 784 | 14,343 | 46 | 0 |
| Horse | 7,260 | 351 | 6,909 | 9 | 0 |
| Mosquito | 25,182 | 1,578 | 23,604 | 95 | 0 |

Duplicate source rows are expected because the source contains repeated observations at a finer grain. Silver explicitly aggregates them to the intended county-month grain.

### Phase 2 — Landing and Bronze ingestion: complete

- [x] Create `eliao.wnv_demo.landing`.
- [x] Upload activity-specific source files.
- [x] Use Auto Loader through streaming `read_files`.
- [x] Maintain managed checkpoints through the Lakeflow pipeline.
- [x] Add ingestion metadata such as source file and ingestion timestamp.
- [x] Normalize the invalid source column name `Avian Phylodiversity` to `Avian_Phylodiversity`.

Bronze tables:

```text
eliao.wnv_demo.bronze_bird_source
eliao.wnv_demo.bronze_horse_source
eliao.wnv_demo.bronze_mosquito_source
```

Bronze preserves source-level rows. It does not silently aggregate, impute, or discard incomplete records.

### Phase 3 — Silver transformations: complete

- [x] Parse and normalize year, month, county, FIPS, coordinates, counts and weather fields.
- [x] Aggregate repeated observations to one row per activity/county/year/month.
- [x] Enforce nonnegative activity counts.
- [x] Detect conflicting metadata among duplicate source records.
- [x] Keep missing values as null.
- [x] Retain the one-month-shifted weather variables supplied by the source.

Silver materialized views:

| View | Rows | Unique business keys |
|---|---:|---:|
| `eliao.wnv_demo.silver_bird` | 784 | 784 |
| `eliao.wnv_demo.silver_horse` | 351 | 351 |
| `eliao.wnv_demo.silver_mosquito` | 1,578 | 1,578 |

### Phase 4 — Gold analytical layer: complete

- [x] Combine the three Silver views with explicit activity columns.
- [x] Produce one row per county/year/month.
- [x] Resolve duplicate rows introduced by overlapping activity sources.
- [x] Validate county FIPS, coordinates and weather consistency.
- [x] Make the result suitable for Genie rather than creating many narrowly defined query tables.

Gold materialized view:

```text
eliao.wnv_demo.gold_county_month_wnv_weather
```

Validated result:

| Check | Result |
|---|---:|
| Rows | 2,261 |
| Unique county/year/month keys | 2,261 |
| Year range | 2002–2022 |
| Counties | 101 |
| FIPS conflicts | 0 |
| Coordinate conflicts | 0 |
| Weather conflicts | 0 |
| Months containing all three activity types | 16 |

Gold should remain the stable structured-data contract for Genie. Additional Gold objects should be created only when an evaluated question cannot be answered reliably or efficiently from this table.

## 6. Next phase — Configure and test the Genie Agent

This is the next work to build.

### 6.1 Create the structured assistant

- [ ] Create a Genie Agent backed by `eliao.wnv_demo.gold_county_month_wnv_weather`.
- [ ] Attach the approved SQL warehouse (`eric's test warehouse`, ID `6197de40f3098d2b`).
- [ ] Add table and column descriptions that explain the county-month grain.
- [ ] Explain that activity counts represent bird, horse, or mosquito surveillance observations.
- [ ] Explain all units and the meaning of each weather field.
- [ ] State that shifted weather fields describe the prior month.
- [ ] State that null means unavailable, not zero.
- [ ] Instruct Genie to avoid causal interpretations and diagnosis.

### 6.2 Add representative example questions

Examples configure and evaluate the semantic behavior; they are not hard-coded query routes.

- Which counties had the largest mosquito WNV counts in July 2012?
- Compare monthly mosquito activity with temperature and precipitation in Cook County.
- Which county-months contain bird, horse, and mosquito activity?
- Show annual mosquito activity totals by county from 2010 through 2015.
- For months with mosquito activity, compare current-month and prior-month weather.

### 6.3 Test Genie as a module

- [ ] Create a small structured-question evaluation set.
- [ ] Check generated SQL for correct table, filters, grouping, ordering and null handling.
- [ ] Compare numeric results with deterministic reference SQL.
- [ ] Test ambiguous county names and missing dates.
- [ ] Test attempts to request unsupported causal or medical conclusions.
- [ ] Record latency and result correctness.

Exit criterion: Genie answers the agreed structured evaluation set accurately enough to become the structured tool used by the router.

## 7. Document ingestion and RAG

This path is independent of Genie and can be tested before the router exists.

### 7.1 Curate and land documents

- [ ] Confirm the authoritative source, publisher, publication date and license for every document.
- [ ] Upload approved documents beneath:

```text
/Volumes/eliao/wnv_demo/landing/documents/
```

- [ ] Begin with `WNV-Outbreak-Communications-Toolkit-2025_508c.pdf`.
- [ ] Add a document manifest containing URI, title, organization, version/date, checksum and ingestion timestamp.

### 7.2 Parse, chunk and index

- [ ] Extract text while preserving page numbers and headings.
- [ ] Create a Delta chunk table in `eliao.wnv_demo`.
- [ ] Include document ID, page, section, chunk text, source URI and checksum in every chunk.
- [ ] Generate embeddings and create a Databricks Vector Search index.
- [ ] Make ingestion idempotent by document checksum/version.

### 7.3 Test RAG as a module

- [ ] Test retrieval separately from answer generation.
- [ ] Build a small question set covering transmission, mosquito behavior, prevention and public-health communication.
- [ ] Require citations that identify document and page/section.
- [ ] Test unsupported questions and retrieval with no relevant passage.
- [ ] Verify that answers do not invent guidance or diagnose a user.

Exit criterion: the RAG module returns grounded answers with valid citations for the document evaluation set.

## 8. Routing and answer orchestration

Build this only after Genie and RAG work independently.

### 8.1 Router contract

The router returns a typed decision:

```text
route: STRUCTURED | DOCUMENT | MIXED | OUT_OF_SCOPE
reason: short explanation
structured_question: optional normalized question for Genie
document_question: optional normalized question for RAG
```

Routing rules:

- `STRUCTURED`: questions asking for counts, rankings, trends, comparisons, locations, dates, or weather relationships in the Delta table.
- `DOCUMENT`: questions asking what WNV is, how it spreads, mosquito information, prevention, or guidance.
- `MIXED`: questions requiring a data result and an explanation/guidance grounded in documents.
- `OUT_OF_SCOPE`: medical diagnosis, unsupported predictions, unavailable data, or unrelated requests.

### 8.2 Tool execution

- [ ] Wrap the Genie Agent behind a structured-tool interface.
- [ ] Wrap RAG behind a document-tool interface.
- [ ] Run both tools for `MIXED` questions, in parallel where supported.
- [ ] Combine results without converting association into causation.
- [ ] Preserve citations from RAG.
- [ ] Report the data period and filters used by Genie.
- [ ] Return a safe limitation statement when the evidence is insufficient.

### 8.3 Test routing as a module

- [ ] Unit-test clear structured and document questions.
- [ ] Test mixed questions.
- [ ] Test ambiguous questions and clarification behavior.
- [ ] Test prompt-injection attempts inside user questions and documents.
- [ ] Test tool failure, timeout and empty-result behavior.

## 9. Evaluation and observability

- [ ] Add MLflow tracing for router decisions, Genie calls, retrieval, generation and final response.
- [ ] Avoid recording secrets or unnecessary personal information.
- [ ] Create versioned evaluation datasets for routing, structured accuracy, retrieval and final-answer quality.
- [ ] Track:
  - routing accuracy;
  - structured result accuracy;
  - retrieval relevance;
  - citation correctness;
  - groundedness;
  - safety-policy adherence;
  - latency and error rate.
- [ ] Define pass thresholds before the final showcase.
- [ ] Run regression evaluation whenever prompts, documents, tables or model endpoints change.

## 10. User interface

- [ ] Build a small Databricks App or Streamlit interface.
- [ ] Show the selected route for demonstration/debug mode.
- [ ] Display structured results as a concise table or chart when helpful.
- [ ] Display document citations with title and page/section.
- [ ] Show data coverage and limitations.
- [ ] Add example questions for structured, document and mixed routes.
- [ ] Add a feedback control for evaluation.

The UI should consume the routing service; it should not contain its own SQL, retrieval, or routing logic.

## 11. Dependency management and CI/CD

### Local development

Use `uv` as the source of truth for Python dependency resolution:

```bash
uv sync --frozen
uv run ruff check .
uv run pytest
uv build
```

- Keep runtime dependencies in `pyproject.toml`.
- Commit `uv.lock`.
- Keep transformations and agent logic in importable modules where practical.
- Keep Databricks resource configuration in the Declarative Automation Bundle.
- Avoid notebook-scoped `%pip` installs as the project dependency strategy.

### Databricks deployment

- Deploy the Lakeflow pipeline, jobs, app, model endpoints and Genie-related resources through the bundle when supported.
- Use bundle variables/targets for environment-specific IDs if a later production environment is added.
- Do not create additional catalogs without explicit approval.
- Store secrets in Databricks secret management or approved identity mechanisms, never in source control.

### CI

- [ ] Install dependencies with `uv sync --frozen`.
- [ ] Run Ruff and pytest.
- [ ] Build the wheel.
- [ ] Validate the Databricks bundle.
- [ ] Run local data-contract tests on fixture data.

### CD

- [ ] Deploy automatically to a development target after CI passes.
- [ ] Run Databricks smoke tests for Bronze, Silver, Gold, Genie and RAG.
- [ ] Require approval before a production deployment if a production target is later introduced.
- [ ] Preserve immutable artifact versions and provide a rollback path.

## 12. Repository structure

```text
.
├── databricks.yml
├── pyproject.toml
├── uv.lock
├── README.md
├── Enterprise_AI_Assistant_Databricks_Project_Plan.md
├── resources/
│   └── pipeline.yml
├── src/
│   └── wnv_assistant/
│       ├── data_validation.py
│       ├── pipelines/
│       │   ├── bronze_ingestion.sql
│       │   ├── silver_transformations.sql
│       │   └── gold_analytics.sql
│       ├── genie/                  # next
│       ├── rag/                    # planned
│       ├── routing/                # planned
│       └── evaluation/             # planned
├── tests/
│   ├── test_data_validation.py
│   └── test_real_data_contract.py
└── data/
    ├── *.csv
    └── documents/
```

New modules and directories should be added only as their phase begins.

## 13. Recommended build order

- [x] 1. Profile and contract-test source data.
- [x] 2. Create the governed landing zone.
- [x] 3. Implement Auto Loader Bronze ingestion.
- [x] 4. Build and validate Silver.
- [x] 5. Build and validate Gold.
- [ ] **6. Configure and evaluate the Genie Agent.**
- [ ] 7. Ingest, index and evaluate the document corpus.
- [ ] 8. Implement and test the router.
- [ ] 9. Add MLflow tracing and end-to-end evaluation.
- [ ] 10. Build the user interface.
- [ ] 11. Add CI/CD and run the final showcase test.

This order keeps each major component independently testable and avoids building orchestration before either answer path is reliable.

## 14. Definition of done

The demonstration is complete when:

- [ ] new source CSV files can be placed in the activity landing folders and processed incrementally;
- [ ] Bronze, Silver and Gold validation checks pass;
- [ ] Genie passes the structured evaluation set;
- [ ] RAG passes the retrieval, groundedness and citation evaluation set;
- [ ] the router correctly handles structured, document, mixed and out-of-scope questions;
- [ ] mixed answers clearly separate measured data from document-based guidance;
- [ ] traces and evaluation results are available in MLflow;
- [ ] the app presents useful answers, filters and citations;
- [ ] CI validates code and deployment configuration;
- [ ] the entire deployment is reproducible from version-controlled configuration without creating resources outside `eliao`.
