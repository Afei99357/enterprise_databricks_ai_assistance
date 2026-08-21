# WNV Assistant

Databricks Lakeflow pipeline that ingests Illinois county-level non-human
West Nile virus surveillance data from CSV through bronze, silver, and gold
Delta layers on Unity Catalog.

This repo previously also held a document-retrieval/text-to-SQL assistant
built on top of this data. That assistant has been retired in favor of
`wnv-genie-assistant`, a separate, lighter-weight Databricks App that answers
analytics questions through a curated Genie Space instead — it queries the
same gold tables this pipeline builds, but doesn't ingest or transform
anything itself. This repo's scope is now just the pipeline: get source data
reliably into Unity Catalog and let Genie (or anything else with the right
grants) do the querying.

## Local validation

Install the locked development environment and run repository-safe checks:

```bash
uv sync --frozen
uv run pytest
uv run ruff check
uv build
```

The default test run uses synthetic fixtures committed under `tests/fixtures`.
Collected datasets and documents are intentionally excluded from Git. See
[`data/README.md`](data/README.md) for the expected local files and checksums.

To validate locally collected source files:

```bash
WNV_RUN_REAL_DATA_TESTS=1 uv run pytest -m real_data
uv run validate-wnv-data --data-directory data/non_human
```

Databricks connectivity is not required for local validation.

## Deploy (Databricks Asset Bundle)

**Prerequisites:**
- [Databricks CLI](https://docs.databricks.com/dev-tools/cli/install.html) installed and authenticated to your workspace (`databricks auth login`, or a named CLI profile — pass it with `--profile <name>` on every command below if you use one).
- A Unity Catalog Volume with the source CSVs already uploaded, at
  `/Volumes/<catalog>/<schema>/landing/non_human/{bird,horse,mosquito}/`.
  **This path is hardcoded** in `src/wnv_assistant/pipelines/bronze_ingestion.sql`
  — deploying with a different `catalog`/`schema` (below) changes where the
  pipeline and its Delta tables are created, but does **not** change where it
  reads source files from. If you deploy to a non-default catalog/schema,
  either upload the CSVs to the matching `eliao.wnv_demo` volume path too, or
  edit the three hardcoded paths in `bronze_ingestion.sql` to match.

**Validate, deploy, and run:**

```bash
# Check the bundle resolves cleanly against your workspace -- no changes made.
databricks bundle validate

# Create/update the Lakeflow pipeline resource in the `dev` target
# (databricks.yml's only target, and its default).
databricks bundle deploy

# Trigger an actual pipeline run. bundle deploy only creates/updates the
# pipeline's *definition* -- it does not start it.
databricks bundle run bronze_ingestion
```

To deploy against a different catalog/schema than the defaults
(`eliao.wnv_demo`, set in `databricks.yml`):

```bash
databricks bundle deploy --var="catalog=<catalog>" --var="schema=<schema>"
```

The pipeline created is named `[dev] WNV Auto Loader Bronze Ingestion` in
the target workspace and builds three layers as materialized/streaming
tables: `bronze_{bird,horse,mosquito}_source` (source-preserving), silver
transformations, and the analytics-ready `gold_county_month_wnv_weather`
table that downstream consumers (like `wnv-genie-assistant`'s Genie Space)
query.

Any CLI profile and its OAuth credentials are local configuration and are
never stored in this repository.
