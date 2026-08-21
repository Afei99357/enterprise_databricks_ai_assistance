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

Databricks connectivity is not required for local validation. Supply a local
CLI profile when validating or deploying the bundle:

```bash
databricks bundle validate --profile wnv-demo
databricks bundle deploy --profile wnv-demo
```

The profile and its OAuth credentials are local configuration and are never
stored in this repository.
