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

# Databricks Git folders are available through the /Workspace filesystem.
# This notebook lives at <repo>/notebooks/ingest_documents.py, so stripping
# the last two path segments gives the repo root. This cell must run first:
# the notebook imports wnv_assistant.documents.*, which only exists once the
# project itself is installed.
from pathlib import Path

notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
repo_workspace_path = notebook_path.rsplit('/', 2)[0]
repo_fs_path = (
    repo_workspace_path
    if repo_workspace_path.startswith('/Workspace/')
    else f'/Workspace{repo_workspace_path}'
)
if not (Path(repo_fs_path) / 'pyproject.toml').is_file():
    raise FileNotFoundError(f'Could not find project root at {repo_fs_path}')
print(f'Repo filesystem path: {repo_fs_path}')

# Install the project as a regular package. Do not mutate sys.path.
%pip install {repo_fs_path} "pdfplumber==0.11.10"
dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "eliao")
dbutils.widgets.text("schema", "wnv_demo")
dbutils.widgets.text("volume_path", "/Volumes/eliao/wnv_demo/documents/")
dbutils.widgets.text(
    "checkpoint_path",
    "/Volumes/eliao/wnv_demo/documents_checkpoints/ingest_documents/",
)
# Embedding-endpoint connection settings. The job supplies these as
# base_parameters (see resources/document_ingestion.yml); the defaults here
# keep an interactive run working. Only the secret's location is passed --
# the token itself is read from the secret scope below, never held in a
# widget, so it stays out of the run UI and the notebook's output.
dbutils.widgets.text("databricks_host", "")
dbutils.widgets.text("secret_scope", "wnv_assistant")
dbutils.widgets.text("secret_key", "token")
dbutils.widgets.text("embedding_endpoint", "databricks-gte-large-en")

catalog = dbutils.widgets.get("catalog").strip()
schema = dbutils.widgets.get("schema").strip()
volume_path = dbutils.widgets.get("volume_path").strip()
checkpoint_path = dbutils.widgets.get("checkpoint_path").strip()
databricks_host = dbutils.widgets.get("databricks_host").strip()
secret_scope = dbutils.widgets.get("secret_scope").strip()
secret_key = dbutils.widgets.get("secret_key").strip()
embedding_endpoint = dbutils.widgets.get("embedding_endpoint").strip()

# An unset host defaults to the workspace this job is already running in,
# which is where the embedding endpoint lives.
if not databricks_host:
    databricks_host = spark.conf.get("spark.databricks.workspaceUrl")

print(f"Catalog:    {catalog}")
print(f"Schema:     {schema}")
print(f"Volume:     {volume_path}")
print(f"Checkpoint: {checkpoint_path}")
print(f"Host:       {databricks_host}")
print(f"Embeddings: {embedding_endpoint}")
print(f"Token from: secret scope '{secret_scope}', key '{secret_key}'")

# COMMAND ----------

from wnv_assistant.documents.embedding import EmbeddingClient
from wnv_assistant.documents.ingest import embed_document_chunks, extract_document_chunks
from wnv_assistant.documents.storage import write_chunks

# Built directly rather than via embed_from_env(): a notebook_task cannot be
# given environment variables by the job definition (the Jobs API has no
# environment_vars field for one), so the settings arrive as widgets and the
# token comes from the secret scope.
embedding_client = EmbeddingClient(
    host=databricks_host,
    token=dbutils.secrets.get(scope=secret_scope, key=secret_key),
    endpoint=embedding_endpoint,
)


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
