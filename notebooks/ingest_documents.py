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
