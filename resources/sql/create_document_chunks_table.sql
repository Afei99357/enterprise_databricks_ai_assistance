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
