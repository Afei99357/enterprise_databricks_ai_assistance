"""Writes embedded document chunks to the document_chunks Delta table."""

from __future__ import annotations

from datetime import datetime, timezone

from .ingest import EmbeddedChunk


def chunks_to_rows(chunks: list[EmbeddedChunk]) -> list[dict]:
    """Convert embedded chunks to document_chunks table row dicts."""
    now = datetime.now(timezone.utc)
    return [
        {
            "chunk_id": ec.chunk.chunk_id,
            "document_name": ec.chunk.document_name,
            "page_number": ec.chunk.page_number,
            "chunk_type": ec.chunk.chunk_type,
            "text": ec.chunk.text,
            "embedding": ec.embedding,
            "is_template_page": ec.chunk.is_template_page,
            "ingested_at": now,
        }
        for ec in chunks
    ]


def write_chunks(spark, chunks: list[EmbeddedChunk], catalog: str, schema: str) -> None:
    """Append embedded chunks to `<catalog>.<schema>.document_chunks`.

    Requires a live SparkSession (the job notebook's `spark` global) --
    not exercised by unit tests, same as Executor._execute_spark.
    """
    rows = chunks_to_rows(chunks)
    if not rows:
        return
    df = spark.createDataFrame(rows)
    catalog_escaped = catalog.replace("`", "``")
    schema_escaped = schema.replace("`", "``")
    df.write.mode("append").saveAsTable(
        f"`{catalog_escaped}`.`{schema_escaped}`.document_chunks"
    )
