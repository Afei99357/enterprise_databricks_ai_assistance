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


def _document_chunks_schema():
    """The document_chunks DataFrame schema, matching the table's DDL exactly.

    Spark's own inference would widen Python `int` to LongType (BIGINT)
    and `float` to DoubleType, but the DDL in
    resources/sql/create_document_chunks_table.sql declares
    `page_number INT` and `embedding ARRAY<FLOAT>` -- narrowing
    mismatches that Delta's schema enforcement rejects on append. Stating
    the schema explicitly is what keeps the write compatible.

    pyspark is imported lazily (same as Executor._execute_spark) because
    it only exists inside a real Databricks/Spark runtime -- a
    module-scope import would break every unit test that imports this
    module.
    """
    from pyspark.sql.types import (
        ArrayType,
        BooleanType,
        FloatType,
        IntegerType,
        StringType,
        StructField,
        StructType,
        TimestampType,
    )

    return StructType(
        [
            StructField("chunk_id", StringType(), nullable=False),
            StructField("document_name", StringType(), nullable=False),
            StructField("page_number", IntegerType(), nullable=False),
            StructField("chunk_type", StringType(), nullable=False),
            StructField("text", StringType(), nullable=False),
            StructField("embedding", ArrayType(FloatType()), nullable=False),
            StructField("is_template_page", BooleanType(), nullable=False),
            StructField("ingested_at", TimestampType(), nullable=False),
        ]
    )


def write_chunks(spark, chunks: list[EmbeddedChunk], catalog: str, schema: str) -> None:
    """Append embedded chunks to `<catalog>.<schema>.document_chunks`.

    Requires a live SparkSession (the job notebook's `spark` global) --
    not exercised by unit tests, same as Executor._execute_spark.
    """
    rows = chunks_to_rows(chunks)
    if not rows:
        return
    # createDataFrame matches a dict row to the schema by field *name*
    # (StructType.toInternal does obj.get(name)), so the field order here
    # need not match chunks_to_rows' key order -- but the names must match
    # exactly, or a missing name silently becomes a null and trips the
    # NOT NULL constraint.
    df = spark.createDataFrame(rows, schema=_document_chunks_schema())
    catalog_escaped = catalog.replace("`", "``")
    schema_escaped = schema.replace("`", "``")
    df.write.mode("append").saveAsTable(
        f"`{catalog_escaped}`.`{schema_escaped}`.document_chunks"
    )
