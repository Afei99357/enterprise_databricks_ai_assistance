"""SQL executor for running validated queries against Databricks.

Supports both Databricks SQL connector (local) and in-process Spark
(Databricks job/notebook).
"""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class QueryResult:
    """Result of executing a SQL query."""

    columns: list[str]
    rows: list[tuple]
    row_count: int
    query: str


class Executor:
    """Executes validated SQL queries against Databricks."""

    def __init__(
        self,
        *,
        host: str | None = None,
        token: str | None = None,
        warehouse_id: str | None = None,
        catalog: str = "eliao",
        schema: str = "wnv_demo",
    ) -> None:
        self.host = host
        self.token = token
        self.warehouse_id = warehouse_id
        self.catalog = catalog
        self.schema = schema

    def execute(self, sql: str) -> QueryResult:
        """Execute a validated SQL query and return results."""
        if self.host and self.token and self.warehouse_id:
            return self._execute_sql_connector(sql)
        else:
            return self._execute_spark(sql)

    def _execute_sql_connector(self, sql: str) -> QueryResult:
        """Execute via databricks-sql-connector (local testing)."""
        from databricks.sql import connect

        host = self.host.removeprefix("https://")
        conn = connect(
            server_hostname=host,
            http_path=f"/sql/1.0/warehouses/{self.warehouse_id}",
            access_token=self.token,
            catalog=self.catalog,
            schema=self.schema,
        )
        try:
            cur = conn.cursor()
            cur.execute(sql)
            columns = [desc[0] for desc in cur.description] if cur.description else []
            rows = cur.fetchall()
            return QueryResult(
                columns=columns,
                rows=[tuple(r) for r in rows],
                row_count=len(rows),
                query=sql,
            )
        finally:
            conn.close()

    def _execute_spark(self, sql: str) -> QueryResult:
        """Execute via in-process Spark (Databricks job/notebook)."""
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.getOrCreate()
        df = spark.sql(sql)
        rows = [tuple(r) for r in df.collect()]
        columns = list(df.columns)
        return QueryResult(
            columns=columns,
            rows=rows,
            row_count=len(rows),
            query=sql,
        )


def executor_from_env() -> Executor:
    """Create an Executor from environment variables.

    Reads WNV_DATABRICKS_HOST, WNV_DATABRICKS_TOKEN,
    WNV_DATABRICKS_WAREHOUSE_ID, WNV_DATABRICKS_CATALOG,
    WNV_DATABRICKS_SCHEMA.
    """
    import os

    return Executor(
        host=os.environ.get("WNV_DATABRICKS_HOST"),
        token=os.environ.get("WNV_DATABRICKS_TOKEN"),
        warehouse_id=os.environ.get("WNV_DATABRICKS_WAREHOUSE_ID"),
        catalog=os.environ.get("WNV_DATABRICKS_CATALOG", "eliao"),
        schema=os.environ.get("WNV_DATABRICKS_SCHEMA", "wnv_demo"),
    )
