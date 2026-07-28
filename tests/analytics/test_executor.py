"""Tests for SQL executor."""

from __future__ import annotations

import pytest

from wnv_assistant.analytics.executor import Executor, executor_from_env


class TestExecutorFromEnv:
    """Test executor creation from environment variables."""

    def test_returns_executor_with_defaults(self) -> None:
        """Uses default catalog/schema when not set."""
        import os

        for key in ("WNV_DATABRICKS_HOST", "WNV_DATABRICKS_TOKEN",
                     "WNV_DATABRICKS_WAREHOUSE_ID", "WNV_DATABRICKS_CATALOG",
                     "WNV_DATABRICKS_SCHEMA"):
            os.environ.pop(key, None)

        exe = executor_from_env()
        assert exe.catalog == "eliao"
        assert exe.schema == "wnv_demo"

    def test_uses_env_values(self) -> None:
        """Reads values from environment."""
        import os

        os.environ["WNV_DATABRICKS_HOST"] = "https://test.databricks.com"
        os.environ["WNV_DATABRICKS_CATALOG"] = "my_catalog"
        os.environ["WNV_DATABRICKS_SCHEMA"] = "my_schema"

        try:
            exe = executor_from_env()
            assert exe.host == "https://test.databricks.com"
            assert exe.catalog == "my_catalog"
            assert exe.schema == "my_schema"
        finally:
            os.environ.pop("WNV_DATABRICKS_HOST", None)
            os.environ.pop("WNV_DATABRICKS_CATALOG", None)
            os.environ.pop("WNV_DATABRICKS_SCHEMA", None)


@pytest.mark.integration
class TestExecutorIntegration:
    """Requires Databricks connectivity."""

    @pytest.fixture(autouse=True)
    def _skip_without_databricks(self) -> None:
        import os

        host = os.environ.get("WNV_DATABRICKS_HOST")
        token = os.environ.get("WNV_DATABRICKS_TOKEN")
        warehouse_id = os.environ.get("WNV_DATABRICKS_WAREHOUSE_ID")
        if not host or not token or not warehouse_id:
            pytest.skip("set WNV_DATABRICKS_* env vars for integration tests")

    def test_execute_simple_query(self) -> None:
        """Executor runs a simple query and returns results."""
        exe = executor_from_env()
        result = exe.execute("SELECT COUNT(*) AS cnt FROM eliao.wnv_demo.gold_county_month_wnv_weather")

        assert result.row_count == 1
        assert result.columns == ["cnt"]
        assert result.rows[0][0] > 0  # Gold table has rows
        assert result.query is not None
