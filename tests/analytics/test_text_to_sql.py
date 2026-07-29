"""Tests for text-to-SQL tool (mock LLM, no Databricks needed)."""

from __future__ import annotations

import pytest

from wnv_assistant.analytics.executor import Executor, QueryResult
from wnv_assistant.analytics.text_to_sql import AnalyticsResult, TextToSQLTool
from wnv_assistant.conversation.models import AnalyticsQuerySpec


class FakeLLMClient:
    """Fake LLMClient for testing generate_sql() without a real endpoint."""

    def __init__(self, reply: str) -> None:
        self.reply = reply

    def chat(self, messages, *, max_tokens=2000, temperature=0.0) -> str:
        return self.reply


def test_generate_sql_parses_json_envelope() -> None:
    from wnv_assistant.analytics.text_to_sql import generate_sql
    from wnv_assistant.conversation.models import AnalyticsQuerySpec

    reply = (
        '{"sql": "SELECT county FROM gold_county_month_wnv_weather LIMIT 5", '
        '"query_spec": {"metric": "mosquito_count", "aggregation": "SUM", '
        '"dimensions": ["county"], "filters": {"year": [2022]}, '
        '"order_by": "total", "order_direction": "DESC", "limit": 5}}'
    )
    client = FakeLLMClient(reply)
    sql, spec = generate_sql(client, "trivial", "system prompt")

    assert sql == "SELECT county FROM gold_county_month_wnv_weather LIMIT 5"
    assert isinstance(spec, AnalyticsQuerySpec)
    assert spec.metric == "mosquito_count"
    assert spec.filters == {"year": [2022]}


def test_generate_sql_falls_back_when_not_json() -> None:
    """A model that ignores the envelope instruction still gets its SQL used."""
    from wnv_assistant.analytics.text_to_sql import generate_sql

    client = FakeLLMClient("SELECT county FROM gold_county_month_wnv_weather LIMIT 5")
    sql, spec = generate_sql(client, "trivial", "system prompt")

    assert sql == "SELECT county FROM gold_county_month_wnv_weather LIMIT 5"
    assert spec is None


def _mock_llm_generate(
    question: str, system_prompt: str
) -> tuple[str, AnalyticsQuerySpec | None]:
    """Return known-good SQL and query spec for test questions."""
    if "Cook County" in question or "cook" in question.lower():
        return (
            "SELECT county, mosquito_count, year FROM gold_county_month_wnv_weather "
            "WHERE county = 'cook' LIMIT 100",
            AnalyticsQuerySpec(
                metric="mosquito_count",
                aggregation="",
                dimensions=["county", "year"],
                filters={"county": ["cook"]},
                limit=100,
            ),
        )
    if "highest" in question.lower() or "top" in question.lower():
        return (
            "SELECT county, SUM(mosquito_count) AS total "
            "FROM gold_county_month_wnv_weather "
            "GROUP BY county ORDER BY total DESC LIMIT 10",
            AnalyticsQuerySpec(
                metric="mosquito_count",
                aggregation="SUM",
                dimensions=["county"],
                filters={},
                order_by="total",
                order_direction="DESC",
                limit=10,
            ),
        )
    return (
        "SELECT county, year, mosquito_count "
        "FROM gold_county_month_wnv_weather LIMIT 10",
        None,
    )


def _mock_executor_failing(*args, **kwargs) -> Executor:
    """Executor that raises on execute."""
    class FailingExecutor(Executor):
        def execute(self, sql: str) -> QueryResult:
            raise RuntimeError("Connection refused")
    return FailingExecutor()


def test_answer_with_valid_sql() -> None:
    """Tool returns AnalyticsResult with data when SQL is valid."""
    tool = TextToSQLTool(
        executor=_mock_executor_failing(),  # won't be called, we mock below
        llm_generate=_mock_llm_generate,
    )
    # Patch executor to return known data
    tool.executor.execute = lambda sql: QueryResult(
        columns=["county", "mosquito_count", "year"],
        rows=[("cook", 42, 2022),("cook", 38, 2021)],
        row_count=2,
        query=sql,
    )

    result = tool.answer("How many mosquito cases in Cook County?")

    assert isinstance(result, AnalyticsResult)
    assert result.row_count == 2
    assert len(result.data) == 2
    assert result.data[0]["county"] == "cook"
    assert not result.warnings
    assert "Found 2 result" in result.answer


def test_answer_with_validation_failure() -> None:
    """Tool returns safe rejection when SQL fails validation."""
    def bad_llm(question: str, system_prompt: str) -> tuple[str, None]:
        return "DROP TABLE gold_county_month_wnv_weather", None

    tool = TextToSQLTool(
        executor=_mock_executor_failing(),
        llm_generate=bad_llm,
    )

    result = tool.answer("Drop the table")

    assert result.warnings
    assert "Validation failed" in result.warnings[0]
    assert result.row_count == 0
    assert result.data == []


def test_answer_empty_result() -> None:
    """Tool returns helpful message when query returns no rows."""
    tool = TextToSQLTool(
        executor=_mock_executor_failing(),
        llm_generate=_mock_llm_generate,
    )
    tool.executor.execute = lambda sql: QueryResult(
        columns=["county", "mosquito_count"],
        rows=[],
        row_count=0,
        query=sql,
    )

    result = tool.answer("Show me data for Antarctica")

    assert result.row_count == 0
    assert "No data found" in result.answer


def test_generated_sql_is_preserved() -> None:
    """Generated SQL is always included for traceability."""
    tool = TextToSQLTool(
        executor=_mock_executor_failing(),
        llm_generate=_mock_llm_generate,
    )
    tool.executor.execute = lambda sql: QueryResult(
        columns=["county"],
        rows=[("cook",)],
        row_count=1,
        query=sql,
    )

    result = tool.answer("Show Cook County")

    assert "SELECT" in result.generated_sql
    assert "gold_county_month_wnv_weather" in result.generated_sql


def test_query_spec_is_preserved() -> None:
    """The structured query spec produced by the LLM reaches AnalyticsResult."""
    tool = TextToSQLTool(
        executor=_mock_executor_failing(),
        llm_generate=_mock_llm_generate,
    )
    tool.executor.execute = lambda sql: QueryResult(
        columns=["county", "total"],
        rows=[("cook", 42)],
        row_count=1,
        query=sql,
    )

    result = tool.answer("Show the highest counties")

    assert result.query_spec is not None
    assert result.query_spec.metric == "mosquito_count"
    assert result.query_spec.aggregation == "SUM"


def test_format_schema_includes_allowed_columns() -> None:
    """Schema prompt includes column descriptions."""
    from wnv_assistant.analytics.schema import format_schema_for_prompt

    schema = format_schema_for_prompt()
    assert "gold_county_month_wnv_weather" in schema
    assert "mosquito_count" in schema
    assert "county" in schema
    assert "NULL" in schema or "not reported" in schema


@pytest.mark.integration
def test_generate_sql_integration() -> None:
    """Real LLM call produces SQL-looking text (requires Databricks env)."""
    import os

    import pytest as _pytest

    if not (
        os.environ.get("WNV_DATABRICKS_HOST")
        and os.environ.get("WNV_DATABRICKS_TOKEN")
        and os.environ.get("WNV_LLM_ENDPOINT")
    ):
        _pytest.skip(
            "set WNV_DATABRICKS_* and WNV_LLM_ENDPOINT for LLM integration tests"
        )

    from wnv_assistant.analytics.schema import format_schema_for_prompt
    from wnv_assistant.analytics.text_to_sql import generate_sql
    from wnv_assistant.llm.databricks_client import client_from_env

    client = client_from_env(endpoint=os.environ["WNV_LLM_ENDPOINT"])
    schema_prompt = format_schema_for_prompt()
    sql, _spec = generate_sql(
        client,
        question="Show top counties by mosquito count",
        system_prompt=f"Generate SQL. Return only the query.\n{schema_prompt}",
    )
    assert "SELECT" in sql.upper()
