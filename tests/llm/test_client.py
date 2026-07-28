"""Tests for LLM client."""

from __future__ import annotations

from wnv_assistant.llm.databricks_client import _keyword_route


class TestKeywordRouteFallback:
    """Test the keyword-based fallback routing."""

    def test_analytics_question(self) -> None:
        route, reason = _keyword_route("How many mosquito cases in 2022?")
        assert route == "ANALYTICS"

    def test_document_question(self) -> None:
        route, reason = _keyword_route("What is West Nile virus?")
        assert route == "DOCUMENT"

    def test_out_of_scope(self) -> None:
        route, reason = _keyword_route("Am I at risk?")
        assert route == "OUT_OF_SCOPE"

    def test_returns_tuple(self) -> None:
        result = _keyword_route("Show data")
        assert isinstance(result, tuple)
        assert len(result) == 2


import pytest


@pytest.mark.integration
class TestDatabricksClientIntegration:
    """Requires Databricks connectivity and LLM endpoint."""

    @pytest.fixture(autouse=True)
    def _skip_without_databricks(self) -> None:
        import os

        host = os.environ.get("WNV_DATABRICKS_HOST")
        token = os.environ.get("WNV_DATABRICKS_TOKEN")
        endpoint = os.environ.get("WNV_LLM_ENDPOINT")
        if not host or not token or not endpoint:
            pytest.skip("set WNV_DATABRICKS_* and WNV_LLM_ENDPOINT for LLM integration tests")

    def test_chat_returns_text(self) -> None:
        """Basic chat completion returns non-empty text."""
        from wnv_assistant.llm.databricks_client import client_from_env

        endpoint = __import__("os").environ.get("WNV_LLM_ENDPOINT", "")
        client = client_from_env(endpoint=endpoint)
        response = client.chat(
            messages=[{"role": "user", "content": "Say hello in one word. Just the word."}],
            max_tokens=200,
        )
        assert response.strip()

    def test_generate_sql_returns_sql(self) -> None:
        """SQL generation returns something that looks like SQL."""
        from wnv_assistant.llm.databricks_client import client_from_env

        endpoint = __import__("os").environ.get("WNV_LLM_ENDPOINT", "")
        client = client_from_env(endpoint=endpoint)
        sql = client.generate_sql(
            question="Show top counties by mosquito count",
            system_prompt="Generate SQL. Return only the query.",
        )
        assert "SELECT" in sql.upper()

    def test_classify_route_returns_valid_route(self) -> None:
        """Route classification returns a known route."""
        from wnv_assistant.llm.databricks_client import client_from_env

        endpoint = __import__("os").environ.get("WNV_LLM_ENDPOINT", "")
        client = client_from_env(endpoint=endpoint)
        route, reason = client.classify_route("How many cases in Cook County?")
        assert route in ("ANALYTICS", "DOCUMENT", "OUT_OF_SCOPE")
        assert reason
