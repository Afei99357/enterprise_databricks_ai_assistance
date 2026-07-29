"""Tests for the consolidated routing module."""

from __future__ import annotations

import pytest

from wnv_assistant.agent.routing import RouteDecision, classify_route, keyword_route


class TestKeywordRoute:
    """Offline fallback routing — never produces MIXED (see keyword_route docstring)."""

    def test_analytics_question(self) -> None:
        decision = keyword_route("How many mosquito cases in Cook County?")
        assert decision.route == "ANALYTICS"

    def test_trend_question(self) -> None:
        decision = keyword_route("Show the trend of WNV activity over the years")
        assert decision.route == "ANALYTICS"

    def test_medical_question(self) -> None:
        decision = keyword_route("Am I at risk for WNV?")
        assert decision.route == "OUT_OF_SCOPE"

    def test_general_knowledge(self) -> None:
        decision = keyword_route("What is West Nile virus?")
        assert decision.route == "DOCUMENT"

    def test_returns_route_decision(self) -> None:
        decision = keyword_route("Show data")
        assert isinstance(decision, RouteDecision)


class FakeLLMClient:
    """Fake LLMClient — records the last messages sent and returns a canned reply."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.last_messages: list[dict] | None = None

    def chat(self, messages, *, max_tokens=2000, temperature=0.0) -> str:
        self.last_messages = messages
        return self.reply


class TestClassifyRoute:
    def test_parses_valid_json_response(self) -> None:
        client = FakeLLMClient('{"route": "MIXED", "reason": "needs both"}')
        decision = classify_route(client, "Show 2022 counts and CDC guidance")
        assert decision.route == "MIXED"
        assert decision.reason == "needs both"

    def test_falls_back_to_keyword_route_on_bad_json(self) -> None:
        client = FakeLLMClient("not json")
        decision = classify_route(client, "How many mosquito cases in 2022?")
        assert decision.route == "ANALYTICS"

    def test_falls_back_to_keyword_route_on_invalid_route_value(self) -> None:
        client = FakeLLMClient('{"route": "analytics", "reason": "lowercase"}')
        decision = classify_route(client, "How many mosquito cases in 2022?")
        assert decision.route == "ANALYTICS"
        assert decision.reason == "Data/analytics question"

    def test_falls_back_to_keyword_route_on_hallucinated_route_value(self) -> None:
        client = FakeLLMClient('{"route": "DATA", "reason": "made up"}')
        decision = classify_route(client, "What is West Nile virus?")
        assert decision.route == "DOCUMENT"
        assert decision.reason == "General knowledge question"

    def test_includes_context_in_system_prompt(self) -> None:
        client = FakeLLMClient('{"route": "ANALYTICS", "reason": "x"}')
        classify_route(client, "What about 2021?", context_prompt="year: [2022]")
        system_message = client.last_messages[0]["content"]
        assert "year: [2022]" in system_message


@pytest.mark.integration
class TestClassifyRouteIntegration:
    """Requires Databricks connectivity and LLM endpoint."""

    @pytest.fixture(autouse=True)
    def _skip_without_databricks(self) -> None:
        import os

        if not (
            os.environ.get("WNV_DATABRICKS_HOST")
            and os.environ.get("WNV_DATABRICKS_TOKEN")
            and os.environ.get("WNV_LLM_ENDPOINT")
        ):
            pytest.skip(
                "set WNV_DATABRICKS_* and WNV_LLM_ENDPOINT for LLM integration tests"
            )

    def test_classify_route_returns_valid_route(self) -> None:
        import os

        from wnv_assistant.llm.databricks_client import client_from_env

        client = client_from_env(endpoint=os.environ["WNV_LLM_ENDPOINT"])
        decision = classify_route(client, "How many cases in Cook County?")
        assert decision.route in ("ANALYTICS", "DOCUMENT", "MIXED", "OUT_OF_SCOPE")
        assert decision.reason
