"""Tests for the orchestrator agent."""

from __future__ import annotations

from wnv_assistant.agent.models import AgentRequest, AgentResponse
from wnv_assistant.agent.orchestrator import Orchestrator
from wnv_assistant.agent.routing import RouteDecision


class MockAnalyticsTool:
    """Mock analytics tool for testing."""

    def __init__(self, data: list[dict] | None = None, error: str | None = None):
        self._data = data or []
        self._error = error

    def answer(self, question: str):
        from wnv_assistant.analytics.text_to_sql import AnalyticsResult

        if self._error:
            raise RuntimeError(self._error)

        return AnalyticsResult(
            answer=f"Found {len(self._data)} results for: {question}",
            data=self._data,
            columns=list(self._data[0].keys()) if self._data else [],
            generated_sql="SELECT * FROM gold LIMIT 10",
            row_count=len(self._data),
            warnings=[],
        )


class TestOrchestratorAnalytics:
    """Test the orchestrator with analytics questions."""

    def test_analytics_route_returns_data(self) -> None:
        """Analytics question routes to tool and returns results."""
        tool = MockAnalyticsTool(data=[
            {"county": "Cook", "mosquito_count": 42, "year": 2022},
        ])
        orch = Orchestrator(
            analytics_tool=tool,
            llm_classify=lambda q: RouteDecision("ANALYTICS"),
        )

        response = orch.run(AgentRequest(question="How many cases in Cook County?"))

        assert response.route == "ANALYTICS"
        assert len(response.tool_results) == 1
        assert response.tool_results[0].success
        assert "Found 1 result" in response.answer

    def test_analytics_tool_failure(self) -> None:
        """Tool failure returns a graceful error message."""
        tool = MockAnalyticsTool(error="Connection timeout")
        orch = Orchestrator(
            analytics_tool=tool,
            llm_classify=lambda q: RouteDecision("ANALYTICS"),
        )

        response = orch.run(AgentRequest(question="Show data"))

        assert response.route == "ANALYTICS"
        assert response.insufficient_evidence  # tool failed, no evidence
        assert "couldn't retrieve" in response.answer.lower()

    def test_analytics_empty_result(self) -> None:
        """Empty data sets insufficient_evidence flag."""
        tool = MockAnalyticsTool(data=[])
        orch = Orchestrator(
            analytics_tool=tool,
            llm_classify=lambda q: RouteDecision("ANALYTICS"),
        )

        response = orch.run(AgentRequest(question="Show data for Antarctica"))

        assert response.insufficient_evidence


class TestOrchestratorOutOfScope:
    """Test out-of-scope handling."""

    def test_out_of_scope_returns_safe_message(self) -> None:
        orch = Orchestrator(
            llm_classify=lambda q: RouteDecision("OUT_OF_SCOPE"),
        )

        response = orch.run(AgentRequest(question="Diagnose my fever"))

        assert response.route == "OUT_OF_SCOPE"
        assert response.warnings
        assert "cannot provide medical advice" in response.answer.lower() or "outside the scope" in response.answer.lower()

    def test_out_of_scope_has_no_tool_results(self) -> None:
        orch = Orchestrator(
            llm_classify=lambda q: RouteDecision("OUT_OF_SCOPE"),
        )

        response = orch.run(AgentRequest(question="What's the weather in Tokyo?"))

        assert response.tool_results == []


class TestOrchestratorDocument:
    """Test document route (not yet available)."""

    def test_document_route_returns_unavailable_message(self) -> None:
        orch = Orchestrator(
            llm_classify=lambda q: RouteDecision("DOCUMENT"),
        )

        response = orch.run(AgentRequest(question="What is WNV?"))

        assert response.route == "DOCUMENT"
        assert "not yet available" in response.answer.lower()


class TestOrchestratorMixed:
    """Test MIXED route: analytics runs, with a warning that RAG isn't available yet."""

    def test_mixed_route_runs_analytics_with_warning(self) -> None:
        tool = MockAnalyticsTool(
            data=[{"county": "Cook", "mosquito_count": 42, "year": 2022}]
        )
        orch = Orchestrator(
            analytics_tool=tool,
            llm_classify=lambda q: RouteDecision("MIXED"),
        )

        response = orch.run(
            AgentRequest(question="Show 2022 counts and CDC guidance")
        )

        assert response.route == "MIXED"
        assert response.tool_results[0].success
        assert any("not yet available" in w.lower() for w in response.warnings)


class TestAgentResponse:
    """Test response serialization."""

    def test_to_dict(self) -> None:
        from wnv_assistant.agent.models import ToolResult

        response = AgentResponse(
            answer="42 cases found",
            route="ANALYTICS",
            tool_results=[ToolResult(
                tool_name="text_to_sql",
                success=True,
                data=[{"county": "Cook", "count": 42}],
                generated_sql="SELECT ...",
            )],
            warnings=[],
        )

        d = response.to_dict()
        assert d["answer"] == "42 cases found"
        assert d["route"] == "ANALYTICS"
        assert len(d["tool_results"]) == 1
        assert d["tool_results"][0]["generated_sql"] == "SELECT ..."
