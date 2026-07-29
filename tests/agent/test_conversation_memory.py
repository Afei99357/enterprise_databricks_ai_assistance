"""End-to-end unit tests for analytics conversation memory."""

from __future__ import annotations

from wnv_assistant.agent.models import AgentRequest
from wnv_assistant.agent.orchestrator import Orchestrator
from wnv_assistant.agent.routing import RouteDecision
from wnv_assistant.analytics.text_to_sql import AnalyticsResult
from wnv_assistant.conversation.models import AnalyticsQuerySpec
from wnv_assistant.conversation.store import InMemoryStore


class ContextAwareAnalyticsTool:
    """Analytics fake that records the context supplied for each question."""

    def __init__(self) -> None:
        self.contexts: list[str] = []

    def answer(self, question: str, context: str = "") -> AnalyticsResult:
        self.contexts.append(context)
        year = 2021 if "2021" in question else 2022
        return AnalyticsResult(
            answer=f"Top counties for {year}",
            data=[{"county": "Cook", "total_mosquito_count": 10}],
            columns=["county", "total_mosquito_count"],
            generated_sql=(
                "SELECT county, SUM(mosquito_count) AS total_mosquito_count "
                "FROM gold_county_month_wnv_weather "
                f"WHERE year = {year} GROUP BY county LIMIT 5"
            ),
            row_count=1,
            warnings=[],
            query_spec=AnalyticsQuerySpec(
                metric="mosquito_count",
                aggregation="SUM",
                dimensions=["county"],
                filters={"year": [year]},
                limit=5,
            ),
        )


def test_follow_up_receives_prior_analytics_context() -> None:
    tool = ContextAwareAnalyticsTool()
    orchestrator = Orchestrator(
        analytics_tool=tool,
        llm_classify=lambda _question: RouteDecision("ANALYTICS"),
        conversation_store=InMemoryStore(),
    )
    identity = {"user_id": "user-1", "conversation_id": "conversation-1"}

    orchestrator.run(
        AgentRequest(
            question="Show top counties by mosquito activity in 2022", **identity
        )
    )
    response = orchestrator.run(AgentRequest(question="What about 2021?", **identity))

    assert response.route == "ANALYTICS"
    assert tool.contexts[0] == ""
    assert "year: [2022]" in tool.contexts[1]
    assert "returned counties: ['Cook']" in tool.contexts[1]


def test_requests_without_identity_do_not_persist_history() -> None:
    tool = ContextAwareAnalyticsTool()
    orchestrator = Orchestrator(
        analytics_tool=tool,
        llm_classify=lambda _question: RouteDecision("ANALYTICS"),
        conversation_store=InMemoryStore(),
    )

    orchestrator.run(AgentRequest(question="Show counties in 2022"))
    orchestrator.run(AgentRequest(question="What about 2021?"))

    assert tool.contexts == ["", ""]
