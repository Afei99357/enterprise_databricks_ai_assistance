"""Orchestrator agent for the WNV assistant.

Routes questions to the appropriate tool (analytics, document, or rejection)
and synthesizes the final answer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .models import AgentRequest, AgentResponse, ToolResult


@dataclass(frozen=True)
class RouteDecision:
    """Result of classifying a question."""

    route: str  # ANALYTICS | DOCUMENT | OUT_OF_SCOPE
    reason: str = ""


ROUTING_SYSTEM_PROMPT = """\
You are a routing agent for a West Nile virus (WNV) surveillance assistant.
Classify each question into one of three categories:

- ANALYTICS: Questions about counts, trends, comparisons, rankings, geography,
  dates, weather relationships, or any question that can be answered from
  structured surveillance data (bird, horse, mosquito counts by county and month).

- DOCUMENT: Questions about what WNV is, how it spreads, prevention, symptoms,
  treatment, CDC guidance, or public health recommendations.
  (Document retrieval is not yet available.)

- OUT_OF_SCOPE: Medical diagnosis, personal health advice, predictions about
  future outbreaks, causal claims, or completely unrelated topics.

Respond with JSON only: {"route": "ANALYTICS|DOCUMENT|OUT_OF_SCOPE", "reason": "..."}
"""

ANSWER_SYSTEM_PROMPT = """\
You are a West Nile virus (WNV) surveillance assistant for Illinois.
Synthesize a clear, concise answer from the provided data.

Rules:
- Describe relationships as associations, never causation
- NULL/missing data means "not reported", not zero
- Never provide medical diagnoses or personal health advice
- Be honest when the data doesn't support an answer
- Include specific numbers from the data when available
"""


class Orchestrator:
    """Routes questions to tools and synthesizes answers."""

    def __init__(
        self,
        analytics_tool: object | None = None,
        llm_classify: callable | None = None,
        llm_answer: callable | None = None,
    ) -> None:
        self.analytics_tool = analytics_tool
        self._llm_classify = llm_classify
        self._llm_answer = llm_answer

    def run(self, request: AgentRequest) -> AgentResponse:
        """Process a request: route -> tool -> synthesize."""
        # Step 1: Route
        decision = self._route(request.question)

        # Step 2: Execute tool
        tool_result = self._execute(decision.route, request.question)

        # Step 3: Synthesize answer
        if decision.route == "OUT_OF_SCOPE":
            return AgentResponse(
                answer=self._out_of_scope_answer(request.question),
                route="OUT_OF_SCOPE",
                tool_results=[],
                warnings=["Question is outside the scope of WNV surveillance data."],
            )

        if decision.route == "DOCUMENT":
            return AgentResponse(
                answer=(
                    "Document retrieval is not yet available. "
                    "I can answer questions about WNV surveillance data "
                    "(counts, trends, geography) if you'd like."
                ),
                route="DOCUMENT",
                tool_results=[],
                warnings=["Document retrieval not yet implemented."],
            )

        # ANALYTICS
        if tool_result and tool_result.success:
            answer = self._synthesize_answer(request.question, tool_result)
        else:
            answer = "I couldn't retrieve the data. The query may have failed or returned no results."

        return AgentResponse(
            answer=answer,
            route="ANALYTICS",
            tool_results=[tool_result] if tool_result else [],
            insufficient_evidence=not tool_result or not tool_result.data,
        )

    def _route(self, question: str) -> RouteDecision:
        """Classify a question into a route."""
        if self._llm_classify:
            result = self._llm_classify(question)
            # Handle both RouteDecision and tuple returns
            if isinstance(result, RouteDecision):
                return result
            if isinstance(result, tuple) and len(result) == 2:
                return RouteDecision(route=result[0], reason=result[1])
            return RouteDecision(route="ANALYTICS")
        return _keyword_route(question)

    def _execute(self, route: str, question: str) -> ToolResult | None:
        """Execute the appropriate tool."""
        if route == "ANALYTICS" and self.analytics_tool:
            try:
                result = self.analytics_tool.answer(question)
                return ToolResult(
                    tool_name="text_to_sql",
                    success=True,
                    data=result.data,
                    columns=result.columns,
                    answer=result.answer,
                    generated_sql=result.generated_sql,
                    error="",
                )
            except Exception as e:
                return ToolResult(
                    tool_name="text_to_sql",
                    success=False,
                    error=str(e),
                )
        return None

    def _synthesize_answer(self, question: str, tool_result: ToolResult) -> str:
        """Synthesize a final answer from tool results."""
        if self._llm_answer:
            return self._llm_answer(question, tool_result)
        # Fallback: use the tool's own answer
        return tool_result.answer

    def _out_of_scope_answer(self, question: str) -> str:
        """Generate a polite out-of-scope response."""
        return (
            "I can help with questions about West Nile virus surveillance data "
            "in Illinois (counts by county, trends over time, weather patterns). "
            "I cannot provide medical advice or diagnoses. "
            "Try asking about WNV activity in a specific county or time period."
        )


def _keyword_route(question: str) -> RouteDecision:
    """Simple keyword-based routing fallback (no LLM needed)."""
    q = question.lower()

    analytics_keywords = [
        "how many", "how much", "count", "total", "compare", "trend",
        "which county", "what year", "highest", "lowest", "average",
        "mosquito", "bird", "horse", "case", "activity", "weather",
        "temperature", "precipitation", "200", "201", "202",
    ]

    out_of_scope_keywords = [
        "diagnose", "am i", "do i have", "should i take", "prescribe",
        "treatment for me", "my symptoms",
    ]

    if any(kw in q for kw in out_of_scope_keywords):
        return RouteDecision(route="OUT_OF_SCOPE", reason="Medical/personal health question")

    if any(kw in q for kw in analytics_keywords):
        return RouteDecision(route="ANALYTICS", reason="Data/analytics question")

    return RouteDecision(route="DOCUMENT", reason="General knowledge question")
