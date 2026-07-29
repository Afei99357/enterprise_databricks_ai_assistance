"""Orchestrator agent for the WNV assistant.

Routes questions to the appropriate tool (analytics, document, or rejection)
and synthesizes the final answer. Supports conversation memory for follow-ups.
"""

from __future__ import annotations

import inspect
import json
import re
import uuid

from wnv_assistant.conversation.context import ConversationContextBuilder
from wnv_assistant.conversation.models import (
    AnalyticsState,
    ConversationTurn,
    QueryFilters,
    QueryProjection,
    ResultContext,
    Route,
)
from wnv_assistant.conversation.store import ConversationStore, InMemoryStore
from wnv_assistant.llm.client import LLMClient

from .models import AgentRequest, AgentResponse, ToolResult
from .routing import RouteDecision, keyword_route

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


def synthesize_answer(
    llm_client: LLMClient, question: str, tool_answer: str, data: list[dict]
) -> str:
    """Call the LLM to write a final answer from tool results."""
    from datetime import date, datetime

    sanitized = []
    for row in data[:5]:
        sanitized.append(
            {
                k: (v.isoformat() if isinstance(v, (date, datetime)) else v)
                for k, v in row.items()
            }
        )
    data_preview = json.dumps(sanitized, indent=2)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a WNV surveillance assistant. Write a clear, "
                "concise answer from the data below. Describe associations, "
                "not causation. Never diagnose. If data is empty, say so plainly."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\nData:\n{data_preview}\n\n"
                f"Tool summary: {tool_answer}"
            ),
        },
    ]
    response = llm_client.chat(messages, max_tokens=500, temperature=0.0)
    return response.strip() or tool_answer


class Orchestrator:
    """Routes questions to tools and synthesizes answers.

    Supports conversation memory: loads history, builds context,
    passes to LLM prompts, saves turns after processing.
    """

    def __init__(
        self,
        analytics_tool: object | None = None,
        llm_classify: callable | None = None,
        llm_answer: callable | None = None,
        conversation_store: ConversationStore | None = None,
    ) -> None:
        self.analytics_tool = analytics_tool
        self._llm_classify = llm_classify
        self._llm_answer = llm_answer
        self._store = conversation_store or InMemoryStore()
        self._context_builder = ConversationContextBuilder()

    def run(self, request: AgentRequest) -> AgentResponse:
        """Process a request: load context -> route -> tool -> synthesize -> save."""
        # Step 0: Load conversation context
        turns = []
        if self._has_conversation_identity(request):
            turns = self._store.get_recent_turns(
                user_id=request.user_id,
                conversation_id=request.conversation_id,
                limit=10,
            )
        context = self._context_builder.build(turns)
        context_prompt = self._context_builder.render_context_prompt(context)

        # Step 1: Route (with context)
        decision = self._route(request.question, context_prompt)

        # Step 2: Execute tool
        tool_result = self._execute(decision.route, request.question, context_prompt)

        # Step 3: Synthesize answer
        response = self._build_response(decision, request.question, tool_result)

        # Step 4: Save turns
        if self._has_conversation_identity(request):
            self._save_turns(request, response, decision)

        return response

    def _route(self, question: str, context_prompt: str) -> RouteDecision:
        """Classify a question into a route."""
        if self._llm_classify:
            result = _call_with_context(
                self._llm_classify, question, context_prompt=context_prompt
            )
            if isinstance(result, RouteDecision):
                return result
            if isinstance(result, tuple) and len(result) == 2:
                return RouteDecision(route=result[0], reason=result[1])
            return RouteDecision(route="ANALYTICS")
        return keyword_route(question)

    def _execute(
        self, route: str, question: str, context_prompt: str
    ) -> ToolResult | None:
        """Execute the appropriate tool."""
        if route in ("ANALYTICS", "MIXED") and self.analytics_tool:
            try:
                result = _call_with_context(
                    self.analytics_tool.answer,
                    question,
                    context_prompt=context_prompt,
                )
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

    def _build_response(
        self,
        decision: RouteDecision,
        question: str,
        tool_result: ToolResult | None,
    ) -> AgentResponse:
        """Build the final agent response."""
        if decision.route == "OUT_OF_SCOPE":
            return AgentResponse(
                answer=self._out_of_scope_answer(question),
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

        # ANALYTICS or MIXED
        if tool_result and tool_result.success:
            answer = self._synthesize_answer(question, tool_result)
        else:
            answer = (
                "I couldn't retrieve the data. "
                "The query may have failed or returned no results."
            )

        warnings = []
        if decision.route == "MIXED":
            warnings.append(
                "Document evidence is not yet available; analytics portion only."
            )

        return AgentResponse(
            answer=answer,
            route=decision.route,
            tool_results=[tool_result] if tool_result else [],
            warnings=warnings,
            insufficient_evidence=not tool_result or not tool_result.data,
        )

    def _synthesize_answer(self, question: str, tool_result: ToolResult) -> str:
        """Synthesize a final answer from tool results."""
        if self._llm_answer:
            return self._llm_answer(question, tool_result)
        return tool_result.answer

    def _out_of_scope_answer(self, question: str) -> str:
        """Generate a polite out-of-scope response."""
        return (
            "I can help with questions about West Nile virus surveillance data "
            "in Illinois (counts by county, trends over time, weather patterns). "
            "I cannot provide medical advice or diagnoses. "
            "Try asking about WNV activity in a specific county or time period."
        )

    @staticmethod
    def _has_conversation_identity(request: AgentRequest) -> bool:
        """Only persist history when both caller identity fields are present."""
        return bool(request.user_id and request.conversation_id)

    def _save_turns(
        self,
        request: AgentRequest,
        response: AgentResponse,
        decision: RouteDecision,
    ) -> None:
        """Save user and assistant turns to conversation store."""
        turn_number = self._next_turn_number(request)

        # User turn
        user_turn = ConversationTurn(
            turn_id=str(uuid.uuid4()),
            turn_number=turn_number,
            role="user",
            content=request.question,
        )

        # Assistant turn
        assistant_turn = ConversationTurn(
            turn_id=str(uuid.uuid4()),
            turn_number=turn_number + 1,
            role="assistant",
            content=response.answer,
            route=Route(response.route),
            generated_sql=(
                response.tool_results[0].generated_sql
                if response.tool_results and response.tool_results[0].generated_sql
                else None
            ),
            result_summary=response.answer[:1500],
        )

        # Extract analytics state from tool result
        if response.route in ("ANALYTICS", "MIXED") and response.tool_results:
            tr = response.tool_results[0]
            if tr.success and tr.data:
                assistant_turn = ConversationTurn(
                    **{
                        **assistant_turn.__dict__,
                        "analytics_state": self._extract_state(tr),
                    },
                )

        self._store.append_turns(
            user_id=request.user_id,
            conversation_id=request.conversation_id,
            turns=[user_turn, assistant_turn],
        )

    def _next_turn_number(self, request: AgentRequest) -> int:
        """Get the next turn number for this conversation."""
        turns = self._store.get_recent_turns(
            user_id=request.user_id,
            conversation_id=request.conversation_id,
            limit=100,
        )
        if not turns:
            return 1
        return max(t.turn_number for t in turns) + 1

    def _extract_state(self, tool_result: ToolResult) -> AnalyticsState | None:
        """Extract analytics state from tool result."""
        if not tool_result.data:
            return None

        # Extract counties from result data
        counties = list(
            {row.get("county") for row in tool_result.data if row.get("county")}
        )[:10]

        years = {
            int(value)
            for value in re.findall(
                r"\byear\s*=\s*'?([12]\d{3})'?",
                tool_result.generated_sql,
                flags=re.IGNORECASE,
            )
        }
        years.update(
            int(row["year"]) for row in tool_result.data if row.get("year") is not None
        )
        limit_match = re.search(
            r"\bLIMIT\s+(\d+)", tool_result.generated_sql, flags=re.IGNORECASE
        )
        aggregation_match = re.search(
            r"\b(SUM|AVG|COUNT|MIN|MAX)\s*\(\s*(\w+)",
            tool_result.generated_sql,
            flags=re.IGNORECASE,
        )

        return AnalyticsState(
            filters=QueryFilters(counties=counties, years=sorted(years)),
            projection=QueryProjection(
                metric=(
                    aggregation_match.group(2).lower() if aggregation_match else None
                ),
                aggregation=(
                    aggregation_match.group(1).upper() if aggregation_match else None
                ),
                limit=int(limit_match.group(1)) if limit_match else None,
            ),
            result_context=ResultContext(
                returned_counties=counties,
                returned_years=sorted(years),
                row_count=len(tool_result.data),
            ),
        )


def _call_with_context(callback, *args, context_prompt: str):
    """Invoke new context-aware callbacks while supporting existing callbacks."""
    try:
        inspect.signature(callback).bind(*args, context_prompt)
    except TypeError:
        return callback(*args)
    return callback(*args, context_prompt)
