"""Data models for the WNV assistant agent."""

from __future__ import annotations

from dataclasses import dataclass, field

from wnv_assistant.conversation.models import AnalyticsQuerySpec


@dataclass(frozen=True)
class AgentRequest:
    """Incoming request to the agent."""

    question: str
    conversation_id: str = ""
    user_id: str = ""


@dataclass(frozen=True)
class ToolResult:
    """Result from calling a tool."""

    tool_name: str
    success: bool
    data: list[dict] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    answer: str = ""
    generated_sql: str = ""
    citations: list[dict] = field(default_factory=list)
    error: str = ""
    query_spec: AnalyticsQuerySpec | None = None


@dataclass(frozen=True)
class AgentResponse:
    """Final response from the agent."""

    answer: str
    route: str  # ANALYTICS | DOCUMENT | OUT_OF_SCOPE
    tool_results: list[ToolResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    insufficient_evidence: bool = False

    def to_dict(self) -> dict:
        """Serialize to dict for MLflow serving."""
        return {
            "answer": self.answer,
            "route": self.route,
            "tool_results": [
                {
                    "tool_name": r.tool_name,
                    "success": r.success,
                    "answer": r.answer,
                    "generated_sql": r.generated_sql,
                    "row_count": len(r.data),
                    "error": r.error,
                }
                for r in self.tool_results
            ],
            "warnings": self.warnings,
            "insufficient_evidence": self.insufficient_evidence,
        }
