"""Conversation memory models for the WNV assistant.

Defines the data structures for conversation turns, analytics state,
query specifications, and context configuration.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class Route(StrEnum):
    """Routing decision for a user question."""

    ANALYTICS = "ANALYTICS"
    DOCUMENT = "DOCUMENT"
    MIXED = "MIXED"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"


class Aggregation(StrEnum):
    """SQL aggregation function."""

    SUM = "SUM"
    AVG = "AVG"
    COUNT = "COUNT"
    MIN = "MIN"
    MAX = "MAX"


@dataclass(frozen=True)
class QueryFilters:
    """Filters applied to the query (WHERE clause semantics)."""

    years: list[int] = field(default_factory=list)
    months: list[int] = field(default_factory=list)
    counties: list[str] = field(default_factory=list)
    states: list[str] = field(default_factory=list)
    source_types: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QueryFilters:
        """Restore filters from a stored JSON object."""
        return cls(
            years=[int(value) for value in data.get("years", [])],
            months=[int(value) for value in data.get("months", [])],
            counties=[str(value) for value in data.get("counties", [])],
            states=[str(value) for value in data.get("states", [])],
            source_types=[str(value) for value in data.get("source_types", [])],
        )


@dataclass(frozen=True)
class QueryProjection:
    """How the query projects and orders data."""

    metric: str | None = None
    aggregation: str | None = None
    group_by: list[str] = field(default_factory=list)
    order_by: str | None = None
    order_direction: str | None = None
    limit: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QueryProjection:
        """Restore a projection from a stored JSON object."""
        return cls(
            metric=data.get("metric"),
            aggregation=data.get("aggregation"),
            group_by=[str(value) for value in data.get("group_by", [])],
            order_by=data.get("order_by"),
            order_direction=data.get("order_direction"),
            limit=data.get("limit"),
        )


@dataclass(frozen=True)
class ResultContext:
    """Entities derived from query results."""

    returned_counties: list[str] = field(default_factory=list)
    returned_years: list[int] = field(default_factory=list)
    row_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResultContext:
        """Restore result entities from a stored JSON object."""
        return cls(
            returned_counties=[
                str(value) for value in data.get("returned_counties", [])
            ],
            returned_years=[int(value) for value in data.get("returned_years", [])],
            row_count=data.get("row_count"),
        )


@dataclass(frozen=True)
class AnalyticsState:
    """Structured state from a successful analytics query.

    Captures filters, projection, and result context for follow-up resolution.
    """

    filters: QueryFilters = field(default_factory=QueryFilters)
    projection: QueryProjection = field(default_factory=QueryProjection)
    result_context: ResultContext = field(default_factory=ResultContext)

    def to_dict(self) -> dict[str, Any]:
        return {
            "filters": self.filters.to_dict(),
            "projection": self.projection.to_dict(),
            "result_context": self.result_context.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnalyticsState:
        """Restore analytics state from the Delta-store JSON payload."""
        return cls(
            filters=QueryFilters.from_dict(data.get("filters", {})),
            projection=QueryProjection.from_dict(data.get("projection", {})),
            result_context=ResultContext.from_dict(data.get("result_context", {})),
        )


@dataclass(frozen=True)
class AnalyticsQuerySpec:
    """Structured query intent emitted by the LLM alongside SQL.

    Represents what the user wants to query in a machine-readable form.
    Used as the primary source for conversation state extraction.
    """

    metric: str
    aggregation: str
    dimensions: list[str]
    filters: dict[str, list[str | int]]
    order_by: str | None = None
    order_direction: str | None = None
    limit: int | None = None

    def to_analytics_state(
        self, result_context: ResultContext | None = None
    ) -> AnalyticsState:
        """Convert query spec to analytics state."""
        filters = QueryFilters(
            years=[int(v) for v in self.filters.get("year", [])],
            months=[int(v) for v in self.filters.get("month", [])],
            counties=[str(v) for v in self.filters.get("county", [])],
            states=[str(v) for v in self.filters.get("state", [])],
            source_types=[str(v) for v in self.filters.get("source_type", [])],
        )
        projection = QueryProjection(
            metric=self.metric,
            aggregation=self.aggregation,
            group_by=self.dimensions,
            order_by=self.order_by,
            order_direction=self.order_direction,
            limit=self.limit,
        )
        return AnalyticsState(
            filters=filters,
            projection=projection,
            result_context=result_context or ResultContext(),
        )


@dataclass(frozen=True)
class ConversationTurn:
    """A single message in a conversation.

    User turns have role="user". Assistant turns have role="assistant"
    and may include route, SQL, analytics state, and citations.
    """

    turn_id: str
    turn_number: int
    role: str  # "user" | "assistant"
    content: str
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    route: Route | None = None
    generated_sql: str | None = None
    result_summary: str | None = None
    analytics_state: AnalyticsState | None = None
    citations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "turn_id": self.turn_id,
            "turn_number": self.turn_number,
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at.isoformat(),
        }
        if self.route:
            d["route"] = self.route.value
        if self.generated_sql:
            d["generated_sql"] = self.generated_sql
        if self.result_summary:
            d["result_summary"] = self.result_summary
        if self.analytics_state:
            d["analytics_state_json"] = json.dumps(self.analytics_state.to_dict())
        if self.citations:
            d["citations_json"] = json.dumps(self.citations)
        return d


@dataclass(frozen=True)
class ConversationContext:
    """Context built from conversation history for the LLM.

    Contains recent messages and the latest analytics state.
    """

    recent_messages: list[ConversationTurn]
    latest_analytics_state: AnalyticsState | None = None


@dataclass(frozen=True)
class ConversationContextConfig:
    """Configuration for conversation context building."""

    max_messages: int = 10
    target_history_chars: int = 10_000
    hard_max_history_chars: int = 16_000
    max_result_summary_chars: int = 1_500
