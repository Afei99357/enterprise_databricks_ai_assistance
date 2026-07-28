"""Tests for conversation models."""

from __future__ import annotations

from wnv_assistant.conversation.models import (
    Aggregation,
    AnalyticsQuerySpec,
    AnalyticsState,
    ConversationTurn,
    QueryFilters,
    QueryProjection,
    ResultContext,
    Route,
)


class TestEnums:
    def test_route_values(self) -> None:
        assert Route.ANALYTICS == "ANALYTICS"
        assert Route.DOCUMENT == "DOCUMENT"
        assert Route.MIXED == "MIXED"
        assert Route.OUT_OF_SCOPE == "OUT_OF_SCOPE"

    def test_aggregation_values(self) -> None:
        assert Aggregation.SUM == "SUM"
        assert Aggregation.AVG == "AVG"


class TestAnalyticsState:
    def test_query_spec_to_state(self) -> None:
        spec = AnalyticsQuerySpec(
            metric="mosquito_count",
            aggregation="SUM",
            dimensions=["county"],
            filters={"year": [2022], "state": ["Illinois"]},
            order_by="mosquito_count",
            order_direction="DESC",
            limit=5,
        )
        state = spec.to_analytics_state(
            ResultContext(returned_counties=["Cook", "DuPage"], row_count=2)
        )

        assert state.filters.years == [2022]
        assert state.filters.states == ["Illinois"]
        assert state.projection.metric == "mosquito_count"
        assert state.projection.aggregation == "SUM"
        assert state.projection.limit == 5
        assert state.result_context.returned_counties == ["Cook", "DuPage"]

    def test_state_to_dict(self) -> None:
        state = AnalyticsState(
            filters=QueryFilters(years=[2022]),
            projection=QueryProjection(metric="mosquito_count"),
            result_context=ResultContext(row_count=5),
        )
        d = state.to_dict()
        assert d["filters"]["years"] == [2022]
        assert d["projection"]["metric"] == "mosquito_count"

    def test_state_round_trips_through_dict(self) -> None:
        state = AnalyticsState(
            filters=QueryFilters(years=[2022], counties=["Cook"]),
            projection=QueryProjection(metric="mosquito_count", aggregation="SUM"),
            result_context=ResultContext(returned_counties=["Cook"], row_count=1),
        )

        assert AnalyticsState.from_dict(state.to_dict()) == state

    def test_empty_filters_serialized_clean(self) -> None:
        filters = QueryFilters()
        assert filters.to_dict() == {}


class TestConversationTurn:
    def test_user_turn(self) -> None:
        turn = ConversationTurn(
            turn_id="t1",
            turn_number=1,
            role="user",
            content="Show top 5 counties",
        )
        d = turn.to_dict()
        assert d["role"] == "user"
        assert d["content"] == "Show top 5 counties"
        assert "route" not in d

    def test_assistant_turn_with_state(self) -> None:
        state = AnalyticsState(
            filters=QueryFilters(years=[2022]),
            projection=QueryProjection(metric="mosquito_count", limit=5),
        )
        turn = ConversationTurn(
            turn_id="t2",
            turn_number=2,
            role="assistant",
            content="Cook had the most.",
            route=Route.ANALYTICS,
            generated_sql="SELECT ...",
            analytics_state=state,
        )
        d = turn.to_dict()
        assert d["route"] == "ANALYTICS"
        assert "analytics_state_json" in d
