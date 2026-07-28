"""Tests for conversation context builder."""

from __future__ import annotations

from wnv_assistant.conversation.context import ConversationContextBuilder
from wnv_assistant.conversation.models import (
    AnalyticsState,
    ConversationContextConfig,
    ConversationTurn,
    QueryFilters,
    QueryProjection,
    ResultContext,
    Route,
)


class TestContextBuilder:
    def test_empty_turns(self) -> None:
        builder = ConversationContextBuilder()
        context = builder.build([])
        assert context.recent_messages == []
        assert context.latest_analytics_state is None

    def test_latest_analytics_state(self) -> None:
        state = AnalyticsState(
            filters=QueryFilters(years=[2022]),
            projection=QueryProjection(metric="mosquito_count", limit=5),
            result_context=ResultContext(returned_counties=["Cook", "DuPage"]),
        )
        turns = [
            ConversationTurn(turn_id="t1", turn_number=1, role="user", content="Q1"),
            ConversationTurn(
                turn_id="t2", turn_number=2, role="assistant",
                content="A1", route=Route.ANALYTICS, analytics_state=state,
            ),
        ]
        context = ConversationContextBuilder().build(turns)
        assert context.latest_analytics_state is state
        assert context.latest_analytics_state.filters.years == [2022]

    def test_max_messages_limit(self) -> None:
        config = ConversationContextConfig(max_messages=4)
        builder = ConversationContextBuilder(config)
        turns = [
            ConversationTurn(turn_id=f"t{i}", turn_number=i, role="user", content=f"Q{i}")
            for i in range(1, 11)
        ]
        context = builder.build(turns)
        assert len(context.recent_messages) == 4

    def test_character_budget_trim(self) -> None:
        config = ConversationContextConfig(
            max_messages=10,
            hard_max_history_chars=200,
        )
        builder = ConversationContextBuilder(config)
        turns = [
            ConversationTurn(
                turn_id=f"t{i}",
                turn_number=i,
                role="user" if i % 2 else "assistant",
                content="X" * 100,
            )
            for i in range(1, 11)
        ]
        context = builder.build(turns)
        total = sum(len(t.content) for t in context.recent_messages)
        assert total <= 200

    def test_render_context_prompt(self) -> None:
        state = AnalyticsState(
            filters=QueryFilters(years=[2022]),
            projection=QueryProjection(metric="mosquito_count", limit=5),
            result_context=ResultContext(returned_counties=["Cook", "DuPage"]),
        )
        turns = [
            ConversationTurn(turn_id="t1", turn_number=1, role="user", content="Show top 5 counties"),
            ConversationTurn(
                turn_id="t2", turn_number=2, role="assistant",
                content="Cook and DuPage were highest.",
                route=Route.ANALYTICS,
                result_summary="Cook: 2004, DuPage: 137",
                analytics_state=state,
            ),
        ]
        context = ConversationContextBuilder().build(turns)
        prompt = ConversationContextBuilder().render_context_prompt(context)

        assert "Recent conversation:" in prompt
        assert "Show top 5 counties" in prompt
        assert "Previous analytics state:" in prompt
        assert "year: [2022]" in prompt
        assert "metric: mosquito_count" in prompt

    def test_newest_state_wins(self) -> None:
        """When multiple turns have analytics state, the newest one is used."""
        state_2022 = AnalyticsState(filters=QueryFilters(years=[2022]))
        state_2021 = AnalyticsState(filters=QueryFilters(years=[2021]))

        turns = [
            ConversationTurn(turn_id="t1", turn_number=1, role="user", content="Q1"),
            ConversationTurn(
                turn_id="t2", turn_number=2, role="assistant",
                content="A1", route=Route.ANALYTICS, analytics_state=state_2022,
            ),
            ConversationTurn(turn_id="t3", turn_number=3, role="user", content="Q2"),
            ConversationTurn(
                turn_id="t4", turn_number=4, role="assistant",
                content="A2", route=Route.ANALYTICS, analytics_state=state_2021,
            ),
        ]
        context = ConversationContextBuilder().build(turns)
        assert context.latest_analytics_state is state_2021
