"""Tests for conversation store."""

from __future__ import annotations

from wnv_assistant.conversation.models import AnalyticsState, ConversationTurn, QueryFilters, QueryProjection, Route
from wnv_assistant.conversation.store import InMemoryStore


class TestInMemoryStore:
    def test_empty_conversation(self) -> None:
        store = InMemoryStore()
        turns = store.get_recent_turns(user_id="u1", conversation_id="c1", limit=10)
        assert turns == []

    def test_append_and_retrieve(self) -> None:
        store = InMemoryStore()
        turns = [
            ConversationTurn(turn_id="t1", turn_number=1, role="user", content="Q1"),
            ConversationTurn(turn_id="t2", turn_number=2, role="assistant", content="A1"),
        ]
        store.append_turns(user_id="u1", conversation_id="c1", turns=turns)

        result = store.get_recent_turns(user_id="u1", conversation_id="c1", limit=10)
        assert len(result) == 2
        assert result[0].content == "Q1"
        assert result[1].content == "A1"

    def test_limit_recent(self) -> None:
        store = InMemoryStore()
        for i in range(10):
            store.append_turns(
                user_id="u1",
                conversation_id="c1",
                turns=[ConversationTurn(turn_id=f"t{i}", turn_number=i, role="user", content=f"Q{i}")],
            )

        result = store.get_recent_turns(user_id="u1", conversation_id="c1", limit=3)
        assert len(result) == 3
        assert result[0].content == "Q7"
        assert result[2].content == "Q9"

    def test_isolated_conversations(self) -> None:
        store = InMemoryStore()
        store.append_turns(
            user_id="u1",
            conversation_id="c1",
            turns=[ConversationTurn(turn_id="t1", turn_number=1, role="user", content="Q1")],
        )
        store.append_turns(
            user_id="u2",
            conversation_id="c2",
            turns=[ConversationTurn(turn_id="t2", turn_number=1, role="user", content="Q2")],
        )

        assert len(store.get_recent_turns(user_id="u1", conversation_id="c1", limit=10)) == 1
        assert len(store.get_recent_turns(user_id="u2", conversation_id="c2", limit=10)) == 1

    def test_append_multiple_turns_atomic(self) -> None:
        """User and assistant turns appended together appear as a pair."""
        store = InMemoryStore()
        user_turn = ConversationTurn(turn_id="t1", turn_number=1, role="user", content="Q1")
        assistant_turn = ConversationTurn(
            turn_id="t2",
            turn_number=2,
            role="assistant",
            content="A1",
            route=Route.ANALYTICS,
            analytics_state=AnalyticsState(
                filters=QueryFilters(years=[2022]),
                projection=QueryProjection(metric="mosquito_count"),
            ),
        )
        store.append_turns(user_id="u1", conversation_id="c1", turns=[user_turn, assistant_turn])

        result = store.get_recent_turns(user_id="u1", conversation_id="c1", limit=10)
        assert len(result) == 2
        assert result[0].role == "user"
        assert result[1].role == "assistant"
        assert result[1].analytics_state is not None
