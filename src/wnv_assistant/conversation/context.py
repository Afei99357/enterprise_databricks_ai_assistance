"""Conversation context builder.

Builds ConversationContext from a sequence of turns, selecting
recent messages within character budget and extracting the latest
analytics state.
"""

from __future__ import annotations

from collections.abc import Sequence

from .models import (
    AnalyticsState,
    ConversationContext,
    ConversationContextConfig,
    ConversationTurn,
    Route,
)


class ConversationContextBuilder:
    """Builds conversation context from stored turns."""

    def __init__(self, config: ConversationContextConfig | None = None):
        self.config = config or ConversationContextConfig()

    def build(
        self,
        turns: Sequence[ConversationTurn],
    ) -> ConversationContext:
        """Build context from turns.

        Selects recent messages within character budget and extracts
        the latest analytics state.
        """
        # Get recent turns within message limit
        recent = list(turns[-self.config.max_messages :])

        # Trim by character budget
        selected = self._select_by_budget(recent)

        # Find latest analytics state
        latest_state = self._latest_analytics_state(selected)

        return ConversationContext(
            recent_messages=list(selected),
            latest_analytics_state=latest_state,
        )

    def _select_by_budget(
        self,
        turns: list[ConversationTurn],
    ) -> list[ConversationTurn]:
        """Select turns within character budget.

        Fills up to target_history_chars (~10k). Will not exceed
        hard_max_history_chars (16k) even if fewer turns are included.
        """
        selected: list[ConversationTurn] = []
        total_chars = 0

        for turn in reversed(turns):
            size = self._turn_char_size(turn)

            # Hard ceiling: never exceed this
            if total_chars + size > self.config.hard_max_history_chars:
                break

            # Soft target: stop if we've reached it and have at least 2 turns
            if total_chars >= self.config.target_history_chars and len(selected) >= 2:
                break

            selected.append(turn)
            total_chars += size

        return list(reversed(selected))

    def _turn_char_size(self, turn: ConversationTurn) -> int:
        """Estimate character size of a turn for prompt budgeting."""
        size = len(turn.content)
        if turn.generated_sql:
            size += len(turn.generated_sql)
        if turn.result_summary:
            size += min(
                len(turn.result_summary),
                self.config.max_result_summary_chars,
            )
        return size

    def _latest_analytics_state(
        self,
        turns: list[ConversationTurn],
    ) -> AnalyticsState | None:
        """Find the latest analytics state from turns."""
        for turn in reversed(turns):
            if (
                turn.role == "assistant"
                and turn.route in (Route.ANALYTICS, Route.MIXED)
                and turn.analytics_state is not None
            ):
                return turn.analytics_state
        return None

    def render_context_prompt(self, context: ConversationContext) -> str:
        """Render context as a string for LLM prompts."""
        lines: list[str] = []

        if context.recent_messages:
            lines.append("Recent conversation:")
            for turn in context.recent_messages:
                role_label = "User" if turn.role == "user" else "Assistant"
                lines.append(f"  {role_label}: {turn.content}")
                if turn.result_summary:
                    summary = turn.result_summary[
                        : self.config.max_result_summary_chars
                    ]
                    lines.append(f"  Result: {summary}")
            lines.append("")

        if context.latest_analytics_state:
            lines.append("Previous analytics state:")
            state = context.latest_analytics_state
            if state.filters.years:
                lines.append(f"  - year: {state.filters.years}")
            if state.filters.counties:
                lines.append(f"  - counties: {state.filters.counties}")
            if state.projection.metric:
                lines.append(f"  - metric: {state.projection.metric}")
            if state.projection.aggregation:
                lines.append(f"  - aggregation: {state.projection.aggregation}")
            if state.projection.group_by:
                lines.append(f"  - group by: {state.projection.group_by}")
            if state.projection.limit:
                lines.append(f"  - limit: {state.projection.limit}")
            if state.result_context.returned_counties:
                lines.append(
                    f"  - returned counties: {state.result_context.returned_counties}"
                )
            lines.append("")

        return "\n".join(lines) if lines else ""
