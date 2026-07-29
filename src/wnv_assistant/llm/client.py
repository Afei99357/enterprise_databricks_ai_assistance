"""Abstract LLM client interface."""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMClient(ABC):
    """Base interface for LLM clients."""

    @abstractmethod
    def chat(self, messages: list[dict], *, max_tokens: int = 2000, temperature: float = 0.0) -> str:
        """Send a chat completion request and return the assistant's text response."""
        ...

    @abstractmethod
    def generate_sql(self, question: str, system_prompt: str) -> str:
        """Generate SQL from a natural language question."""
        ...

    def synthesize_answer(self, question: str, tool_answer: str, data: list[dict]) -> str:
        """Synthesize a final answer from tool results.

        Override in subclasses for LLM-based synthesis. Falls back to tool answer.
        """
        return tool_answer
