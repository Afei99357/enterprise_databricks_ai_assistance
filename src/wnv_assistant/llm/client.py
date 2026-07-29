"""Abstract LLM client interface."""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMClient(ABC):
    """Base interface for LLM clients — a thin transport, no domain logic."""

    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        *,
        max_tokens: int = 2000,
        temperature: float = 0.0,
    ) -> str:
        """Send a chat completion request and return the assistant's text response."""
        ...
