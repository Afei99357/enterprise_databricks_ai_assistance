"""Databricks Model Serving LLM client."""

from __future__ import annotations

import json
import time
from urllib import error, request

from .client import LLMClient


class DatabricksLLMClient(LLMClient):
    """Calls Databricks Model Serving endpoints.

    Uses the OpenAI-compatible chat completions format.
    """

    def __init__(
        self,
        *,
        host: str,
        token: str,
        endpoint: str = "databricks-meta-llama-3-3-70b-instruct",
        model: str | None = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        self.host = host.removeprefix("https://").removesuffix("/")
        self.token = token
        self.endpoint = endpoint
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    @property
    def _url(self) -> str:
        return f"https://{self.host}/serving-endpoints/{self.endpoint}/invocations"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    def chat(
        self, messages: list[dict], *, max_tokens: int = 2000, temperature: float = 0.0
    ) -> str:
        """Send a chat completion request to Databricks Model Serving."""
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if self.model:
            payload["model"] = self.model

        body = json.dumps(payload).encode()

        for attempt in range(self.max_retries):
            try:
                req = request.Request(
                    self._url, data=body, headers=self._headers(), method="POST"
                )
                with request.urlopen(req, timeout=120) as resp:
                    result = json.loads(resp.read().decode())
                    content = self._extract_content(result)
                    return content or ""
            except error.HTTPError as e:
                if e.code in (429, 500, 502, 503):
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay * (2**attempt))
                        continue
                raise RuntimeError(
                    f"LLM API error {e.code}: {e.read().decode()[:200]}"
                ) from e
            except Exception:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2**attempt))
                    continue
                raise

        return ""

    def _extract_content(self, result: dict) -> str:
        """Extract text content from various response formats."""
        message = result.get("choices", [{}])[0].get("message", {})
        content = message.get("content", "")

        # Plain string content
        if isinstance(content, str):
            return content

        # List of content parts (reasoning models)
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(part.get("text", ""))
            return "\n".join(parts)

        # Top-level content fallback
        content = result.get("content", "")
        return content if isinstance(content, str) else ""


def client_from_env(
    endpoint: str = "databricks-meta-llama-3-3-70b-instruct",
) -> DatabricksLLMClient:
    """Create a Databricks LLM client from environment variables."""
    import os

    host = os.environ.get("WNV_DATABRICKS_HOST", "")
    token = os.environ.get("WNV_DATABRICKS_TOKEN", "")
    endpoint_name = os.environ.get("WNV_LLM_ENDPOINT", endpoint)

    if not host or not token:
        raise RuntimeError("Set WNV_DATABRICKS_HOST and WNV_DATABRICKS_TOKEN")

    return DatabricksLLMClient(host=host, token=token, endpoint=endpoint_name)
