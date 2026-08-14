"""Databricks Foundation Model API embedding client."""

from __future__ import annotations

import json
import time
from urllib import error, request


class EmbeddingClient:
    """Calls a Databricks Model Serving embeddings endpoint.

    Uses the OpenAI-compatible embeddings format.
    """

    def __init__(
        self,
        *,
        host: str,
        token: str,
        endpoint: str = "databricks-gte-large-en",
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        self.host = host.removeprefix("https://").removesuffix("/")
        self.token = token
        self.endpoint = endpoint
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

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, preserving input order."""
        payload = {"input": texts}
        body = json.dumps(payload).encode()

        for attempt in range(self.max_retries):
            try:
                req = request.Request(
                    self._url, data=body, headers=self._headers(), method="POST"
                )
                with request.urlopen(req, timeout=60) as resp:
                    result = json.loads(resp.read().decode())
                    return [row["embedding"] for row in result["data"]]
            except error.HTTPError as e:
                if e.code in (429, 500, 502, 503):
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay * (2**attempt))
                        continue
                raise RuntimeError(
                    f"Embedding API error {e.code}: {e.read().decode()[:200]}"
                ) from e
            except Exception:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2**attempt))
                    continue
                raise

        return []


def embed_from_env() -> EmbeddingClient:
    """Create an EmbeddingClient from environment variables.

    Reads WNV_DATABRICKS_HOST, WNV_DATABRICKS_TOKEN, WNV_EMBEDDING_ENDPOINT.
    """
    import os

    host = os.environ.get("WNV_DATABRICKS_HOST", "")
    token = os.environ.get("WNV_DATABRICKS_TOKEN", "")
    endpoint = os.environ.get("WNV_EMBEDDING_ENDPOINT", "databricks-gte-large-en")

    if not host or not token:
        raise RuntimeError("Set WNV_DATABRICKS_HOST and WNV_DATABRICKS_TOKEN")

    return EmbeddingClient(host=host, token=token, endpoint=endpoint)
