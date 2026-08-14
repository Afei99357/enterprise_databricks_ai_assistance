"""Tests for the Databricks embeddings client."""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from wnv_assistant.documents.embedding import EmbeddingClient, embed_from_env


class TestEmbeddingClient:
    def test_embed_returns_vectors_in_order(self) -> None:
        client = EmbeddingClient(host="https://test.databricks.com", token="tok")
        response_body = json.dumps(
            {"data": [{"embedding": [0.1, 0.2]}, {"embedding": [0.3, 0.4]}]}
        ).encode()

        mock_resp = MagicMock()
        mock_resp.read.return_value = response_body
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = client.embed(["first text", "second text"])

        assert result == [[0.1, 0.2], [0.3, 0.4]]

    def test_url_strips_scheme_and_trailing_slash(self) -> None:
        client = EmbeddingClient(host="https://test.databricks.com/", token="tok")
        assert client._url == (
            "https://test.databricks.com/serving-endpoints/"
            "databricks-gte-large-en/invocations"
        )


class TestEmbedFromEnv:
    def test_raises_without_host_and_token(self) -> None:
        for key in ("WNV_DATABRICKS_HOST", "WNV_DATABRICKS_TOKEN"):
            os.environ.pop(key, None)
        with pytest.raises(RuntimeError, match="WNV_DATABRICKS_HOST"):
            embed_from_env()

    def test_uses_env_values(self) -> None:
        os.environ["WNV_DATABRICKS_HOST"] = "https://test.databricks.com"
        os.environ["WNV_DATABRICKS_TOKEN"] = "tok"
        try:
            client = embed_from_env()
            assert client.host == "test.databricks.com"
            assert client.endpoint == "databricks-gte-large-en"
        finally:
            os.environ.pop("WNV_DATABRICKS_HOST", None)
            os.environ.pop("WNV_DATABRICKS_TOKEN", None)
