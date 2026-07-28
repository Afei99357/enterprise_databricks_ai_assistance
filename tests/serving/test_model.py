"""Tests for MLflow serving wrapper."""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="MLflow model logging requires full environment setup")
class TestMLflowModelLogging:
    """Tests for logging the model to MLflow."""

    def test_log_model_creates_artifacts(self, tmp_path) -> None:
        """log_model creates model artifacts in the specified directory."""
        from wnv_assistant.serving import log_model

        model_uri = log_model(
            str(tmp_path),
            config={"llm_endpoint": "test-endpoint"},
        )
        assert model_uri


class TestWNVAssistantModelInputParsing:
    """Test input parsing without MLflow (unit tests)."""

    def test_empty_question_returns_safe_response(self) -> None:
        """Empty question returns a helpful message."""
        from wnv_assistant.serving import WNVAssistantModel

        model = WNVAssistantModel()
        model.artifacts = {}

        result = model._handle_request("")
        assert result["route"] == "OUT_OF_SCOPE"
        assert result["insufficient_evidence"]
        assert "Please provide a question" in result["answer"]

    def test_question_returns_dict_with_expected_keys(self) -> None:
        """Response dict has all expected keys."""
        from wnv_assistant.agent.orchestrator import Orchestrator
        from wnv_assistant.serving import WNVAssistantModel

        model = WNVAssistantModel()
        model.artifacts = {}
        # Use keyword routing instead of LLM
        model.orchestrator = Orchestrator(
            llm_classify=lambda q: ("ANALYTICS", "test"),
        )

        # This will fail on tool execution (no analytics tool), but tests the routing
        result = model._handle_request("How many cases?")
        assert "answer" in result
        assert "route" in result
        assert "tool_results" in result
        assert "warnings" in result

    def test_predict_with_dict_input(self) -> None:
        """predict handles dict input with question key."""
        from unittest.mock import MagicMock
        from wnv_assistant.serving import WNVAssistantModel

        model = WNVAssistantModel()
        model.artifacts = {}
        model.orchestrator = MagicMock()
        model.orchestrator.run.return_value.to_dict.return_value = {
            "answer": "42 cases",
            "route": "ANALYTICS",
            "tool_results": [],
            "warnings": [],
        }

        result = model.predict(None, {"question": "How many cases?"})
        assert result["answer"] == "42 cases"
        model.orchestrator.run.assert_called_once()

    def test_predict_with_dataframe_input(self) -> None:
        """predict handles pandas DataFrame input."""
        import pandas as pd
        from unittest.mock import MagicMock
        from wnv_assistant.serving import WNVAssistantModel

        model = WNVAssistantModel()
        model.artifacts = {}
        model.orchestrator = MagicMock()
        model.orchestrator.run.return_value.to_dict.return_value = {
            "answer": "42 cases",
            "route": "ANALYTICS",
            "tool_results": [],
            "warnings": [],
        }

        df = pd.DataFrame([{"question": "How many?"}, {"question": "Which county?"}])
        results = model.predict(None, df)

        assert len(results) == 2
        assert model.orchestrator.run.call_count == 2


@pytest.mark.integration
class TestServingIntegration:
    """Requires Databricks connectivity and LLM endpoint."""

    @pytest.fixture(autouse=True)
    def _skip_without_databricks(self) -> None:
        import os

        if not all(os.environ.get(k) for k in (
            "WNV_DATABRICKS_HOST", "WNV_DATABRICKS_TOKEN",
            "WNV_DATABRICKS_WAREHOUSE_ID", "WNV_LLM_ENDPOINT",
        )):
            pytest.skip("set WNV_DATABRICKS_* and WNV_LLM_ENDPOINT for serving integration tests")

    def test_end_to_end_prediction(self, tmp_path) -> None:
        """Full pipeline: log model -> load -> predict."""
        import os
        from wnv_assistant.serving import log_model, load_model

        model_uri = log_model(
            str(tmp_path),
            config={
                "databricks_host": os.environ["WNV_DATABRICKS_HOST"],
                "databricks_token": os.environ["WNV_DATABRICKS_TOKEN"],
                "databricks_warehouse_id": os.environ["WNV_DATABRICKS_WAREHOUSE_ID"],
                "llm_endpoint": os.environ["WNV_LLM_ENDPOINT"],
            },
        )

        model = load_model(model_uri)
        result = model.predict({"question": "Show mosquito activity in Boone County 2022"})

        assert "answer" in result
        assert result["route"] in ("ANALYTICS", "DOCUMENT", "OUT_OF_SCOPE")
