"""MLflow PyFunc wrapper for the WNV assistant agent.

Packages the orchestrator + analytics tool + LLM client as a deployable
MLflow model for Databricks Model Serving.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from mlflow.pyfunc import PythonModel


class WNVAssistantModel(PythonModel):
    """MLflow model that serves the WNV assistant agent."""

    def load_context(self, context) -> None:
        """Initialize the agent components on model load."""
        import os

        from wnv_assistant.agent.orchestrator import Orchestrator
        from wnv_assistant.analytics.executor import executor_from_env
        from wnv_assistant.analytics.text_to_sql import TextToSQLTool
        from wnv_assistant.llm.databricks_client import DatabricksLLMClient

        # Get config from model artifacts or environment
        host = os.environ.get("WNV_DATABRICKS_HOST", "")
        token = os.environ.get("WNV_DATABRICKS_TOKEN", "")
        warehouse_id = os.environ.get("WNV_DATABRICKS_WAREHOUSE_ID", "")
        llm_endpoint = os.environ.get("WNV_LLM_ENDPOINT", "databricks-gpt-oss-20b")

        # Try to load config from artifacts
        config_path = context.artifacts.get("config")
        if config_path:
            with open(config_path) as f:
                config = json.load(f)
            host = host or config.get("databricks_host")
            token = token or config.get("databricks_token")
            warehouse_id = warehouse_id or config.get("databricks_warehouse_id")
            llm_endpoint = llm_endpoint or config.get("llm_endpoint")

        # Initialize LLM client
        self.llm = DatabricksLLMClient(
            host=host,
            token=token,
            endpoint=llm_endpoint,
        )

        # Initialize analytics tool
        self.analytics_tool = TextToSQLTool(
            executor=executor_from_env(),
            llm_generate=lambda q, sys: self.llm.generate_sql(q, sys),
        )

        # Initialize orchestrator
        self.orchestrator = Orchestrator(
            analytics_tool=self.analytics_tool,
            llm_classify=lambda q: self.llm.classify_route(q),
            llm_answer=lambda q, tr: self.llm.synthesize_answer(q, tr.answer, tr.data),
        )

    def predict(
        self,
        context,
        model_input,
        params: dict | None = None,
    ) -> dict:
        """Handle a prediction request.

        Accepts:
        - dict with "question" key
        - dict matching AgentRequest fields
        - pandas DataFrame with "question" column
        """
        # Parse input
        if isinstance(model_input, dict):
            question = model_input.get("question", "")
            conversation_id = model_input.get("conversation_id", "")
            user_id = model_input.get("user_id", "")
        elif hasattr(model_input, "to_dict"):
            # pandas DataFrame
            records = model_input.to_dict(orient="records")
            return [
                self._handle_request(
                    r.get("question", ""),
                    r.get("conversation_id", ""),
                    r.get("user_id", ""),
                )
                for r in records
            ]
        else:
            question = str(model_input)
            conversation_id = ""
            user_id = ""

        return self._handle_request(question, conversation_id, user_id)

    def _handle_request(
        self, question: str, conversation_id: str = "", user_id: str = ""
    ) -> dict:
        """Process a single request."""
        from wnv_assistant.agent.models import AgentRequest

        if not question.strip():
            return {
                "answer": "Please provide a question about WNV surveillance data.",
                "route": "OUT_OF_SCOPE",
                "tool_results": [],
                "warnings": ["Empty question"],
                "insufficient_evidence": True,
            }

        request = AgentRequest(
            question=question,
            conversation_id=conversation_id,
            user_id=user_id,
        )

        response = self.orchestrator.run(request)
        return response.to_dict()


def log_model(
    model_dir: str,
    *,
    config: dict | None = None,
    artifact_path: str = "wnv-assistant",
) -> str:
    """Log the WNV assistant as an MLflow model.

    Args:
        model_dir: Directory to store temporary model artifacts.
        config: Optional config dict (host, warehouse_id, llm_endpoint).
        artifact_path: Path within the MLflow run for the model.

    Returns:
        Runs URI of the logged MLflow model.
    """
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    # Save config if provided
    if config:
        config_path = Path(model_dir) / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f)
        artifacts = {"config": str(config_path)}
    else:
        artifacts = {}

    # Unity Catalog requires a signature. Use DataFrame examples because that is
    # the format Databricks Model Serving sends to a PyFunc model.
    input_example = pd.DataFrame(
        [
            {
                "question": "Show top 5 counties by mosquito activity in 2022",
                "conversation_id": "",
                "user_id": "",
            }
        ]
    )
    output_example = pd.DataFrame(
        [
            {
                "answer": "Example response",
                "route": "ANALYTICS",
                "tool_results": [],
                "warnings": [],
                "insufficient_evidence": False,
            }
        ]
    )

    model_info = mlflow.pyfunc.log_model(
        artifact_path=artifact_path,
        python_model=WNVAssistantModel(),
        artifacts=artifacts,
        # Bundle the assistant package with the model. Model Serving adds this
        # directory to Python's import path when it loads the registered model.
        code_paths=[str(Path(__file__).parent)],
        signature=infer_signature(input_example, output_example),
        input_example=input_example,
        pip_requirements=[
            "databricks-sql-connector>=4.0.0",
            "mlflow>=2.20.0",
            "pandas>=1.5.0",
        ],
    )
    return model_info.model_uri


def load_model(model_uri: str):
    """Load the WNV assistant from an MLflow model URI."""
    return mlflow.pyfunc.load_model(model_uri)
