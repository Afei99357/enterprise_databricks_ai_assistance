"""Analytics eval suite.

Runs representative questions (tests/evals/questions.yml) through the real
orchestrator with a real LLM and checks routing accuracy and SQL shape.

Scope: this checks routing + whether the generated SQL references the
expected table/columns. It does not score final-answer text correctness or
safety automatically — every case prints the full answer so a human can
spot-check those dimensions. A stronger automated check (e.g. an LLM judge)
is future work, not part of this suite.

Run: uv run pytest tests/evals/run_eval.py -v -m eval
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from wnv_assistant.agent.models import AgentRequest
from wnv_assistant.agent.orchestrator import Orchestrator, synthesize_answer
from wnv_assistant.agent.routing import classify_route
from wnv_assistant.analytics.executor import executor_from_env
from wnv_assistant.analytics.text_to_sql import TextToSQLTool, generate_sql
from wnv_assistant.llm.databricks_client import client_from_env

QUESTIONS_PATH = Path(__file__).parent / "questions.yml"


def _load_questions() -> list[dict]:
    with open(QUESTIONS_PATH) as f:
        return yaml.safe_load(f)


def _requires_databricks() -> bool:
    return bool(
        os.environ.get("WNV_DATABRICKS_HOST")
        and os.environ.get("WNV_DATABRICKS_TOKEN")
        and os.environ.get("WNV_LLM_ENDPOINT")
    )


@pytest.fixture(scope="module")
def orchestrator() -> Orchestrator:
    if not _requires_databricks():
        pytest.skip("set WNV_DATABRICKS_* and WNV_LLM_ENDPOINT to run the eval suite")

    llm = client_from_env(endpoint=os.environ["WNV_LLM_ENDPOINT"])
    analytics_tool = TextToSQLTool(
        executor=executor_from_env(),
        llm_generate=lambda q, sys: generate_sql(llm, q, sys),
    )
    return Orchestrator(
        analytics_tool=analytics_tool,
        llm_classify=lambda q, ctx="": classify_route(llm, q, ctx),
        llm_answer=lambda q, tr: synthesize_answer(llm, q, tr.answer, tr.data),
    )


@pytest.mark.eval
@pytest.mark.parametrize("case", _load_questions(), ids=lambda c: c["question"][:50])
def test_eval_question(case: dict, orchestrator: Orchestrator) -> None:
    response = orchestrator.run(AgentRequest(question=case["question"]))

    print(f"\nQ: {case['question']}")
    print(f"Route: {response.route} (expected {case['expected_route']})")
    print(f"Answer: {response.answer}")

    assert response.route == case["expected_route"], (
        f"routing mismatch for {case['question']!r}: "
        f"got {response.route}, expected {case['expected_route']}"
    )

    if case["expected_route"] in ("ANALYTICS", "MIXED") and response.tool_results:
        sql = response.tool_results[0].generated_sql.lower()
        print(f"SQL: {sql}")
        for keyword in case["expected_sql_keywords"]:
            assert keyword.lower() in sql, (
                f"expected {keyword!r} in generated SQL for "
                f"{case['question']!r}, got: {sql}"
            )
        assert response.tool_results[0].success, (
            f"tool call failed for {case['question']!r}: "
            f"{response.tool_results[0].error}"
        )
