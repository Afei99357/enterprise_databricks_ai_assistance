"""Conversation store interface and implementations.

Provides a Protocol for storing and retrieving conversation turns,
with an in-memory implementation for tests and a Delta-backed
implementation for production.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from datetime import datetime
from typing import Protocol

from .models import ConversationTurn


class ConversationStore(Protocol):
    """Interface for conversation persistence."""

    def get_recent_turns(
        self,
        *,
        user_id: str,
        conversation_id: str,
        limit: int,
    ) -> Sequence[ConversationTurn]: ...

    def append_turns(
        self,
        *,
        user_id: str,
        conversation_id: str,
        turns: Sequence[ConversationTurn],
    ) -> None: ...


class InMemoryStore:
    """In-memory conversation store for testing."""

    def __init__(self) -> None:
        self._turns: dict[tuple[str, str], list[ConversationTurn]] = {}

    def get_recent_turns(
        self,
        *,
        user_id: str,
        conversation_id: str,
        limit: int,
    ) -> list[ConversationTurn]:
        key = (user_id, conversation_id)
        return list(self._turns.get(key, [])[-limit:])

    def append_turns(
        self,
        *,
        user_id: str,
        conversation_id: str,
        turns: Sequence[ConversationTurn],
    ) -> None:
        key = (user_id, conversation_id)
        self._turns.setdefault(key, []).extend(turns)


class DeltaStore:
    """Delta-backed conversation store for production.

    Requires Databricks connectivity. Table must exist before use.
    """

    def __init__(
        self,
        *,
        host: str,
        token: str,
        warehouse_id: str,
        catalog: str = "eliao",
        schema: str = "wnv_demo",
        table: str = "assistant_turns",
    ) -> None:
        self._conn = _get_connection(
            host=host.removeprefix("https://"),
            token=token,
            warehouse_id=warehouse_id,
            catalog=catalog,
            schema=schema,
        )
        self._catalog = catalog
        self._schema = schema
        self._table = table
        self._fully_qualified = f"{catalog}.{schema}.{table}"

    def get_recent_turns(
        self,
        *,
        user_id: str,
        conversation_id: str,
        limit: int,
    ) -> list[ConversationTurn]:
        sql = f"""
        SELECT * FROM {self._fully_qualified}
        WHERE user_id = ? AND conversation_id = ?
        ORDER BY turn_number DESC
        LIMIT ?
        """
        rows = _execute(self._conn, sql, (user_id, conversation_id, limit))
        return list(reversed([_row_to_turn(row) for row in rows]))

    def append_turns(
        self,
        *,
        user_id: str,
        conversation_id: str,
        turns: Sequence[ConversationTurn],
    ) -> None:
        table = self._fully_qualified
        for turn in turns:
            d = turn.to_dict()
            sql = f"""
            INSERT INTO {table} (
                user_id, conversation_id, turn_id, turn_number, role, content,
                created_at, route, generated_sql, result_summary,
                analytics_state_json, citations_json
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """
            _execute(
                self._conn,
                sql,
                (
                    user_id,
                    conversation_id,
                    d["turn_id"],
                    d["turn_number"],
                    d["role"],
                    d["content"],
                    d["created_at"],
                    d.get("route"),
                    d.get("generated_sql"),
                    d.get("result_summary"),
                    d.get("analytics_state_json"),
                    d.get("citations_json"),
                ),
            )


def _row_to_turn(row: tuple) -> ConversationTurn:
    """Convert a database row to a ConversationTurn."""
    columns = [
        "user_id",
        "conversation_id",
        "turn_id",
        "turn_number",
        "role",
        "content",
        "route",
        "generated_sql",
        "result_summary",
        "analytics_state_json",
        "citations_json",
        "created_at",
    ]
    d = dict(zip(columns, row, strict=False))

    from .models import AnalyticsState, Route

    state = None
    if d.get("analytics_state_json"):
        state = AnalyticsState.from_dict(json.loads(d["analytics_state_json"]))

    citations = []
    if d.get("citations_json"):
        citations = json.loads(d["citations_json"])

    route = None
    if d.get("route"):
        route = Route(d["route"])

    return ConversationTurn(
        turn_id=d["turn_id"],
        turn_number=d["turn_number"],
        role=d["role"],
        content=d["content"],
        created_at=datetime.fromisoformat(d["created_at"])
        if isinstance(d["created_at"], str)
        else d["created_at"],
        route=route,
        generated_sql=d.get("generated_sql"),
        result_summary=d.get("result_summary"),
        analytics_state=state,
        citations=citations,
    )


def _get_connection(
    host: str, token: str, warehouse_id: str, catalog: str, schema: str
):
    """Get a Databricks SQL connection."""
    from databricks.sql import connect

    return connect(
        server_hostname=host,
        http_path=f"/sql/1.0/warehouses/{warehouse_id}",
        access_token=token,
        catalog=catalog,
        schema=schema,
    )


def _execute(conn, sql: str, params: tuple | None = None):
    """Execute SQL and return rows."""
    cur = conn.cursor()
    try:
        if params:
            cur.execute(sql, params)
        else:
            cur.execute(sql)
        if cur.description:
            return cur.fetchall()
        return []
    finally:
        cur.close()


def store_from_env() -> ConversationStore:
    """Build the configured conversation store for a serving process.

    ``memory`` is useful only for local development. Production serving uses
    the Delta store so requests routed to different replicas share history.
    """
    mode = os.environ.get("WNV_CONVERSATION_STORE", "delta").lower()
    if mode == "memory":
        return InMemoryStore()
    if mode != "delta":
        raise ValueError("WNV_CONVERSATION_STORE must be 'delta' or 'memory'")

    required = {
        "host": os.environ.get("WNV_DATABRICKS_HOST"),
        "token": os.environ.get("WNV_DATABRICKS_TOKEN"),
        "warehouse_id": os.environ.get("WNV_DATABRICKS_WAREHOUSE_ID"),
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        variables = ", ".join(
            f"WNV_DATABRICKS_{name.upper()}" for name in missing
        )
        raise RuntimeError(
            f"Delta conversation memory requires {variables}"
        )

    return DeltaStore(
        host=required["host"],
        token=required["token"],
        warehouse_id=required["warehouse_id"],
        catalog=os.environ.get("WNV_DATABRICKS_CATALOG", "eliao"),
        schema=os.environ.get("WNV_DATABRICKS_SCHEMA", "wnv_demo"),
        table=os.environ.get("WNV_CONVERSATION_TABLE", "assistant_turns"),
    )
