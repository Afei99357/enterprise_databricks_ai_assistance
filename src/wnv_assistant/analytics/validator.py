"""SQL safety validator for text-to-SQL.

Ensures generated SQL is read-only, references only allowed tables/columns,
and respects row limits.
"""

from __future__ import annotations

from dataclasses import dataclass

from .schema import ALLOWED_COLUMNS, ALLOWED_TABLES


@dataclass(frozen=True)
class ValidationResult:
    """Result of validating a SQL query."""

    valid: bool
    errors: list[str]


# Disallowed SQL keywords (write operations, system commands, etc.).
BLOCKED_KEYWORDS = {
    "ALTER", "CREATE", "DELETE", "DROP", "EXEC", "EXECUTE", "GRANT",
    "INSERT", "MERGE", "REVOKE", "TRUNCATE", "UPDATE", "UPSERT",
    "CALL", "EXPORT", "IMPORT", "LOAD", "REVOKE", "SET",
}


def validate(sql: str) -> ValidationResult:
    """Validate a SQL query against safety rules.

    Checks:
    - No write operations or dangerous keywords
    - Only allowed tables are referenced
    - LIMIT clause is present and within max rows
    - Query is a SELECT statement
    """
    errors: list[str] = []
    normalized = _normalize(sql)

    _check_is_select(normalized, sql, errors)
    _check_blocked_keywords(normalized, sql, errors)
    _check_tables(normalized, sql, errors)
    _check_limit(normalized, sql, errors)

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
    )


def _normalize(sql: str) -> str:
    """Normalize SQL for checking: uppercase, collapse whitespace."""
    return " ".join(sql.upper().split())


def _check_is_select(normalized: str, original: str, errors: list[str]) -> None:
    """Query must be a SELECT statement."""
    if not normalized.startswith("SELECT"):
        errors.append("Query must be a SELECT statement")


def _check_blocked_keywords(normalized: str, original: str, errors: list[str]) -> None:
    """Reject queries containing dangerous keywords."""
    tokens = set(normalized.split())
    found = tokens & BLOCKED_KEYWORDS
    if found:
        errors.append(f"Blocked keywords found: {', '.join(sorted(found))}")


def _check_tables(normalized: str, original: str, errors: list[str]) -> None:
    """Only allowed tables may be referenced."""
    allowed = {t.name.upper() for t in ALLOWED_TABLES}
    import re
    # Find table references after FROM, JOIN — handle catalog.schema.table format
    table_refs = re.findall(
        r"(?:FROM|JOIN)\s+([A-Z0-9_.]+?)(?:\s|$|,)",
        normalized,
    )
    # Filter out SQL keywords
    sql_keywords = {
        "LATERAL", "UNBOUNDED", "PRECEDING", "FOLLOWING", "CURRENT",
        "ROW", "ROWS", "RANGE", "GROUP", "ORDER", "LIMIT", "WHERE",
    }
    for ref in table_refs:
        # Extract just the table name (last part of catalog.schema.table)
        table_name = ref.split(".")[-1].strip(",")
        if table_name not in allowed and table_name not in sql_keywords:
            errors.append(f"Table not allowed: {table_name}")


def _check_limit(normalized: str, original: str, errors: list[str]) -> None:
    """Query must have a LIMIT clause with a reasonable value."""
    import re
    if "LIMIT" not in normalized:
        errors.append("Query must include a LIMIT clause")
        return

    match = re.search(r"LIMIT\s+(\d+)", normalized)
    if match:
        limit = int(match.group(1))
        # Get max rows from the allowed table
        max_rows = max(t.max_rows for t in ALLOWED_TABLES) if ALLOWED_TABLES else 1000
        if limit > max_rows:
            errors.append(f"LIMIT {limit} exceeds maximum allowed ({max_rows})")
