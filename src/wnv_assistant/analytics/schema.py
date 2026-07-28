"""Semantic layer for the Gold WNV table.

This is the single source of truth for what the text-to-SQL tool is allowed
to query. Every column gets a plain-English description that the LLM uses
to generate correct SQL.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ColumnDescription:
    """Human-readable description of a column for the LLM prompt."""

    name: str
    type: str
    description: str


@dataclass(frozen=True)
class TableDescription:
    """Allowed table with column descriptions and query constraints."""

    name: str
    description: str
    columns: list[ColumnDescription]
    max_rows: int = 10_000
    required_filters: list[str] = field(default_factory=list)


# Canonical Gold table — the only table the tool may query.
GOLD_TABLE = TableDescription(
    name="gold_county_month_wnv_weather",
    description=(
        "One row per Illinois county per month (2002-2022). Contains non-human "
        "West Nile virus surveillance counts (bird, horse, mosquito) and "
        "one-month-shifted weather variables. NULL count means not reported, "
        "not zero."
    ),
    columns=[
        ColumnDescription("year", "INT", "Observation year (2002-2022)"),
        ColumnDescription("month", "INT", "Observation month (1-12)"),
        ColumnDescription(
            "observation_month", "DATE", "First day of the observation month"
        ),
        ColumnDescription("county", "STRING", "Illinois county name (lowercase)"),
        ColumnDescription("county_fips", "STRING", "5-digit county FIPS code"),
        ColumnDescription(
            "bird_count", "INT",
            "Number of bird WNV observations. NULL = not reported for this county-month.",
        ),
        ColumnDescription(
            "horse_count", "INT",
            "Number of horse WNV observations. NULL = not reported.",
        ),
        ColumnDescription(
            "mosquito_count", "INT",
            "Number of mosquito pools testing positive for WNV. NULL = not reported.",
        ),
        ColumnDescription(
            "bird_reported", "BOOLEAN", "True if bird data was reported for this county-month"
        ),
        ColumnDescription(
            "horse_reported", "BOOLEAN", "True if horse data was reported"
        ),
        ColumnDescription(
            "mosquito_reported", "BOOLEAN", "True if mosquito data was reported"
        ),
        ColumnDescription(
            "reported_activity_types", "INT",
            "Count of source types (bird/horse/mosquito) that reported data (0-3)",
        ),
        ColumnDescription(
            "total_reported_non_human_activity", "INT",
            "Sum of bird + horse + mosquito counts (treats NULL as 0)",
        ),
        ColumnDescription("latitude", "DOUBLE", "County seat latitude"),
        ColumnDescription("longitude", "DOUBLE", "County seat longitude"),
        ColumnDescription("land_area_2010", "DOUBLE", "Land area in square kilometers (2010 census)"),
        ColumnDescription(
            "avian_phylodiversity", "DOUBLE",
            "Measure of bird species diversity in the county",
        ),
        ColumnDescription(
            "temperature_c_1m_shift", "DOUBLE",
            "Average temperature in Celsius, shifted by one month",
        ),
        ColumnDescription(
            "precipitation_mm_1m_shift", "DOUBLE",
            "Total precipitation in mm, shifted by one month",
        ),
        ColumnDescription(
            "wind_speed_1m_shift", "DOUBLE",
            "Wind speed magnitude in m/s, shifted by one month",
        ),
    ],
    max_rows=5000,
)

# List of all allowed tables. Currently just one.
ALLOWED_TABLES: list[TableDescription] = [GOLD_TABLE]

# Build lookup: table name -> set of allowed column names.
ALLOWED_COLUMNS: dict[str, set[str]] = {
    t.name: {c.name for c in t.columns}
    for t in ALLOWED_TABLES
}

# Build prompt-friendly schema text for the LLM.
def format_schema_for_prompt() -> str:
    """Return a compact schema description for the LLM system prompt."""
    parts: list[str] = []
    for table in ALLOWED_TABLES:
        parts.append(f"Table: {table.name}")
        parts.append(f"  Description: {table.description}")
        parts.append("  Columns:")
        for col in table.columns:
            parts.append(f"    - {col.name} ({col.type}): {col.description}")
        parts.append(f"  Max rows: {table.max_rows}")
        parts.append("")
    return "\n".join(parts)
