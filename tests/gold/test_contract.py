"""M0: Gold table contract tests.

Schema contract runs locally. Integration test requires Databricks connectivity.
"""

from __future__ import annotations

import pytest

# Canonical Gold table contract — single source of truth for the agent.
GOLD_TABLE_NAME = "gold_county_month_wnv_weather"
GOLD_CATALOG = "eliao"
GOLD_SCHEMA = "wnv_demo"
GOLD_FULLY_QUALIFIED = f"{GOLD_CATALOG}.{GOLD_SCHEMA}.{GOLD_TABLE_NAME}"

# Expected columns in the Gold materialized view.
EXPECTED_COLUMNS = [
    # Keys
    "year",
    "month",
    "observation_month",
    "county",
    # Geography
    "county_fips",
    "latitude",
    "longitude",
    "land_area_2010",
    # Activity counts (per source type, NULL means not reported)
    "bird_count",
    "horse_count",
    "mosquito_count",
    # Reporting flags
    "bird_reported",
    "horse_reported",
    "mosquito_reported",
    "reported_activity_types",
    "total_reported_non_human_activity",
    # Ecology
    "avian_phylodiversity",
    # Weather (one-month shifted)
    "wind_u_1m_shift",
    "wind_v_1m_shift",
    "wind_speed_1m_shift",
    "temperature_k_1m_shift",
    "temperature_c_1m_shift",
    "precipitation_m_1m_shift",
    "precipitation_mm_1m_shift",
    "snowfall_1m_shift",
    "surface_runoff_1m_shift",
    "high_vegetation_index_1m_shift",
    "low_vegetation_index_1m_shift",
    # Consistency checks
    "fips_consistent",
    "coordinates_consistent",
    "weather_consistent",
    # Provenance
    "bird_source_file",
    "horse_source_file",
    "mosquito_source_file",
    "latest_ingested_at",
]

# Known validated statistics from pipeline deployment.
EXPECTED_ROW_COUNT = 2_261
EXPECTED_MIN_YEAR = 2002
EXPECTED_MAX_YEAR = 2022
EXPECTED_COUNTY_COUNT = 101


class TestGoldSchemaContract:
    """Local tests — validate the contract definition itself is well-formed."""

    def test_no_duplicate_columns(self) -> None:
        assert len(EXPECTED_COLUMNS) == len(set(EXPECTED_COLUMNS))

    def test_required_keys_present(self) -> None:
        for col in ("year", "month", "county"):
            assert col in EXPECTED_COLUMNS, f"Missing key column: {col}"

    def test_all_activity_counts_present(self) -> None:
        for source in ("bird", "horse", "mosquito"):
            assert f"{source}_count" in EXPECTED_COLUMNS
            assert f"{source}_reported" in EXPECTED_COLUMNS

    def test_consistency_columns_present(self) -> None:
        for check in ("fips_consistent", "coordinates_consistent", "weather_consistent"):
            assert check in EXPECTED_COLUMNS

    def test_row_count_is_positive(self) -> None:
        assert EXPECTED_ROW_COUNT > 0

    def test_year_range_is_valid(self) -> None:
        assert EXPECTED_MIN_YEAR < EXPECTED_MAX_YEAR


@pytest.mark.integration
class TestGoldTableIntegration:
    """Requires Databricks connectivity. Gate: WNV_DATABRICKS_HOST + WNV_DATABRICKS_TOKEN.

    Uses databricks-sql-connector (install: pip install databricks-sql-connector).
    """

    @pytest.fixture(autouse=True)
    def _skip_without_databricks(self) -> None:
        import os

        host = os.environ.get("WNV_DATABRICKS_HOST")
        token = os.environ.get("WNV_DATABRICKS_TOKEN")
        if not host or not token:
            pytest.skip(
                "set WNV_DATABRICKS_HOST and WNV_DATABRICKS_TOKEN for integration tests"
            )

    @pytest.fixture
    def cursor(self):
        """Databricks SQL cursor for the test session."""
        import os
        from databricks.sql import connect

        host = os.environ["WNV_DATABRICKS_HOST"].removeprefix("https://")
        token = os.environ["WNV_DATABRICKS_TOKEN"]
        catalog = os.environ.get("WNV_DATABRICKS_CATALOG", GOLD_CATALOG)
        schema = os.environ.get("WNV_DATABRICKS_SCHEMA", GOLD_SCHEMA)
        warehouse_id = os.environ.get("WNV_DATABRICKS_WAREHOUSE_ID")

        if not warehouse_id:
            pytest.skip("set WNV_DATABRICKS_WAREHOUSE_ID for SQL integration tests")

        conn = connect(
            server_hostname=host,
            http_path=f"/sql/1.0/warehouses/{warehouse_id}",
            access_token=token,
            catalog=catalog,
            schema=schema,
        )
        cur = conn.cursor()
        yield cur
        cur.close()
        conn.close()

    def test_gold_table_row_count(self, cursor) -> None:
        """Gold table has the expected number of county-month rows."""
        cursor.execute(f"SELECT COUNT(*) AS cnt FROM {GOLD_FULLY_QUALIFIED}")
        row = cursor.fetchone()
        actual = row[0]
        assert actual == EXPECTED_ROW_COUNT, (
            f"Expected {EXPECTED_ROW_COUNT} rows, got {actual}. "
            "Re-run the pipeline if the source data changed."
        )

    def test_gold_table_unique_keys(self, cursor) -> None:
        """Every county/year/month combination appears exactly once."""
        cursor.execute(
            f"SELECT year, month, county, COUNT(*) AS cnt "
            f"FROM {GOLD_FULLY_QUALIFIED} "
            f"GROUP BY year, month, county HAVING cnt > 1"
        )
        duplicates = len(cursor.fetchall())
        assert duplicates == 0, f"Found {duplicates} duplicate county-month keys"

    def test_gold_table_year_range(self, cursor) -> None:
        """Year values fall within the expected range."""
        cursor.execute(
            f"SELECT MIN(year), MAX(year) FROM {GOLD_FULLY_QUALIFIED}"
        )
        row = cursor.fetchone()
        min_year, max_year = row[0], row[1]
        assert min_year == EXPECTED_MIN_YEAR, f"Expected min year {EXPECTED_MIN_YEAR}, got {min_year}"
        assert max_year == EXPECTED_MAX_YEAR, f"Expected max year {EXPECTED_MAX_YEAR}, got {max_year}"

    def test_gold_table_county_count(self, cursor) -> None:
        """Distinct counties match the expected count."""
        cursor.execute(
            f"SELECT COUNT(DISTINCT county) FROM {GOLD_FULLY_QUALIFIED}"
        )
        row = cursor.fetchone()
        actual = row[0]
        assert actual == EXPECTED_COUNTY_COUNT, (
            f"Expected {EXPECTED_COUNTY_COUNT} counties, got {actual}"
        )

    def test_gold_table_no_null_keys(self, cursor) -> None:
        """year, month, and county are never NULL."""
        cursor.execute(
            f"SELECT COUNT(*) FROM {GOLD_FULLY_QUALIFIED} "
            f"WHERE year IS NULL OR month IS NULL OR county IS NULL"
        )
        row = cursor.fetchone()
        nulls = row[0]
        assert nulls == 0, f"Found {nulls} rows with NULL keys"

    def test_gold_table_consistency_checks_pass(self, cursor) -> None:
        """All rows pass FIPS, coordinate, and weather consistency checks."""
        cursor.execute(
            f"SELECT COUNT(*) FROM {GOLD_FULLY_QUALIFIED} "
            f"WHERE fips_consistent = false OR coordinates_consistent = false "
            f"OR weather_consistent = false"
        )
        row = cursor.fetchone()
        failures = row[0]
        assert failures == 0, f"Found {failures} rows with consistency failures"

    def test_gold_table_columns_match_contract(self, cursor) -> None:
        """Every expected column exists in the live table."""
        cursor.execute(f"DESCRIBE EXTENDED {GOLD_FULLY_QUALIFIED}")
        actual_columns = {row[0] for row in cursor.fetchall()}
        missing = set(EXPECTED_COLUMNS) - actual_columns
        assert not missing, f"Missing columns in Gold table: {sorted(missing)}"
