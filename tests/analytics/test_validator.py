"""Tests for SQL validator."""

from __future__ import annotations

from wnv_assistant.analytics.validator import validate


class TestSafeQueries:
    """Valid queries should pass."""

    def test_simple_select(self) -> None:
        result = validate("SELECT county, mosquito_count FROM gold_county_month_wnv_weather LIMIT 10")
        assert result.valid, f"Should be valid: {result.errors}"

    def test_select_with_where(self) -> None:
        sql = (
            "SELECT county, bird_count, horse_count "
            "FROM gold_county_month_wnv_weather "
            "WHERE year = 2022 AND county = 'cook' LIMIT 100"
        )
        result = validate(sql)
        assert result.valid, f"Should be valid: {result.errors}"

    def test_select_with_aggregation(self) -> None:
        sql = (
            "SELECT county, SUM(mosquito_count) AS total "
            "FROM gold_county_month_wnv_weather "
            "GROUP BY county ORDER BY total DESC LIMIT 50"
        )
        result = validate(sql)
        assert result.valid, f"Should be valid: {result.errors}"

    def test_select_with_case(self) -> None:
        sql = (
            "SELECT county, "
            "CASE WHEN mosquito_count > 0 THEN 'positive' ELSE 'negative' END AS status "
            "FROM gold_county_month_wnv_weather LIMIT 100"
        )
        result = validate(sql)
        assert result.valid, f"Should be valid: {result.errors}"

    def test_limit_at_boundary(self) -> None:
        sql = "SELECT * FROM gold_county_month_wnv_weather LIMIT 5000"
        result = validate(sql)
        assert result.valid, f"Should be valid: {result.errors}"


class TestBlockedQueries:
    """Dangerous queries should be rejected."""

    def test_drop_table(self) -> None:
        result = validate("DROP TABLE gold_county_month_wnv_weather")
        assert not result.valid
        assert any("Blocked keywords" in e for e in result.errors)

    def test_insert(self) -> None:
        result = validate("INSERT INTO gold_county_month_wnv_weather VALUES (1)")
        assert not result.valid

    def test_delete(self) -> None:
        result = validate("DELETE FROM gold_county_month_wnv_weather WHERE 1=1")
        assert not result.valid

    def test_update(self) -> None:
        result = validate("UPDATE gold_county_month_wnv_weather SET county = 'hack'")
        assert not result.valid

    def test_non_select(self) -> None:
        result = validate("CREATE TABLE hack AS SELECT * FROM gold")
        assert not result.valid
        assert any("must be a SELECT" in e for e in result.errors)

    def test_unallowed_table(self) -> None:
        result = validate("SELECT * FROM silver_bird LIMIT 10")
        assert not result.valid
        assert any("Table not allowed" in e for e in result.errors)

    def test_no_limit(self) -> None:
        result = validate("SELECT county FROM gold_county_month_wnv_weather")
        assert not result.valid
        assert any("LIMIT" in e for e in result.errors)

    def test_limit_exceeds_max(self) -> None:
        result = validate("SELECT * FROM gold_county_month_wnv_weather LIMIT 100000")
        assert not result.valid
        assert any("exceeds maximum" in e for e in result.errors)

    def test_multiple_errors(self) -> None:
        result = validate("DROP TABLE hack_table")
        assert not result.valid
        assert len(result.errors) >= 2  # blocked keyword + not SELECT + table not allowed
