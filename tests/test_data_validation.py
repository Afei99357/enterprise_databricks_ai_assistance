from __future__ import annotations

import csv
from pathlib import Path

import pytest

from wnv_assistant.data_validation import (
    CONSISTENCY_COLUMNS,
    DataValidationError,
    validate_all,
    validate_source,
)

FIXTURE_DIRECTORY = Path(__file__).parent / "fixtures" / "non_human"


def _write_source(
    path: Path,
    *,
    outcome_column: str = "Bird",
    conflicting: bool = False,
) -> None:
    columns = [
        "Year",
        "Month",
        "County",
        outcome_column,
        "Date",
        *CONSISTENCY_COLUMNS,
    ]
    base = {
        "Year": "2002",
        "Month": "8",
        "County": "adams",
        outcome_column: "2",
        "Date": "2002-08-01",
        "FIPS": "17001",
        "County_Seat_Longitude": "-91.379304",
        "County_Seat_Latitude": "39.933675",
        "u10_1m_shift": "0.1",
        "v10_1m_shift": "0.2",
        "t2m_1m_shift": "300.0",
        "lai_hv_1m_shift": "2.3",
        "lai_lv_1m_shift": "2.9",
        "src_1m_shift": "0.01",
        "sf_1m_shift": "0.0",
        "sro_1m_shift": "0.001",
        "tp_1m_shift": "0.002",
    }
    duplicate = dict(base)
    if conflicting:
        duplicate[outcome_column] = "3"

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows([base, duplicate])


def test_consistent_duplicates_are_accepted(tmp_path: Path) -> None:
    source = tmp_path / "bird.csv"
    _write_source(source)

    report = validate_source(source, "bird", "Bird")

    assert report.raw_rows == 2
    assert report.unique_county_months == 1
    assert report.duplicate_rows == 1
    assert report.conflict_groups == 0
    assert report.incomplete_metadata_rows == 0


def test_conflicting_duplicates_are_rejected(tmp_path: Path) -> None:
    source = tmp_path / "bird.csv"
    _write_source(source, conflicting=True)

    with pytest.raises(DataValidationError, match="conflicting county-month"):
        validate_source(source, "bird", "Bird")


def test_negative_outcome_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "bird.csv"
    _write_source(source)
    content = source.read_text(encoding="utf-8").replace(",2,", ",-2,")
    source.write_text(content, encoding="utf-8")

    with pytest.raises(DataValidationError, match="negative"):
        validate_source(source, "bird", "Bird")


def test_synthetic_source_set_satisfies_data_contract() -> None:
    reports = validate_all(FIXTURE_DIRECTORY)

    assert [report.source for report in reports] == ["bird", "horse", "mosquito"]
    assert all(report.raw_rows == 2 for report in reports)
    assert all(report.unique_county_months == 1 for report in reports)
    assert all(report.duplicate_rows == 1 for report in reports)
    assert all(report.conflict_groups == 0 for report in reports)
