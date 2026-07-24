"""Validate the Illinois county-level non-human WNV source files."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SOURCE_FILES = {
    "bird": ("bird_illinois_county_02_to_22.csv", "Bird"),
    "horse": ("horse_illinois_county_02_to_22.csv", "Horse"),
    "mosquito": ("mos_illinois_county_02_to_22.csv", "Mosquito"),
}

KEY_COLUMNS = ("Year", "Month", "County")
REQUIRED_SHARED_COLUMNS = (
    "Year",
    "Month",
    "County",
    "FIPS",
    "County_Seat_Longitude",
    "County_Seat_Latitude",
    "Date",
    "u10_1m_shift",
    "v10_1m_shift",
    "t2m_1m_shift",
    "tp_1m_shift",
)
CONSISTENCY_COLUMNS = (
    "FIPS",
    "County_Seat_Longitude",
    "County_Seat_Latitude",
    "u10_1m_shift",
    "v10_1m_shift",
    "t2m_1m_shift",
    "lai_hv_1m_shift",
    "lai_lv_1m_shift",
    "src_1m_shift",
    "sf_1m_shift",
    "sro_1m_shift",
    "tp_1m_shift",
)


class DataValidationError(ValueError):
    """Raised when a source file violates the expected data contract."""


@dataclass(frozen=True)
class ValidationReport:
    """Summary of one validated source file."""

    source: str
    path: Path
    raw_rows: int
    unique_county_months: int
    duplicate_rows: int
    conflict_groups: int
    incomplete_metadata_rows: int
    minimum_year: int
    maximum_year: int
    county_count: int


def _required_value(
    row: dict[str, str], column: str, row_number: int, path: Path
) -> str:
    value = (row.get(column) or "").strip()
    if not value:
        raise DataValidationError(
            f"{path}: row {row_number} has an empty required field {column!r}"
        )
    return value


def _number(
    row: dict[str, str],
    column: str,
    row_number: int,
    path: Path,
    *,
    nonnegative: bool = False,
) -> float:
    value = _required_value(row, column, row_number, path)
    try:
        number = float(value)
    except ValueError as exc:
        raise DataValidationError(
            f"{path}: row {row_number} has nonnumeric {column!r}: {value!r}"
        ) from exc
    if not math.isfinite(number):
        raise DataValidationError(
            f"{path}: row {row_number} has non-finite {column!r}: {value!r}"
        )
    if nonnegative and number < 0:
        raise DataValidationError(
            f"{path}: row {row_number} has negative {column!r}: {value!r}"
        )
    return number


def _optional_number(
    row: dict[str, str], column: str, row_number: int, path: Path
) -> float | None:
    value = (row.get(column) or "").strip()
    if not value:
        return None
    try:
        number = float(value)
    except ValueError as exc:
        raise DataValidationError(
            f"{path}: row {row_number} has nonnumeric {column!r}: {value!r}"
        ) from exc
    if not math.isfinite(number):
        raise DataValidationError(
            f"{path}: row {row_number} has non-finite {column!r}: {value!r}"
        )
    return number


def validate_source(path: Path, source: str, outcome_column: str) -> ValidationReport:
    """Validate one CSV and ensure duplicate logical records are consistent."""

    if not path.is_file():
        raise DataValidationError(f"Source file does not exist: {path}")

    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        required = set(REQUIRED_SHARED_COLUMNS) | {outcome_column}
        missing = sorted(required - columns)
        if missing:
            raise DataValidationError(
                f"{path}: missing required columns: {', '.join(missing)}"
            )

        rows: list[dict[str, str]] = []
        normalized: list[dict[str, Any]] = []
        incomplete_metadata_rows = 0
        for row_number, row in enumerate(reader, start=2):
            year = int(_number(row, "Year", row_number, path))
            month = int(_number(row, "Month", row_number, path))
            county = _required_value(row, "County", row_number, path).casefold()

            if not 1900 <= year <= 2100:
                raise DataValidationError(
                    f"{path}: row {row_number} has invalid year: {year}"
                )
            if not 1 <= month <= 12:
                raise DataValidationError(
                    f"{path}: row {row_number} has invalid month: {month}"
                )

            outcome = _number(
                row, outcome_column, row_number, path, nonnegative=True
            )
            values = {
                column: _optional_number(row, column, row_number, path)
                for column in CONSISTENCY_COLUMNS
            }
            if any(value is None for value in values.values()):
                incomplete_metadata_rows += 1
            rows.append(row)
            normalized.append(
                {
                    "key": (year, month, county),
                    "outcome": outcome,
                    "consistent_values": values,
                }
            )

    if not rows:
        raise DataValidationError(f"{path}: source file contains no data rows")

    groups: dict[tuple[int, int, str], list[dict[str, Any]]] = {}
    for row in normalized:
        groups.setdefault(row["key"], []).append(row)

    conflicts: list[str] = []
    for key, group in groups.items():
        expected = (group[0]["outcome"], group[0]["consistent_values"])
        if any(
            (item["outcome"], item["consistent_values"]) != expected
            for item in group[1:]
        ):
            conflicts.append(f"{key[2]} {key[0]}-{key[1]:02d}")

    if conflicts:
        examples = ", ".join(conflicts[:5])
        raise DataValidationError(
            f"{path}: found {len(conflicts)} conflicting county-month groups; "
            f"examples: {examples}"
        )

    years = [item["key"][0] for item in normalized]
    counties = {item["key"][2] for item in normalized}
    return ValidationReport(
        source=source,
        path=path,
        raw_rows=len(rows),
        unique_county_months=len(groups),
        duplicate_rows=len(rows) - len(groups),
        conflict_groups=0,
        incomplete_metadata_rows=incomplete_metadata_rows,
        minimum_year=min(years),
        maximum_year=max(years),
        county_count=len(counties),
    )


def validate_all(data_directory: Path) -> list[ValidationReport]:
    """Validate every configured source in the given directory."""

    reports = []
    for source, (filename, outcome_column) in SOURCE_FILES.items():
        reports.append(
            validate_source(data_directory / filename, source, outcome_column)
        )
    return reports


def format_reports(reports: list[ValidationReport]) -> str:
    """Render reports as a compact console table."""

    header = (
        f"{'Source':<10} {'Raw rows':>10} {'County-months':>14} "
        f"{'Duplicate rows':>15} {'Conflicts':>10} {'Incomplete':>11} "
        f"{'Years':>11} {'Counties':>9}"
    )
    divider = "-" * len(header)
    lines = [header, divider]
    for report in reports:
        lines.append(
            f"{report.source:<10} {report.raw_rows:>10,} "
            f"{report.unique_county_months:>14,} "
            f"{report.duplicate_rows:>15,} {report.conflict_groups:>10} "
            f"{report.incomplete_metadata_rows:>11,} "
            f"{report.minimum_year}-{report.maximum_year:<6} "
            f"{report.county_count:>9}"
        )
    return "\n".join(lines)


def main() -> int:
    """Command-line entry point."""

    parser = argparse.ArgumentParser(
        description="Validate Illinois county-level non-human WNV CSV files."
    )
    parser.add_argument(
        "--data-directory",
        type=Path,
        default=Path("data/non_human"),
        help="Directory containing the three source CSV files.",
    )
    args = parser.parse_args()

    try:
        reports = validate_all(args.data_directory)
    except DataValidationError as exc:
        parser.exit(1, f"Validation failed: {exc}\n")

    print(format_reports(reports))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
