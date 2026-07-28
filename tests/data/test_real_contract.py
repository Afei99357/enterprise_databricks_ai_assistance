import os
from pathlib import Path

import pytest

from wnv_assistant.data_validation import validate_all


@pytest.mark.real_data
def test_collected_source_files_satisfy_data_contract() -> None:
    if os.environ.get("WNV_RUN_REAL_DATA_TESTS") != "1":
        pytest.skip("set WNV_RUN_REAL_DATA_TESTS=1 to validate local source files")

    data_directory = Path(
        os.environ.get("WNV_DATA_DIRECTORY", "data/non_human")
    )
    reports = validate_all(data_directory)

    assert [report.source for report in reports] == ["bird", "horse", "mosquito"]
    assert all(report.conflict_groups == 0 for report in reports)
    assert all(report.minimum_year == 2002 for report in reports)
    assert all(report.maximum_year == 2022 for report in reports)
