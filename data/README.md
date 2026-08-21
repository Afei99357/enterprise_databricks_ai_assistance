# Local source data

Source datasets and documents are intentionally excluded from Git. Store the
authoritative copies in governed Databricks Volumes for deployed workflows and
place local development copies under the paths below only when needed.

Before redistributing any source, verify its publisher, source URL, license,
terms of use, and required attribution. Those provenance details have not yet
been verified for every collected file.

## Expected local files

| Local path | SHA-256 |
|---|---|
| `non_human/bird_illinois_county_02_to_22.csv` | `f82c0ff2baf59d05ae6243b10735221077c7531f4682463eeeee211f490b88fd` |
| `non_human/horse_illinois_county_02_to_22.csv` | `422266d3adbec8d9a568e64e7c0def741db235366118a6a8b1318f4ffc651f1a` |
| `non_human/mos_illinois_county_02_to_22.csv` | `36842541a49e4578988e12f9cac2ad4ec118037c475d3c4aef8fc0f4ba95ce51` |
| `non_human_data_2002_2017_monthly_aggregated_by_county.csv` | `3f26073c6c84980d957ddd66eff9c07ec3532192d49b9151c415d0ed768c4087` |
| `non_human_data_2002_2017_monthly_climate.csv` | `e9652fc4fc3f7e3be9dffe44f0cacea2e784ec809bf5590152693b10db92f898` |

The twelve GeoTIFF files under `landscape/` are not used by the current WNV
demonstration pipeline.

## Validation

Run the full local data contract only after placing the three activity files
under `data/non_human`:

```bash
WNV_RUN_REAL_DATA_TESTS=1 uv run pytest -m real_data
uv run validate-wnv-data --data-directory data/non_human
```

The normal CI test suite does not require these files.
