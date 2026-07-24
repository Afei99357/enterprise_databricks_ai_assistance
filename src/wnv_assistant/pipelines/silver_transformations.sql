-- Silver views normalize the three sources to one row per county and month.
-- Repeated enrichment rows are collapsed only after consistency checks.

CREATE OR REFRESH MATERIALIZED VIEW silver_bird (
  CONSTRAINT valid_bird_key
    EXPECT (
      year BETWEEN 2002 AND 2100
      AND month BETWEEN 1 AND 12
      AND county IS NOT NULL
    ) ON VIOLATION FAIL UPDATE,
  CONSTRAINT valid_bird_count
    EXPECT (bird_count >= 0) ON VIOLATION FAIL UPDATE,
  CONSTRAINT consistent_bird_group
    EXPECT (distinct_value_max <= 1) ON VIOLATION FAIL UPDATE
)
COMMENT 'Validated bird WNV observations at one row per Illinois county-month'
AS
SELECT
  Year AS year,
  Month AS month,
  make_date(Year, Month, 1) AS observation_month,
  initcap(trim(County)) AS county,
  CAST(MAX(FIPS) AS INT) AS county_fips,
  MAX(Bird) AS bird_count,
  MAX(County_Seat_Latitude) AS latitude,
  MAX(County_Seat_Longitude) AS longitude,
  MAX(Land_Area_2010) AS land_area_2010,
  MAX(Avian_Phylodiversity) AS avian_phylodiversity,
  MAX(u10_1m_shift) AS wind_u_1m_shift,
  MAX(v10_1m_shift) AS wind_v_1m_shift,
  sqrt(
    pow(MAX(u10_1m_shift), 2) + pow(MAX(v10_1m_shift), 2)
  ) AS wind_speed_1m_shift,
  MAX(t2m_1m_shift) AS temperature_k_1m_shift,
  MAX(t2m_1m_shift) - 273.15 AS temperature_c_1m_shift,
  MAX(tp_1m_shift) AS precipitation_m_1m_shift,
  MAX(tp_1m_shift) * 1000 AS precipitation_mm_1m_shift,
  MAX(sf_1m_shift) AS snowfall_1m_shift,
  MAX(sro_1m_shift) AS surface_runoff_1m_shift,
  MAX(lai_hv_1m_shift) AS high_vegetation_index_1m_shift,
  MAX(lai_lv_1m_shift) AS low_vegetation_index_1m_shift,
  COUNT(*) AS source_row_count,
  GREATEST(
    COUNT(DISTINCT Bird),
    COUNT(DISTINCT FIPS),
    COUNT(DISTINCT County_Seat_Latitude),
    COUNT(DISTINCT County_Seat_Longitude),
    COUNT(DISTINCT Land_Area_2010),
    COUNT(DISTINCT Avian_Phylodiversity),
    COUNT(DISTINCT u10_1m_shift),
    COUNT(DISTINCT v10_1m_shift),
    COUNT(DISTINCT t2m_1m_shift),
    COUNT(DISTINCT tp_1m_shift),
    COUNT(DISTINCT sf_1m_shift),
    COUNT(DISTINCT sro_1m_shift),
    COUNT(DISTINCT lai_hv_1m_shift),
    COUNT(DISTINCT lai_lv_1m_shift)
  ) AS distinct_value_max,
  MAX(source_file) AS source_file,
  MAX(ingested_at) AS latest_ingested_at
FROM bronze_bird_source
GROUP BY Year, Month, initcap(trim(County));

CREATE OR REFRESH MATERIALIZED VIEW silver_horse (
  CONSTRAINT valid_horse_key
    EXPECT (
      year BETWEEN 2002 AND 2100
      AND month BETWEEN 1 AND 12
      AND county IS NOT NULL
    ) ON VIOLATION FAIL UPDATE,
  CONSTRAINT valid_horse_count
    EXPECT (horse_count >= 0) ON VIOLATION FAIL UPDATE,
  CONSTRAINT consistent_horse_group
    EXPECT (distinct_value_max <= 1) ON VIOLATION FAIL UPDATE
)
COMMENT 'Validated horse WNV observations at one row per Illinois county-month'
AS
SELECT
  Year AS year,
  Month AS month,
  make_date(Year, Month, 1) AS observation_month,
  initcap(trim(County)) AS county,
  CAST(MAX(FIPS) AS INT) AS county_fips,
  MAX(Horse) AS horse_count,
  MAX(County_Seat_Latitude) AS latitude,
  MAX(County_Seat_Longitude) AS longitude,
  MAX(Land_Area_2010) AS land_area_2010,
  MAX(Avian_Phylodiversity) AS avian_phylodiversity,
  MAX(u10_1m_shift) AS wind_u_1m_shift,
  MAX(v10_1m_shift) AS wind_v_1m_shift,
  sqrt(
    pow(MAX(u10_1m_shift), 2) + pow(MAX(v10_1m_shift), 2)
  ) AS wind_speed_1m_shift,
  MAX(t2m_1m_shift) AS temperature_k_1m_shift,
  MAX(t2m_1m_shift) - 273.15 AS temperature_c_1m_shift,
  MAX(tp_1m_shift) AS precipitation_m_1m_shift,
  MAX(tp_1m_shift) * 1000 AS precipitation_mm_1m_shift,
  MAX(sf_1m_shift) AS snowfall_1m_shift,
  MAX(sro_1m_shift) AS surface_runoff_1m_shift,
  MAX(lai_hv_1m_shift) AS high_vegetation_index_1m_shift,
  MAX(lai_lv_1m_shift) AS low_vegetation_index_1m_shift,
  COUNT(*) AS source_row_count,
  GREATEST(
    COUNT(DISTINCT Horse),
    COUNT(DISTINCT FIPS),
    COUNT(DISTINCT County_Seat_Latitude),
    COUNT(DISTINCT County_Seat_Longitude),
    COUNT(DISTINCT Land_Area_2010),
    COUNT(DISTINCT Avian_Phylodiversity),
    COUNT(DISTINCT u10_1m_shift),
    COUNT(DISTINCT v10_1m_shift),
    COUNT(DISTINCT t2m_1m_shift),
    COUNT(DISTINCT tp_1m_shift),
    COUNT(DISTINCT sf_1m_shift),
    COUNT(DISTINCT sro_1m_shift),
    COUNT(DISTINCT lai_hv_1m_shift),
    COUNT(DISTINCT lai_lv_1m_shift)
  ) AS distinct_value_max,
  MAX(source_file) AS source_file,
  MAX(ingested_at) AS latest_ingested_at
FROM bronze_horse_source
GROUP BY Year, Month, initcap(trim(County));

CREATE OR REFRESH MATERIALIZED VIEW silver_mosquito (
  CONSTRAINT valid_mosquito_key
    EXPECT (
      year BETWEEN 2002 AND 2100
      AND month BETWEEN 1 AND 12
      AND county IS NOT NULL
    ) ON VIOLATION FAIL UPDATE,
  CONSTRAINT valid_mosquito_count
    EXPECT (mosquito_count >= 0) ON VIOLATION FAIL UPDATE,
  CONSTRAINT consistent_mosquito_group
    EXPECT (distinct_value_max <= 1) ON VIOLATION FAIL UPDATE
)
COMMENT 'Validated mosquito WNV observations at one row per Illinois county-month'
AS
SELECT
  Year AS year,
  Month AS month,
  make_date(Year, Month, 1) AS observation_month,
  initcap(trim(County)) AS county,
  CAST(MAX(FIPS) AS INT) AS county_fips,
  MAX(Mosquito) AS mosquito_count,
  MAX(County_Seat_Latitude) AS latitude,
  MAX(County_Seat_Longitude) AS longitude,
  MAX(Land_Area_2010) AS land_area_2010,
  MAX(Avian_Phylodiversity) AS avian_phylodiversity,
  MAX(u10_1m_shift) AS wind_u_1m_shift,
  MAX(v10_1m_shift) AS wind_v_1m_shift,
  sqrt(
    pow(MAX(u10_1m_shift), 2) + pow(MAX(v10_1m_shift), 2)
  ) AS wind_speed_1m_shift,
  MAX(t2m_1m_shift) AS temperature_k_1m_shift,
  MAX(t2m_1m_shift) - 273.15 AS temperature_c_1m_shift,
  MAX(tp_1m_shift) AS precipitation_m_1m_shift,
  MAX(tp_1m_shift) * 1000 AS precipitation_mm_1m_shift,
  MAX(sf_1m_shift) AS snowfall_1m_shift,
  MAX(sro_1m_shift) AS surface_runoff_1m_shift,
  MAX(lai_hv_1m_shift) AS high_vegetation_index_1m_shift,
  MAX(lai_lv_1m_shift) AS low_vegetation_index_1m_shift,
  COUNT(*) AS source_row_count,
  GREATEST(
    COUNT(DISTINCT Mosquito),
    COUNT(DISTINCT FIPS),
    COUNT(DISTINCT County_Seat_Latitude),
    COUNT(DISTINCT County_Seat_Longitude),
    COUNT(DISTINCT Land_Area_2010),
    COUNT(DISTINCT Avian_Phylodiversity),
    COUNT(DISTINCT u10_1m_shift),
    COUNT(DISTINCT v10_1m_shift),
    COUNT(DISTINCT t2m_1m_shift),
    COUNT(DISTINCT tp_1m_shift),
    COUNT(DISTINCT sf_1m_shift),
    COUNT(DISTINCT sro_1m_shift),
    COUNT(DISTINCT lai_hv_1m_shift),
    COUNT(DISTINCT lai_lv_1m_shift)
  ) AS distinct_value_max,
  MAX(source_file) AS source_file,
  MAX(ingested_at) AS latest_ingested_at
FROM bronze_mosquito_source
GROUP BY Year, Month, initcap(trim(County));
