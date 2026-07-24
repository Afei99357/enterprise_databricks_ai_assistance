-- Gold combines all reported county-month keys while preserving the
-- distinction between a missing report and an observed zero.

CREATE OR REFRESH MATERIALIZED VIEW gold_county_month_wnv_weather (
  CONSTRAINT valid_gold_key
    EXPECT (
      year BETWEEN 2002 AND 2100
      AND month BETWEEN 1 AND 12
      AND county IS NOT NULL
    ) ON VIOLATION FAIL UPDATE,
  CONSTRAINT consistent_cross_source_fips
    EXPECT (fips_consistent) ON VIOLATION FAIL UPDATE,
  CONSTRAINT consistent_cross_source_coordinates
    EXPECT (coordinates_consistent) ON VIOLATION FAIL UPDATE,
  CONSTRAINT consistent_cross_source_weather
    EXPECT (weather_consistent) ON VIOLATION FAIL UPDATE
)
COMMENT 'Analytics-ready Illinois county-month WNV activity and weather'
AS
WITH county_months AS (
  SELECT DISTINCT year, month, county
  FROM (
    SELECT year, month, county FROM silver_bird
    UNION ALL
    SELECT year, month, county FROM silver_horse
    UNION ALL
    SELECT year, month, county FROM silver_mosquito
  )
),
joined AS (
  SELECT
    k.year,
    k.month,
    make_date(k.year, k.month, 1) AS observation_month,
    k.county,
    b.county_fips AS bird_fips,
    h.county_fips AS horse_fips,
    m.county_fips AS mosquito_fips,
    b.bird_count,
    h.horse_count,
    m.mosquito_count,
    b.latitude AS bird_latitude,
    h.latitude AS horse_latitude,
    m.latitude AS mosquito_latitude,
    b.longitude AS bird_longitude,
    h.longitude AS horse_longitude,
    m.longitude AS mosquito_longitude,
    b.land_area_2010 AS bird_land_area,
    h.land_area_2010 AS horse_land_area,
    m.land_area_2010 AS mosquito_land_area,
    b.avian_phylodiversity AS bird_avian_phylodiversity,
    h.avian_phylodiversity AS horse_avian_phylodiversity,
    m.avian_phylodiversity AS mosquito_avian_phylodiversity,
    b.wind_u_1m_shift AS bird_wind_u,
    h.wind_u_1m_shift AS horse_wind_u,
    m.wind_u_1m_shift AS mosquito_wind_u,
    b.wind_v_1m_shift AS bird_wind_v,
    h.wind_v_1m_shift AS horse_wind_v,
    m.wind_v_1m_shift AS mosquito_wind_v,
    b.temperature_k_1m_shift AS bird_temperature_k,
    h.temperature_k_1m_shift AS horse_temperature_k,
    m.temperature_k_1m_shift AS mosquito_temperature_k,
    b.precipitation_m_1m_shift AS bird_precipitation_m,
    h.precipitation_m_1m_shift AS horse_precipitation_m,
    m.precipitation_m_1m_shift AS mosquito_precipitation_m,
    b.snowfall_1m_shift AS bird_snowfall,
    h.snowfall_1m_shift AS horse_snowfall,
    m.snowfall_1m_shift AS mosquito_snowfall,
    b.surface_runoff_1m_shift AS bird_surface_runoff,
    h.surface_runoff_1m_shift AS horse_surface_runoff,
    m.surface_runoff_1m_shift AS mosquito_surface_runoff,
    b.high_vegetation_index_1m_shift AS bird_high_vegetation,
    h.high_vegetation_index_1m_shift AS horse_high_vegetation,
    m.high_vegetation_index_1m_shift AS mosquito_high_vegetation,
    b.low_vegetation_index_1m_shift AS bird_low_vegetation,
    h.low_vegetation_index_1m_shift AS horse_low_vegetation,
    m.low_vegetation_index_1m_shift AS mosquito_low_vegetation,
    b.source_file AS bird_source_file,
    h.source_file AS horse_source_file,
    m.source_file AS mosquito_source_file,
    GREATEST(
      b.latest_ingested_at,
      h.latest_ingested_at,
      m.latest_ingested_at
    ) AS latest_ingested_at
  FROM county_months k
  LEFT JOIN silver_bird b
    ON k.year = b.year AND k.month = b.month AND k.county = b.county
  LEFT JOIN silver_horse h
    ON k.year = h.year AND k.month = h.month AND k.county = h.county
  LEFT JOIN silver_mosquito m
    ON k.year = m.year AND k.month = m.month AND k.county = m.county
)
SELECT
  year,
  month,
  observation_month,
  county,
  COALESCE(mosquito_fips, bird_fips, horse_fips) AS county_fips,
  bird_count,
  horse_count,
  mosquito_count,
  bird_count IS NOT NULL AS bird_reported,
  horse_count IS NOT NULL AS horse_reported,
  mosquito_count IS NOT NULL AS mosquito_reported,
  CAST(bird_count IS NOT NULL AS INT)
    + CAST(horse_count IS NOT NULL AS INT)
    + CAST(mosquito_count IS NOT NULL AS INT) AS reported_activity_types,
  COALESCE(bird_count, 0)
    + COALESCE(horse_count, 0)
    + COALESCE(mosquito_count, 0) AS total_reported_non_human_activity,
  COALESCE(mosquito_latitude, bird_latitude, horse_latitude) AS latitude,
  COALESCE(mosquito_longitude, bird_longitude, horse_longitude) AS longitude,
  COALESCE(mosquito_land_area, bird_land_area, horse_land_area) AS land_area_2010,
  COALESCE(
    mosquito_avian_phylodiversity,
    bird_avian_phylodiversity,
    horse_avian_phylodiversity
  ) AS avian_phylodiversity,
  COALESCE(mosquito_wind_u, bird_wind_u, horse_wind_u) AS wind_u_1m_shift,
  COALESCE(mosquito_wind_v, bird_wind_v, horse_wind_v) AS wind_v_1m_shift,
  sqrt(
    pow(COALESCE(mosquito_wind_u, bird_wind_u, horse_wind_u), 2)
    + pow(COALESCE(mosquito_wind_v, bird_wind_v, horse_wind_v), 2)
  ) AS wind_speed_1m_shift,
  COALESCE(
    mosquito_temperature_k,
    bird_temperature_k,
    horse_temperature_k
  ) AS temperature_k_1m_shift,
  COALESCE(
    mosquito_temperature_k,
    bird_temperature_k,
    horse_temperature_k
  ) - 273.15 AS temperature_c_1m_shift,
  COALESCE(
    mosquito_precipitation_m,
    bird_precipitation_m,
    horse_precipitation_m
  ) AS precipitation_m_1m_shift,
  COALESCE(
    mosquito_precipitation_m,
    bird_precipitation_m,
    horse_precipitation_m
  ) * 1000 AS precipitation_mm_1m_shift,
  COALESCE(
    mosquito_snowfall,
    bird_snowfall,
    horse_snowfall
  ) AS snowfall_1m_shift,
  COALESCE(
    mosquito_surface_runoff,
    bird_surface_runoff,
    horse_surface_runoff
  ) AS surface_runoff_1m_shift,
  COALESCE(
    mosquito_high_vegetation,
    bird_high_vegetation,
    horse_high_vegetation
  ) AS high_vegetation_index_1m_shift,
  COALESCE(
    mosquito_low_vegetation,
    bird_low_vegetation,
    horse_low_vegetation
  ) AS low_vegetation_index_1m_shift,
  CASE
    WHEN bird_fips IS NOT NULL AND horse_fips IS NOT NULL
      AND bird_fips <> horse_fips THEN false
    WHEN bird_fips IS NOT NULL AND mosquito_fips IS NOT NULL
      AND bird_fips <> mosquito_fips THEN false
    WHEN horse_fips IS NOT NULL AND mosquito_fips IS NOT NULL
      AND horse_fips <> mosquito_fips THEN false
    ELSE true
  END AS fips_consistent,
  CASE
    WHEN bird_latitude IS NOT NULL AND horse_latitude IS NOT NULL
      AND (
        abs(bird_latitude - horse_latitude) > 0.0000001
        OR abs(bird_longitude - horse_longitude) > 0.0000001
      ) THEN false
    WHEN bird_latitude IS NOT NULL AND mosquito_latitude IS NOT NULL
      AND (
        abs(bird_latitude - mosquito_latitude) > 0.0000001
        OR abs(bird_longitude - mosquito_longitude) > 0.0000001
      ) THEN false
    WHEN horse_latitude IS NOT NULL AND mosquito_latitude IS NOT NULL
      AND (
        abs(horse_latitude - mosquito_latitude) > 0.0000001
        OR abs(horse_longitude - mosquito_longitude) > 0.0000001
      ) THEN false
    ELSE true
  END AS coordinates_consistent,
  CASE
    WHEN bird_temperature_k IS NOT NULL AND horse_temperature_k IS NOT NULL
      AND abs(bird_temperature_k - horse_temperature_k) > 0.0000001 THEN false
    WHEN bird_temperature_k IS NOT NULL AND mosquito_temperature_k IS NOT NULL
      AND abs(bird_temperature_k - mosquito_temperature_k) > 0.0000001 THEN false
    WHEN horse_temperature_k IS NOT NULL AND mosquito_temperature_k IS NOT NULL
      AND abs(horse_temperature_k - mosquito_temperature_k) > 0.0000001 THEN false
    WHEN bird_precipitation_m IS NOT NULL AND horse_precipitation_m IS NOT NULL
      AND abs(bird_precipitation_m - horse_precipitation_m) > 0.0000001 THEN false
    WHEN bird_precipitation_m IS NOT NULL
      AND mosquito_precipitation_m IS NOT NULL
      AND abs(bird_precipitation_m - mosquito_precipitation_m) > 0.0000001
      THEN false
    WHEN horse_precipitation_m IS NOT NULL
      AND mosquito_precipitation_m IS NOT NULL
      AND abs(horse_precipitation_m - mosquito_precipitation_m) > 0.0000001
      THEN false
    ELSE true
  END AS weather_consistent,
  bird_source_file,
  horse_source_file,
  mosquito_source_file,
  latest_ingested_at
FROM joined;
