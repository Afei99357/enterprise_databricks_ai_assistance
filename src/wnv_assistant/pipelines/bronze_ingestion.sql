-- Bronze ingestion is intentionally source-preserving. Auto Loader and
-- Lakeflow manage file discovery, schema tracking, and checkpoints.

CREATE OR REFRESH STREAMING TABLE bronze_bird_source
COMMENT 'Source-preserving Illinois bird WNV observations ingested by Auto Loader'
AS
SELECT
  * EXCEPT (`Avian Phylodiversity`),
  `Avian Phylodiversity` AS Avian_Phylodiversity,
  'bird' AS source_type,
  _metadata.file_path AS source_file,
  _metadata.file_size AS source_file_size,
  _metadata.file_modification_time AS source_modification_time,
  current_timestamp() AS ingested_at
FROM STREAM read_files(
  '/Volumes/eliao/wnv_demo/landing/non_human/bird/',
  format => 'csv',
  header => true,
  inferColumnTypes => true,
  schemaEvolutionMode => 'addNewColumns',
  rescuedDataColumn => '_rescued_data'
);

CREATE OR REFRESH STREAMING TABLE bronze_horse_source
COMMENT 'Source-preserving Illinois horse WNV observations ingested by Auto Loader'
AS
SELECT
  * EXCEPT (`Avian Phylodiversity`),
  `Avian Phylodiversity` AS Avian_Phylodiversity,
  'horse' AS source_type,
  _metadata.file_path AS source_file,
  _metadata.file_size AS source_file_size,
  _metadata.file_modification_time AS source_modification_time,
  current_timestamp() AS ingested_at
FROM STREAM read_files(
  '/Volumes/eliao/wnv_demo/landing/non_human/horse/',
  format => 'csv',
  header => true,
  inferColumnTypes => true,
  schemaEvolutionMode => 'addNewColumns',
  rescuedDataColumn => '_rescued_data'
);

CREATE OR REFRESH STREAMING TABLE bronze_mosquito_source
COMMENT 'Source-preserving Illinois mosquito WNV observations ingested by Auto Loader'
AS
SELECT
  * EXCEPT (`Avian Phylodiversity`),
  `Avian Phylodiversity` AS Avian_Phylodiversity,
  'mosquito' AS source_type,
  _metadata.file_path AS source_file,
  _metadata.file_size AS source_file_size,
  _metadata.file_modification_time AS source_modification_time,
  current_timestamp() AS ingested_at
FROM STREAM read_files(
  '/Volumes/eliao/wnv_demo/landing/non_human/mosquito/',
  format => 'csv',
  header => true,
  inferColumnTypes => true,
  schemaEvolutionMode => 'addNewColumns',
  rescuedDataColumn => '_rescued_data'
);
