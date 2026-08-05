# Data

This repository uses a cleaned subset of public Backblaze drive telemetry.

Expected input file:

`df_cleaned.parquet`

Required columns:

- `serial_number`
- `date`
- `failure`
- `smart_5_raw`
- `smart_197_raw`
- `smart_198_raw`

The full dataset is not included in this repository because of its size.

To reproduce the modeling workflow, obtain the original public Backblaze data, create the cleaned Parquet file, and place it at:

`data/df_cleaned.parquet`

The current public repository focuses on the modeling and evaluation workflow from the cleaned dataset.
