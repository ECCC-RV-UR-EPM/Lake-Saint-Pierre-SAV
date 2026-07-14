# Data Management

This document describes the data products included in the SAV NC code and data package and summarizes the processing steps represented by the packaged files.

## Scope of this package

The current `Data/` folder contains the main model-training and supporting inputs included for submission. These are the filled daily training tables for water temperature, TSS, TP, and TN, the annual SAV training table, and the WRTDS tributary products.

Large generated products are not included in this compact package. These include full daily prediction archives, annual prediction grids, fitted model bundles, scenario maps, and figure-production workspaces.

## Included data products

### Water-temperature inputs

Files:

- `Data/Temp_training_daily_observe_to_train_clip_filled.xlsx`

The filled training workbook contains the gridded lake-surface water-temperature training records used by the daily water-temperature model. In this packaged version, the water-temperature training records span 2003-2019 after filtering the early records to keep the training period consistent with the cascade workflow.

### TSS, TP, and TN training inputs

Files:

- `Data/TSS_training_daily_filled.xlsx`
- `Data/TP_training_daily_filled.xlsx`
- `Data/TN_training_daily_filled.xlsx`

These workbooks contain the filled daily training records used for the TSS, TP, and TN model stages. They include observed water-quality records after alignment with the modelling grid and relevant environmental or boundary-forcing predictors.

### WRTDS tributary products

Files:

- `Data/WRTDS/*.csv`

The `WRTDS` folder contains daily WRTDS concentration products for tributary TSS, TP, and TN. The files are organized by tributary and constituent. These data support the tributary boundary-forcing workflow and the updated figure products.

### Annual SAV training table

File:

- `Data/cascade_training_table.csv`

This table contains the assembled annual SAV training records used by the annual SAV classifier. It includes SAV observations, annual environmental predictor values, year, and grid-cell identifiers.

## Processing summary

The workflow represented by this package combines field observations, remote-sensing data, hydrologic inputs, WRTDS tributary products, and gridded environmental predictors.

At a high level, the processing steps are:

1. Water-temperature observations were clipped to the Lake Saint-Pierre modelling domain and joined to the model grid.
2. Satellite water-temperature observations were formatted for use in the daily water-temperature workflow.
3. Daily water temperature, TSS, TP, and TN training tables were filled and aligned with grid, hydrologic, and environmental predictors.
4. WRTDS tributary concentration products were prepared for tributary boundary-forcing and figure-supporting analyses.
5. Annual SAV observations were joined with annual environmental predictor values to create `cascade_training_table.csv`.
6. The annual SAV model uses the assembled annual table for leave-one-year-out validation and final classifier training.

## Separately archived data products

The following products are part of the full computational record but are not included in the current compact package:

- prediction-ready daily input grids;
- full daily water temperature, TSS, TP, and TN prediction outputs;
- annual prediction grids used for full annual SAV map prediction;
- spatial-constraint files used during map generation;
- fitted model bundles;
- annual SAV map outputs and scenario outputs;
- figure-generation directories and exported manuscript figure workspaces.

These products can be deposited separately in a data repository and cited in the manuscript Data Availability statement.

## Data scope and interpretation

The packaged files provide the principal training inputs and workflow documentation for the modelling analyses. The annual SAV training table represents the final assembled table used for annual model fitting and validation. Full map regeneration and scenario reruns require the generated annual prediction grids and model-output directories described in the code comments and manuscript workflow.

Survey coverage and input-data availability vary among years and locations. The annual SAV workflow addresses this structure through year-based validation and by fitting preprocessing steps within each training fold when the annual model is rerun from the complete analysis archive.
