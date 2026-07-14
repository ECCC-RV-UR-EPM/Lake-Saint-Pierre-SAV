# Upstream Preprocessing Notes

This file documents the upstream preparation workflow represented by the code package. The current compact `Data/` folder includes the main filled training tables and WRTDS products, while large prediction-ready grids and generated daily outputs are not included.

## Included upstream inputs

The following upstream inputs are included directly in `Data/`:

- `Data/Temp_training_daily_observe_to_train_clip_filled.xlsx`
- `Data/TSS_training_daily_filled.xlsx`
- `Data/TP_training_daily_filled.xlsx`
- `Data/TN_training_daily_filled.xlsx`
- `Data/WRTDS/*.csv`

These files are the packaged training and supporting data products for the daily water temperature, TSS, TP, and TN model stages and the WRTDS tributary workflow.

## Water-temperature preparation

Relevant scripts:

```text
Code/Clip_LSP.py
Code/Satellite_i_j_date_temp.py
Code/Fill_inputs_daily_water_temp_train.py
Code/Train_water_temp_LOYO_satellite.py
Code/Fill_inputs_daily_water_temp_prediction.py
Code/Predict_water_temp_2002_2024.py
```

Included data products:

- `Data/Temp_training_daily_observe_to_train_clip_filled.xlsx`

The included filled workbook is the training table for the water-temperature model stage. Full prediction-ready daily grids and generated daily temperature outputs are not included in the compact package.

## TSS preparation

Relevant scripts:

```text
Code/Fill_inputs_daily_TSS_train.py
Code/Train_TSS_model.py
Code/Fill_inputs_daily_TSS_prediction.py
Code/Predict_TSS_2002_2024.py
```

Included data product:

- `Data/TSS_training_daily_filled.xlsx`

The included workbook is the filled daily TSS training table. Full prediction-ready TSS input grids, cached parquet inputs, fitted daily model objects, and generated daily TSS prediction outputs are not included in the compact package.

## TP preparation

Relevant scripts:

```text
Code/Fill_inputs_daily_TP_train.py
Code/Train_TP_model.py
Code/Fill_inputs_daily_TP_prediction.py
Code/Predict_TP_2002_2024.py
```

Included data product:

- `Data/TP_training_daily_filled.xlsx`

The included workbook is the filled daily TP training table. Full prediction-ready TP input grids, cached parquet inputs, fitted daily model objects, and generated daily TP prediction outputs are not included in the compact package.

## TN preparation

Relevant scripts:

```text
Code/Fill_inputs_daily_TN_train.py
Code/Train_TN_model.py
Code/Fill_inputs_daily_TN_prediction.py
Code/Predict_TN_2002_2024.py
```

Included data product:

- `Data/TN_training_daily_filled.xlsx`

The included workbook is the filled daily TN training table. Full prediction-ready TN input grids, cached parquet inputs, fitted daily model objects, and generated daily TN prediction outputs are not included in the compact package.

## WRTDS products

Included data product:

- `Data/WRTDS/*.csv`

The WRTDS files provide daily tributary concentration products for TSS, TP, and TN. These products support the tributary boundary-forcing workflow and updated figure products. Entry/exit WRTDS products, if used for specific figures, should be documented with the corresponding figure package or deposited with the full data archive.

## Annual SAV training table

Included data product:

- `Data/cascade_training_table.csv`

This table is the assembled annual SAV training table used by the annual SAV classifier. The scripts `Code/Build_cascade_annual_inputs.py`, `Code/sav_annual_common.py`, and `Code/Run_annual_SAV_baseline.py` document how the annual table is assembled and used in the complete workflow. Full annual prediction grids and generated map outputs are not included in the current compact package.

## Products not included in this compact package

The following products are documented by the code but are not included in the current `Data/` folder:

- raw intermediate workbooks used before the filled training tables;
- prediction-ready daily input grids;
- full daily prediction outputs for water temperature, TSS, TP, and TN;
- fitted daily model objects;
- annual prediction grids;
- fitted annual SAV model bundles;
- annual map and scenario outputs;
- figure-production workspaces.

These products can be archived separately when required for full end-to-end reproduction.
