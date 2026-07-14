# Model Training and Validation

This document summarizes the model-training inputs, annual SAV training design, and evaluation metrics documented by the SAV NC code and data package.

## Daily environmental model inputs

The current `Data/` folder includes filled training tables for the daily environmental model stages:

- `Data/Temp_training_daily_observe_to_train_clip_filled.xlsx`
- `Data/TSS_training_daily_filled.xlsx`
- `Data/TP_training_daily_filled.xlsx`
- `Data/TN_training_daily_filled.xlsx`

The corresponding model scripts are:

```text
Code/Train_water_temp_LOYO_satellite.py
Code/Train_TSS_model.py
Code/Train_TP_model.py
Code/Train_TN_model.py
```

The water-temperature training records used by the water-temperature workflow are included in:

- `Data/Temp_training_daily_observe_to_train_clip_filled.xlsx`

WRTDS tributary products used in the boundary-forcing workflow are included under:

- `Data/WRTDS/`

## Annual SAV model

The annual SAV model is implemented in:

```text
Code/sav_annual_common.py
Code/Run_annual_SAV_baseline.py
```

The annual model is a `RandomForestClassifier` fitted to SAV observations and annual environmental predictors. The assembled annual training table included in the package is:

- `Data/cascade_training_table.csv`

The annual classifier uses this table to evaluate temporal transfer across observed SAV survey years and to fit the final annual SAV model in the full analysis workflow.

## Annual SAV predictors

The annual model uses environmental and spatial predictors assembled from the upstream workflow. The feature set documented in `sav_annual_common.py` includes:

```text
Water_temp
Water_depth
TSS_pred
TP_pred
TN_pred
Year_norm
north_flag
year_x_north
tss_x_north
depth_x_north
temp_x_north
tp_x_north
tn_x_north
tss_x_year
```

The interaction terms represent spatial structure and temporal or environmental differences between north-shore and other parts of the study area.

## Training configuration

The annual baseline classifier uses:

```text
RandomForestClassifier(
    n_estimators=320,
    random_state=42,
    class_weight="balanced_subsample",
    n_jobs=1,
)
```

Missing predictor values are handled with `SimpleImputer(strategy="median")`. During leave-one-year-out validation, the imputer is fitted on the training years and then applied to the held-out year.

## Validation design

The annual SAV workflow uses leave-one-year-out temporal validation. For each observed SAV survey year, the model is trained using all other survey years and evaluated on the held-out year.

This validation design evaluates out-of-sample transfer across survey years and matches the temporal structure of the annual SAV application.

## Validation metrics

The annual baseline script reports the following per-year metrics when the full annual workflow is run:

```text
Year
Accuracy
BAcc
Precision
Recall
Specificity
F1
ROC_AUC
PR_AUC
Threshold_fold
```

`BAcc` is balanced accuracy. `ROC_AUC` and `PR_AUC` are threshold-independent metrics when both classes are present in the held-out year. `Threshold_fold` is the fold-specific Youden threshold used for classification metrics.

## Full prediction and scenario workflow

The code package also includes scripts for annual prediction and scenario analyses:

```text
Code/Run_annual_SAV_baseline.py
Code/Run_annual_SAV_tss_scenarios.py
Code/Run_annual_SAV_tp_scenarios.py
Code/Run_annual_SAV_tn_scenarios.py
Code/Run_annual_SAV_water_temp_scenarios.py
Code/Run_annual_SAV_water_depth_scenarios.py
```

Full reruns of annual map prediction and scenario analyses require annual prediction grids, spatial-constraint files, and generated model-output directories from the complete analysis archive. These large generated products are not included in the current compact `Data/` folder.

## Interpretation notes

The packaged annual table supports inspection of the annual training data and model design. The complete end-to-end map and scenario workflow is documented in the code, while large generated products should be archived separately for full reproduction.
