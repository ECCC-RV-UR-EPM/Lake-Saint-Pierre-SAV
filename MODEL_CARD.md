# Lake Saint-Pierre Annual SAV Model

## Overview

This file describes the fitted model stored in
`cascade_v3_model_bundle.pkl`. The model predicts annual submerged aquatic
vegetation occurrence for 200 m grid cells in Lake Saint-Pierre, Quebec,
Canada. It is the final stage of the modeling workflow described in the
manuscript.

The model is a random forest classifier implemented with scikit-learn. It was
developed to reproduce the annual SAV reconstruction for 2002 to 2024 and the
environmental sensitivity analyses included in the study.

## Training data

The model was trained using
`Data/Training/SAV_Code_Training/Training_data/cascade_training_table.parquet`.
A CSV version of the same table is also included in the package.

The training table contains 27,416 SAV observations collected in 2007, 2012,
2013, 2014, 2015, 2016, 2017, 2019, and 2021. Among these observations, 20,372
represent SAV presence and 7,044 represent SAV absence. The environmental
predictors are August and September averages derived from the upstream daily
models.

## Model inputs and output

The model uses eight predictors in the following order:

`Water_temp`, `Water_depth`, `TSS_pred`, `TP_pred`, `TN_pred`, `Year_norm`,
`north_flag`, and `tss_x_year`.

`Year_norm` is the normalized year for the 2002 to 2024 period. `north_flag`
identifies cells within the predefined north-shore area. `tss_x_year` is the
interaction between TSS and normalized year. Missing predictor values are
filled with medians calculated from the training data.

The model returns the probability of SAV occurrence for each grid cell.
Probabilities are converted to presence or absence using a threshold of
0.3908505913242214.

## Model settings

The random forest contains 320 trees. It uses balanced subsampling to account
for the unequal numbers of presence and absence observations. The random seed
is 42 and the model runs with one processing job. Other settings use the
scikit-learn defaults in the tested software environment.

## Model evaluation

Performance was evaluated with leave-one-year-out cross-validation. In each
validation round, all observations from one year were excluded from training
and used for testing. A fold-specific Youden threshold was used to calculate
the classification metrics for each held-out year. After all held-out
predictions were combined, a pooled Youden threshold was calculated and
retained in the final model bundle for annual prediction.

Across the nine held-out years, mean accuracy was 0.926, mean balanced
accuracy was 0.885, mean precision was 0.934, mean recall was 0.970, and mean
F1 score was 0.951. Mean ROC AUC was 0.927 and mean PR AUC was 0.963.
Year-specific results are available in
`Data/Training/SAV_Code_Training/results/cascade_v3_loyo_per_year.csv`.

## Model file

The model file contains the fitted random forest, the fitted median imputer,
the classification threshold, and the ordered list of predictor names.

The file is produced by running:

```text
python Code/Training/Run_annual_SAV_baseline.py
```

The training and prediction functions used by this script are defined in
`Code/Training/sav_annual_common.py`. Running the baseline script also produces the
validation results, feature importance, annual SAV maps, and area summaries.

The saved model can be loaded through the function `load_model_bundle()` in
`Code/Training/sav_annual_common.py`.

## Software

The package was tested with Python 3.10.19. The tested package versions are
listed in `requirements.txt`. They include scikit-learn 1.7.2, pandas 2.3.3,
NumPy 2.2.6, and SciPy 1.15.2.

## Limitations

The model was developed specifically for Lake Saint-Pierre and has not been
validated for other aquatic systems. SAV observations were not available in
every year, and their spatial coverage was uneven. Part of the southern sector
was not surveyed because access to the former firing-range area was
restricted.

Water temperature, TSS, TP, and TN are predicted by upstream models, so their
uncertainty can affect the SAV results. The north-shore definition and fixed
spatial constraints also contribute spatial structure to the final maps. The
analysis uses a fixed 200 m grid and does not account for annual changes in the
lake boundary.

The environmental scenarios are sensitivity analyses. They change one annual
predictor while holding the other predictors constant. They should not be
interpreted as fully propagated forecasts or as direct evidence of causation.
