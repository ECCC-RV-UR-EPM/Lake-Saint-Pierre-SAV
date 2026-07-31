# NC Submission Package

This package contains the data, code, and figure scripts used to reproduce the modeling workflow for the Lake Saint-Pierre submerged aquatic vegetation (SAV) manuscript.

## Software availability

The Python source code and selected processed datasets used in this study are
included with the submission and are available from the public repository
below:

- Public repository: https://github.com/ECCC-RV-UR-EPM/Lake-Saint-Pierre-SAV
- Data used in the study: [`Data/`](./Data/)
- Source code used for the analyses and figures: [`Code/`](./Code/)
- Python 3.10.19: https://www.python.org/downloads/release/python-31019/

The workflow is implemented in Python and does not require a separately
compiled executable. It was developed and tested using Python 3.10.19. The
required dependencies and their tested versions are listed in
`requirements.txt`.

## Computing requirements and runtime

The analyses were run on a 64-bit Windows 11 computer with a 12th Gen Intel
Core i7-1255U processor (10 cores and 12 logical processors) and 16 GB of RAM.
The models were run on the CPU without GPU acceleration.

For a 24-year run, preparation of the yearly prediction inputs took about
3 to 4 hours for TSS and about 5 hours each for TP and TN. Prediction of the
24 yearly outputs took about 40 minutes each for TSS, TP, and TN. The complete
water-temperature stage took approximately 1.5 hours. Construction of the
annual grids, fitting and validation of the annual SAV model, baseline
prediction, and the full set of annual sensitivity scenarios took
approximately 15 minutes.

The annual SAV random forest used `n_jobs=1`. Other libraries used their
default CPU settings. The reported times are approximate because they were
estimated from consecutive output-file timestamps and may vary with disk
speed and cached inputs. Carbon emissions were not estimated.

## Package layout

- `Code/`: main modeling, preprocessing, prediction, and analysis scripts.
- `Data/01_temperature/`: staged inputs and outputs for the daily water-temperature branch.
- `Data/02_tss/`: staged inputs and outputs for the daily TSS branch.
- `Data/03_tp_tn/`: staged inputs and outputs for the daily TP and TN branches.
- `Data/04_sav_annual/`: annual SAV training data, annual predictor grids, fitted-model outputs, and annual results.
- `Data/05_spatial_constraints/`: north-shore polygon and fixed spatial constraints used by the annual SAV stage.
- `requirements.txt`: Python dependencies used to run the package.

## Environment setup

This package was tested with:

- Python 3.10.19
- the package versions listed in `requirements.txt`

For reproducibility, create one fresh virtual environment for the entire package and install:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Path configuration

The annual SAV scripts use package-relative paths. Some optional upstream daily preprocessing, training, and prediction scripts define a `base_dir`, `BASE_DIR`, or input-file path near the beginning of the script. Before running the full staged rebuild on another computer, update those path settings so that they point to the corresponding `Data/` directory in the local copy of this package.

Run all scripts using the package environment created from `requirements.txt`.
Use the package versions listed in `requirements.txt` to ensure consistent
nearest-neighbor matching and model outputs.

## Fast manuscript reproduction

To reproduce the main manuscript annual SAV results directly from the packaged annual inputs, run:

1. `Code/Run_annual_SAV_baseline.py`
2. `Code/Run_annual_SAV_tss_scenarios.py`
3. `Code/Run_annual_SAV_tp_scenarios.py`
4. `Code/Run_annual_SAV_tn_scenarios.py`
5. `Code/Run_annual_SAV_water_temp_scenarios.py`
6. `Code/Run_annual_SAV_water_depth_scenarios.py`

These annual scripts read from:

- `Data/04_sav_annual/core_data/cascade_training_table.parquet` and `Data/04_sav_annual/core_data/cascade_training_table.csv`: the annual SAV training table. The CSV file is a human-readable export of the same packaged training table stored in parquet format. This table contains annual SAV samples and the predictor values used by the random-forest model, including `Year`, `i`, `j`, `SAV`, `Water_temp`, `TSS_pred`, `TP_pred`, `TN_pred`, `Water_depth`, `Bathymetry_depth`, and `Year_norm`.
- `Data/04_sav_annual/core_data/annual_grids/`: annual predictor grids used to generate baseline hindcasts and scenario predictions for each year.

The annual scripts use functions defined in:

- `Code/sav_annual_common.py`: common training, prediction, validation, thresholding, and map/area summarization logic used by the baseline and annual scenario scripts.

Outputs are written to:

- `Data/04_sav_annual/results/`

## Full staged rebuild using the complete data package

This workflow requires the complete intermediate data package included with
the submission. The smaller public repository supports model training from
the processed training tables and reproduction of the annual SAV results from
the packaged annual inputs.

Run the scripts in the following order. The `Fill_*_prediction.py` scripts create the yearly prediction-input caches required by the corresponding prediction scripts.

1. `Code/Train_water_temp_LOYO_satellite.py`
2. `Code/Fill_temp_inputs_daily_prediction.py`
3. `Code/Predict_water_temp_2002_2024.py`
4. `Code/Train_TSS_model.py`
5. `Code/Fill_inputs_daily_TSS_prediction.py`
6. `Code/Predict_TSS_2002_2024.py`
7. `Code/Train_TP_model.py`
8. `Code/Fill_inputs_daily_TP_prediction.py`
9. `Code/Predict_TP_2002_2024.py`
10. `Code/Train_TN_model.py`
11. `Code/Fill_inputs_daily_TN_prediction.py`
12. `Code/Predict_TN_2002_2024.py`
13. `Code/Build_cascade_annual_inputs.py`
14. `Code/Run_annual_SAV_baseline.py`
15. `Code/Run_annual_SAV_tss_scenarios.py`
16. `Code/Run_annual_SAV_tp_scenarios.py`
17. `Code/Run_annual_SAV_tn_scenarios.py`
18. `Code/Run_annual_SAV_water_temp_scenarios.py`
19. `Code/Run_annual_SAV_water_depth_scenarios.py`

## Main script summary

### Daily environmental stages

- `Code/Train_water_temp_LOYO_satellite.py`: fits residual water-temperature correction models with leave-one-year-out validation for channel and non-channel cells.
- `Code/Fill_temp_inputs_daily_prediction.py`: builds yearly water-temperature prediction-input caches from the packaged daily grid files and environmental inputs.
- `Code/Predict_water_temp_2002_2024.py`: generates daily water-temperature fields from 2002 to 2024 using the trained residual models and yearly cached inputs.
- `Code/Train_TSS_model.py`: fits daily TSS models with year-wise leave-one-year-out validation for channel and non-channel cells.
- `Code/Fill_inputs_daily_TSS_prediction.py`: prepares yearly TSS prediction inputs using the daily water-temperature outputs and packaged environmental inputs.
- `Code/Predict_TSS_2002_2024.py`: generates daily TSS prediction fields from 2002 to 2024.
- `Code/Train_TP_model.py`: fits daily TP models with year-wise leave-one-year-out validation for channel and non-channel cells.
- `Code/Fill_inputs_daily_TP_prediction.py`: prepares yearly TP prediction inputs using the required upstream daily outputs.
- `Code/Predict_TP_2002_2024.py`: generates daily TP prediction fields from 2002 to 2024.
- `Code/Train_TN_model.py`: fits daily TN models with year-wise leave-one-year-out validation for channel and non-channel cells.
- `Code/Fill_inputs_daily_TN_prediction.py`: prepares yearly TN prediction inputs using the required upstream daily outputs.
- `Code/Predict_TN_2002_2024.py`: generates daily TN prediction fields from 2002 to 2024.

### Annual SAV stage

- `Code/Build_cascade_annual_inputs.py`: assembles annual August-to-September predictor grids (`annual_grids/cascade_grid_YYYY.parquet`) from the packaged upstream daily prediction branches. If the raw annual SAV observation workbook `Data/04_sav_annual/references/All_year_previous_2.xlsx` is available, it can also rebuild `cascade_training_table.parquet` and `cascade_training_table.csv` by merging those annual grids with observed SAV records. This workbook is the source table containing annual SAV observation locations and labels used to assemble the packaged annual SAV training table. The annual SAV training years used in the package are 2007, 2012, 2013, 2014, 2015, 2016, 2017, 2019, and 2021.
- The package already includes these assembled annual inputs in `Data/04_sav_annual/core_data/`, so standard reproduction of the annual SAV results does not require rerunning `Code/Build_cascade_annual_inputs.py`. If the raw annual SAV workbook is not distributed, the script falls back to the packaged `cascade_training_table.parquet` and still allows regeneration of the annual predictor grids.
- `Code/sav_annual_common.py`: shared implementation used by the annual SAV baseline and scenario scripts. It reads the packaged annual training table and annual predictor grids from `Data/04_sav_annual/core_data/`.
- `Code/Run_annual_SAV_baseline.py`: entry-point script that calls `sav_annual_common.py` to fit the annual SAV random-forest model, evaluates it with leave-one-year-out validation, and generates baseline hindcast maps and area summaries.
- `Code/Run_annual_SAV_tss_scenarios.py`: applies the fitted annual SAV model to the baseline, `TSS +25%`, `TSS -25%`, and `TSS -50%` annual scenarios.
- `Code/Run_annual_SAV_tp_scenarios.py`: applies the fitted annual SAV model to the baseline, `TP +25%`, `TP -25%`, and `TP -50%` annual scenarios.
- `Code/Run_annual_SAV_tn_scenarios.py`: applies the fitted annual SAV model to the baseline, `TN +25%`, `TN -25%`, and `TN -50%` annual scenarios.
- `Code/Run_annual_SAV_water_temp_scenarios.py`: applies the fitted annual SAV model to the baseline, `Water temperature +10%` and `Water temperature -10%` annual scenarios.
- `Code/Run_annual_SAV_water_depth_scenarios.py`: applies the fitted annual SAV model to the baseline, `Water depth +1 m` and `Water depth -1 m` annual scenarios.

All annual scenario scripts depend on the baseline annual SAV model structure implemented through `Code/sav_annual_common.py` and use the packaged annual predictor grids under `Data/04_sav_annual/core_data/annual_grids/`.

## Optional upstream preprocessing

The following optional preprocessing scripts regenerate filled training tables and yearly prediction-input caches:

- `Code/Fill_inputs_daily_Temp_train.py`: regenerates the filled water-temperature training table.
- `Code/Fill_temp_inputs_daily_prediction.py`: regenerates yearly water-temperature prediction-input caches.
- `Code/Fill_inputs_daily_TSS_train.py`: regenerates the filled TSS training table.
- `Code/Fill_inputs_daily_TSS_prediction.py`: regenerates yearly TSS prediction inputs.
- `Code/Fill_inputs_daily_TP_train.py`: regenerates the filled TP training table.
- `Code/Fill_inputs_daily_TP_prediction.py`: regenerates yearly TP prediction inputs.
- `Code/Fill_inputs_daily_TN_train.py`: regenerates the filled TN training table.
- `Code/Fill_inputs_daily_TN_prediction.py`: regenerates yearly TN prediction inputs.

These preprocessing scripts are optional when the corresponding filled training tables or yearly cached inputs are already present in the package.
