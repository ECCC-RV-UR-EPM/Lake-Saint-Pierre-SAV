# SAV NC Code and Data Package

This package contains code and data used for the Lake Saint-Pierre submerged aquatic vegetation (SAV) modelling workflow. The packaged data focus on the training inputs for the daily water temperature, TSS, TP, and TN models, the annual SAV training table, and the WRTDS tributary concentration products used in the updated figure and boundary-forcing workflow.

The package is organized as a compact review package. It is intended to document the computational workflow and provide the main model-training inputs included for submission. Large generated prediction grids, fitted model bundles, map outputs, and figure-production workspaces are not included here and can be archived separately if required by the journal or data availability plan.

## Package layout

- `Code/`: Python scripts for daily water temperature, TSS, TP, TN modelling, annual SAV modelling, and scenario analyses.
- `Data/`: packaged model-training and supporting data files.
- `Data/WRTDS/`: WRTDS daily concentration products for tributary TSS, TP, and TN.
- `DATA_MANAGEMENT.md`: summary of included data products and preprocessing steps.
- `MODEL_TRAINING_AND_VALIDATION.md`: summary of model training, validation design, and evaluation metrics.
- `README_upstream_preprocessing.md`: notes on the upstream water temperature, TSS, TP, TN, and WRTDS data-preparation workflow.
- `requirements.txt`: Python package versions used for the workflow.

## Included data files

The `Data/` folder currently includes:

- `Data/Temp_training_daily_observe_to_train_clip_filled.xlsx`
- `Data/TSS_training_daily_filled.xlsx`
- `Data/TP_training_daily_filled.xlsx`
- `Data/TN_training_daily_filled.xlsx`
- `Data/cascade_training_table.csv`
- `Data/WRTDS/*.csv`

These files correspond to:

- satellite lake-surface water temperature observations used in the water-temperature workflow;
- filled training tables for daily water temperature, TSS, TP, and TN models;
- the assembled annual SAV training table used by the annual SAV classifier;
- WRTDS tributary concentration time series for TSS, TP, and TN.

## Environment setup

The workflow was tested with Python 3.12.10 and the package versions listed in `requirements.txt`.

Create and activate a clean Python environment, then install the required packages:

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Most scripts use package-relative paths. Some upstream scripts document intermediate products that are not included in this compact package; those products should be obtained from the full data archive before rerunning the complete prediction workflow.

## Main workflow represented by this package

The package documents three connected stages:

1. Preparation and training of daily water temperature, TSS, TP, and TN models.
2. Use of WRTDS tributary products as external boundary-forcing inputs and figure-supporting data.
3. Annual SAV model training from the assembled annual table `Data/cascade_training_table.csv`.

The included `cascade_training_table.csv` is the final annual training table for the SAV classifier. It contains observed SAV records and annual predictor values assembled from the upstream environmental modelling workflow.

## Annual SAV model scripts

The annual SAV model is implemented in:

```text
Code/sav_annual_common.py
Code/Run_annual_SAV_baseline.py
```

These scripts document the annual classifier, leave-one-year-out validation design, and prediction workflow. Full reruns of annual map prediction and scenario scripts require the associated annual prediction grids, spatial-constraint files, and generated model outputs, which are not included in the current compact `Data/` folder.

## Upstream daily model scripts

The daily environmental modelling scripts include:

```text
Code/Train_water_temp_LOYO_satellite.py
Code/Train_TSS_model.py
Code/Train_TP_model.py
Code/Train_TN_model.py
```

The filled training tables included in `Data/` provide the main training inputs for these model stages. Additional prediction-ready daily grids and generated daily outputs are documented in `README_upstream_preprocessing.md`.

## Notes for reviewers

This package is intended to make the model-training inputs and computational workflow inspectable. The included files should be read together with the manuscript Methods and Data Availability statement. Large generated products, including full daily prediction archives, annual prediction grids, fitted model bundles, scenario maps, and figure-generation workspaces, should be archived separately when needed for full end-to-end reproduction.
