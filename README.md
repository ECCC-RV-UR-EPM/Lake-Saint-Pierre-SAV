# Lake Saint-Pierre SAV Modeling Package

This package contains the training code, model-evaluation code, and selected
data used for the Lake Saint-Pierre submerged aquatic vegetation (SAV)
modeling study.

##1. Download and copy to the following path: below input file, trained model file and results folder are shared separately due to GitHub file-sharing limitations.
1.1> input file: "LSP_LSWT_200m_rev02_2000-2024_satellite_reduced.csv" => "..\Data\Training\"
https://drive.google.com/file/d/1kaIVeiORp3j2IZLWYPIN-8H3L3itu4CA/view?usp=drive_link

1.2> trained model "cascade_v3_model_bundle.pkl"=> "..\Data\Test"
https://drive.google.com/file/d/1eOU-7cGiG2d5SYO5SIVAoA6yx_wcQbQy/view?usp=sharing

1.3> "..\Data\Training\SAV_Code_Training\results"
https://drive.google.com/drive/folders/1LOUgJJaplTO2Z2k6XJsWTt7c_PmH8ly8?usp=sharing

##2. Software

The workflow was developed and tested with Python 3.10.19 on a 64-bit
Windows 11 computer. The tested dependency versions are listed in
`requirements.txt`.

Create and activate a virtual environment before installing the dependencies:

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

##3. Computing requirements and approximate runtime

The analyses were run on a 64-bit Windows 11 computer with a 12th Gen Intel
Core i7-1255U processor (10 cores and 12 logical processors) and 16 GB of RAM.
The models were run on the CPU without GPU acceleration.

For the 23-year period from 2002 to 2024, preparation of the yearly prediction
inputs took approximately 3-4 hours for TSS and approximately 5 hours each for
TP and TN. Prediction of the 23 yearly outputs took approximately 40 minutes
each for TSS, TP, and TN. The complete water-temperature stage took
approximately 1.5 hours. Construction of the annual grids, fitting and
leave-one-year-out validation of the annual SAV model, baseline prediction,
and the full set of annual sensitivity scenarios took approximately
15 minutes.

The annual SAV random forest used `n_jobs=1`. Other libraries used their
default CPU settings. Runtime estimates were derived from consecutive
output-file timestamps and may vary with processor performance, disk speed,
and cached inputs. Carbon emissions were not estimated.

##4. Package layout

- `Code/Training/`: preprocessing, model-training, prediction, annual SAV,
  and scenario-analysis scripts.
- `Code/Test/`: standalone leave-one-year-out evaluation code for the
  annual SAV classifier.
- `Data/Training/`: input data and retained outputs associated with model
  training.
- `Data/Training/SAV_Code_Training/Training_data/`: annual SAV training
  table and annual predictor grids.
- `Data/Training/SAV_Code_Training/results/`: retained annual SAV model and
  evaluation outputs.
- `Data/Training/Spatial_constraints/`: spatial constraints used by the
  annual SAV model.
- `Data/Test/`: self-contained inputs for the annual SAV model test.
- `requirements.txt`: tested Python dependencies.
- `MODEL_CARD.md`: annual SAV model description, inputs, evaluation, and
  limitations.

##5. Training data

`Data/Training/` contains the filled observation tables used to train the
water-temperature, TSS, TP, and TN models, together with their corresponding
source tables and spatial lookup files.

The annual SAV training materials are stored under:

```text
Data/Training/SAV_Code_Training/
```

The principal annual SAV inputs are:

```text
Data/Training/SAV_Code_Training/Training_data/cascade_training_table.parquet
Data/Training/SAV_Code_Training/Training_data/cascade_training_table.csv
Data/Training/SAV_Code_Training/Training_data/annual_grids/
```

The CSV and Parquet versions of `cascade_training_table` contain the same
training records. The Parquet file is used by the Python workflow, while the
CSV file is included as a human-readable export.

The annual SAV training years are 2007, 2012, 2013, 2014, 2015, 2016, 2017,
2019, and 2021. Annual hindcasts cover 2002-2024.

##6. Training code

The principal model-training scripts are:

1. `Code/Training/Train_water_temp_LOYO_satellite.py`
2. `Code/Training/Train_TSS_model.py`
3. `Code/Training/Train_TP_model.py`
4. `Code/Training/Train_TN_model.py`
5. `Code/Training/Run_annual_SAV_baseline.py`

The daily preprocessing and prediction scripts are also retained in
`Code/Training/`. These scripts document the complete staged workflow used to
generate water-temperature, TSS, TP, and TN fields from 2002 to 2024.

Some upstream scripts contain a `base_dir`, `BASE_DIR`, or other input path
near the beginning of the file. Before running those scripts on another
computer, set these paths to the corresponding local data location. The
reduced `Sent_to_Reza` package contains the training and annual-test materials
listed above; it does not include every large yearly intermediate prediction
cache used in the full staged rebuild.

##7. Workflow and execution order

Run the commands below from the package root, where `README.md`,
`requirements.txt`, `Code/`, and `Data/` are located.

###7.1 Route A: reproduce the annual SAV results from packaged annual inputs

This is the recommended route for reproducing the annual SAV model, baseline
maps, and sensitivity scenarios. It uses the packaged training table and
annual predictor grids under
`Data/Training/SAV_Code_Training/Training_data/`.

Run the baseline script first:

```text
python Code/Training/Run_annual_SAV_baseline.py
```

This script fits the eight-predictor annual SAV random forest, performs
leave-one-year-out validation, saves the fitted model bundle, and generates
the baseline annual SAV maps and summaries. Its outputs are written to:

```text
Data/Training/SAV_Code_Training/results/
```

After the baseline run has completed, run the scenario scripts:

```text
python Code/Training/Run_annual_SAV_tss_scenarios.py
python Code/Training/Run_annual_SAV_tp_scenarios.py
python Code/Training/Run_annual_SAV_tn_scenarios.py
python Code/Training/Run_annual_SAV_water_temp_scenarios.py
python Code/Training/Run_annual_SAV_water_depth_scenarios.py
```

The baseline script must be run before the scenario scripts because the
scenario analyses use the fitted annual SAV model bundle produced by the
baseline run.

###7.2 Route B: rebuild the staged environmental and annual SAV workflow

Use this route only when the complete upstream daily inputs and yearly
prediction files are available. Before running these scripts, update any
`base_dir`, `BASE_DIR`, or explicit input paths near the beginning of the
scripts so that they point to the corresponding files on the local computer.

| Step | Script | Main output |
|---:|---|---|
| 1 | `Fill_inputs_daily_Temp_train.py` | Filled water-temperature training table |
| 2 | `Train_water_temp_LOYO_satellite.py` | Channel and non-channel water-temperature model bundles |
| 3 | `Fill_temp_inputs_daily_prediction.py` | Yearly water-temperature prediction-input caches |
| 4 | `Predict_water_temp_2002_2024.py` | Daily water-temperature predictions |
| 5 | `Fill_inputs_daily_TSS_train.py` | Filled TSS training table |
| 6 | `Train_TSS_model.py` | Channel and non-channel TSS model bundles |
| 7 | `Fill_inputs_daily_TSS_prediction.py` | Yearly TSS prediction inputs |
| 8 | `Predict_TSS_2002_2024.py` | Daily TSS predictions |
| 9 | `Fill_inputs_daily_TP_train.py` | Filled TP training table |
| 10 | `Train_TP_model.py` | Channel and non-channel TP model bundles |
| 11 | `Fill_inputs_daily_TP_prediction.py` | Yearly TP prediction inputs |
| 12 | `Predict_TP_2002_2024.py` | Daily TP predictions |
| 13 | `Fill_inputs_daily_TN_train.py` | Filled TN training table |
| 14 | `Train_TN_model.py` | Channel and non-channel TN model bundles |
| 15 | `Fill_inputs_daily_TN_prediction.py` | Yearly TN prediction inputs |
| 16 | `Predict_TN_2002_2024.py` | Daily TN predictions |
| 17 | `Build_cascade_annual_inputs.py` | Annual predictor grids and annual SAV training table |
| 18 | `Run_annual_SAV_baseline.py` | Fitted annual SAV model, LOYO results, and baseline maps |
| 19 | `Run_annual_SAV_tss_scenarios.py` | TSS sensitivity results |
| 20 | `Run_annual_SAV_tp_scenarios.py` | TP sensitivity results |
| 21 | `Run_annual_SAV_tn_scenarios.py` | TN sensitivity results |
| 22 | `Run_annual_SAV_water_temp_scenarios.py` | Water-temperature sensitivity results |
| 23 | `Run_annual_SAV_water_depth_scenarios.py` | Water-depth sensitivity results |

All scripts in this table are located under `Code/Training/`. The
`Fill_*_train.py` steps may be skipped when the corresponding filled training
tables are already available. The `Fill_*_prediction.py` steps may be skipped
when the corresponding yearly prediction-input caches are already available.
The reduced package does not contain every large intermediate cache required
for a complete Route B run.

##8. Annual SAV model

The annual SAV random-forest classifier uses the following eight predictors:

```text
Water_temp
Water_depth
TSS_pred
TP_pred
TN_pred
Year_norm
north_flag
tss_x_year
```

The model contains 320 trees, uses balanced subsampling, and uses random seed
42. Missing predictor values are imputed using medians calculated from the
training data. Additional details are provided in `MODEL_CARD.md`.

##9. Annual SAV model test

The model test is not a prerequisite for Route A. The packaged file
`Data/Test/cascade_v3_model_bundle.pkl` is a copy of the model bundle produced
by `Run_annual_SAV_baseline.py`, and is included so that the model-evaluation
workflow can be run as a standalone analysis.

Run the leave-one-year-out evaluation from the package root:

```text
python Code/Test/Test_SAV_model_LOYO.py
```

The test script reads:

```text
Data/Test/cascade_training_table.parquet
Data/Test/cascade_v3_model_bundle.pkl
Data/Test/north_polygon_paper.json
```

The ordered eight-predictor list, random-forest configuration, and median
imputer configuration are read directly from the packaged model bundle.

Year-specific metrics are calculated using a separate Youden threshold for
each held-out year. After all held-out predictions are combined, a pooled
Youden threshold is calculated for the overall LOYO summary and retained in
the final model bundle for annual prediction.

Test outputs are written to:

```text
Code/Test/outputs/
```

The output files are:

- `sav_loyo_performance_metrics.csv`
- `sav_loyo_confusion_matrix.csv`
- `sav_loyo_metrics_by_year.csv`
- `sav_loyo_fold_summary.csv`

##10. Reproducibility notes

Random seeds are fixed in the model-training and evaluation workflows.
Package versions in `requirements.txt` should be used because changes in
scientific Python libraries can affect nearest-neighbor matching, fitted
models, and serialized model compatibility.

The environmental scenarios change one annual predictor while holding the
other predictors constant. They are sensitivity analyses and should not be
interpreted as fully propagated forecasts or as direct evidence of
causation.
