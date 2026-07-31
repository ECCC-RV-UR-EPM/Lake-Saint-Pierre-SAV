"""Assemble annual SAV cascade inputs from packaged daily prediction outputs.

This script performs data preparation only. It:

1. reads daily Water temperature, TSS, TP, TN, and depth data;
2. selects August and September;
3. calculates annual grid-cell means;
4. writes cascade_grid_2002.parquet through cascade_grid_2024.parquet;
5. matches historical SAV observations to annual environmental grids;
6. writes cascade_training_table.parquet and CSV.

It does not train an SAV model or generate SAV predictions.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


PKG = Path(__file__).resolve().parents[1]
DATA_DIR = PKG / "Data"
ANNUAL_DIR = DATA_DIR / "04_sav_annual"
CORE_DIR = ANNUAL_DIR / "core_data"
GRID_DIR = CORE_DIR / "annual_grids"
REF_DIR = ANNUAL_DIR / "references"

GRID_DIR.mkdir(parents=True, exist_ok=True)


TRAIN_OBS = REF_DIR / "All_year_previous_2.xlsx"

# Daily predicted water temperature
TEMP_DIR = (
    DATA_DIR
    / "01_temperature"
    / "Prediction"
    / "Final_Temp_Output_monthly_latlon"
)

# Daily predicted TSS
TSS_RES_DIR = (
    DATA_DIR
    / "02_tss"
    / "Prediction"
    / "TSS_prediction_results"
)

# Daily water depth and bathymetry
TSS_FILL_DIR = (
    DATA_DIR
    / "02_tss"
    / "Prediction"
    / "TSS_prediction_filled_parquet"
)

# Daily predicted TP
TP_RES_DIR = (
    DATA_DIR
    / "03_tp_tn"
    / "TP_prediction_LSP"
    / "Prediction"
    / "TP_prediction_results"
)

# Daily predicted TN
TN_RES_DIR = (
    DATA_DIR
    / "03_tp_tn"
    / "TN_prediction_LSP"
    / "Prediction"
    / "TN_prediction_results"
)

PACKAGED_TRAINING = CORE_DIR / "cascade_training_table.parquet"
PACKAGED_TRAINING_CSV = CORE_DIR / "cascade_training_table.csv"

YEARS = list(range(2002, 2025))
TRAIN_YEARS = [2007, 2012, 2013, 2014, 2015, 2016, 2017, 2019, 2021]
MONTHS = {8, 9}






def build_annual_grid(year: int, overwrite: bool = False) -> pd.DataFrame:
    out_file = GRID_DIR / f"cascade_grid_{year}.parquet"
    if out_file.exists() and not overwrite:
        return pd.read_parquet(out_file)

    temp = pd.read_parquet(TEMP_DIR / f"Temp_{year}_daily.parquet")
    tss = pd.read_parquet(TSS_RES_DIR / f"TSS_{year}_daily.parquet")
    tp = pd.read_parquet(TP_RES_DIR / f"TP_{year}_daily.parquet")
    tn = pd.read_parquet(TN_RES_DIR / f"TN_{year}_daily.parquet")
    depth = pd.read_parquet(
        TSS_FILL_DIR / f"TSS_{year}.parquet",
        columns=["i", "j", "Date", "Water_depth", "Bathymetry_depth"],
    )

    for df in (temp, tss, tp, tn, depth):
        df["Date"] = pd.to_datetime(df["Date"])

    temp = temp[temp["Date"].dt.month.isin(MONTHS)].copy()
    tss = tss[tss["Date"].dt.month.isin(MONTHS)].copy()
    tp = tp[tp["Date"].dt.month.isin(MONTHS)].copy()
    tn = tn[tn["Date"].dt.month.isin(MONTHS)].copy()
    depth = depth[depth["Date"].dt.month.isin(MONTHS)].copy()

    df = temp.merge(tss, on=["Date", "i", "j"], how="inner")
    df = df.merge(tp, on=["Date", "i", "j"], how="inner")
    df = df.merge(tn, on=["Date", "i", "j"], how="inner")
    df = df.merge(depth, on=["Date", "i", "j"], how="left")

    annual = df.groupby(["i", "j"], as_index=False).agg(
        {
            "Water_temp": "mean",
            "TSS_pred": "mean",
            "TP_pred": "mean",
            "TN_pred": "mean",
            "Water_depth": "mean",
            "Bathymetry_depth": "mean",
        }
    )
    annual["Year"] = year
    annual.to_parquet(out_file, index=False)
    return annual


def rebuild_all_annual_grids(overwrite: bool = False) -> None:
    for year in YEARS:
        out_file = GRID_DIR / f"cascade_grid_{year}.parquet"
        existed = out_file.exists()

        annual = build_annual_grid(year, overwrite=overwrite)

        if existed and not overwrite:
            action = "Reused"
        else:
            action = "Built"

        print(
            f"{action} annual grid for {year}: "
            f"{len(annual):,} cells"
        )

def load_or_build_training_table(overwrite: bool = False) -> pd.DataFrame:
    if TRAIN_OBS.exists():
        if PACKAGED_TRAINING.exists() and not overwrite:
            return pd.read_parquet(PACKAGED_TRAINING)

        obs = pd.read_excel(TRAIN_OBS)[["Year", "i", "j", "SAV"]].copy()
        obs["SAV"] = pd.to_numeric(obs["SAV"], errors="coerce")
        obs = obs[obs["SAV"].isin([0, 1])].copy()

        annuals = [pd.read_parquet(GRID_DIR / f"cascade_grid_{year}.parquet")
            for year in TRAIN_YEARS]
        annual_all = pd.concat(annuals, ignore_index=True)
        merged = obs.merge(annual_all, on=["Year", "i", "j"], how="left")
        merged["Year_norm"] = (merged["Year"] - min(YEARS)) / (max(YEARS) - min(YEARS))
        merged.to_parquet(PACKAGED_TRAINING, index=False)
        merged.to_csv(PACKAGED_TRAINING_CSV, index=False)
        print("Rebuilt cascade_training_table.parquet from raw annual SAV observations.")
        return merged

    if PACKAGED_TRAINING.exists():
        print(
            "Raw annual SAV observation workbook not found at "
            f"{TRAIN_OBS}. Using packaged cascade_training_table.parquet instead."
        )
        return pd.read_parquet(PACKAGED_TRAINING)

    raise FileNotFoundError(
        "Could not find either the raw annual SAV workbook "
        f"({TRAIN_OBS}) or the packaged training table ({PACKAGED_TRAINING})."
    )


def main() -> None:
    rebuild_all_annual_grids(overwrite=True)
    training = load_or_build_training_table(overwrite=True)

    print("\nCascade annual input construction completed.")
    print(f"Annual grids: {GRID_DIR}")
    print(f"Training table: {PACKAGED_TRAINING}")
    print(f"Training table CSV: {PACKAGED_TRAINING_CSV}")
    print(f"Training rows: {len(training):,}")


if __name__ == "__main__":
    main()
