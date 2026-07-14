import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


HERE = Path(__file__).resolve().parent
BASE_DIR = HERE.parent / "Data" / "01_temperature" / "Temp_prediction_LSP"

INPUT_FILE = BASE_DIR / "Temp_training_daily_observe_to_train_clip.xlsx"
OUTPUT_FILE = BASE_DIR / "Temp_training_daily_observe_to_train_clip_filled.xlsx"
COMBINE_FILE = BASE_DIR / "Combine.xlsx"
GRID_FILE = BASE_DIR / "lat_lon_UMT_i_j.csv"
JOINED_FILE = BASE_DIR / "JOINED_with_ij.csv"
SAT_FILE = BASE_DIR / "LSP_LSWT_200m_rev02_2000-2024_satellite.csv"


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = (
        out.columns.astype(str)
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
    )
    return out


def normalize_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.normalize()


def load_satellite_table(path: Path) -> pd.DataFrame:
    sat = pd.read_csv(path)
    sat.columns = sat.columns.str.strip()

    i_col = next((c for c in sat.columns if c.lower() in {"i", "i_index"}), None)
    j_col = next((c for c in sat.columns if c.lower() in {"j", "j_index"}), None)
    date_col = next((c for c in sat.columns if "date" in c.lower()), None)
    temp_col = next((c for c in sat.columns if "temp" in c.lower()), None)

    if not all([i_col, j_col, date_col, temp_col]):
        raise ValueError("Satellite file is missing one or more required columns: i, j, date, temperature")

    sat = sat[[i_col, j_col, date_col, temp_col]].copy()
    sat.columns = ["i", "j", "Date", "Satellite_temp"]
    sat["Date"] = normalize_date(sat["Date"])
    sat = sat.dropna(subset=["Date", "Satellite_temp"])
    sat["i"] = sat["i"].astype(int)
    sat["j"] = sat["j"].astype(int)
    return sat


def main() -> None:
    print("Loading clipped temperature observations...")
    df = pd.read_excel(INPUT_FILE)
    original_cols = list(df.columns)
    df = clean_columns(df)
    df["Date"] = normalize_date(df["Date"])

    print("Loading grid...")
    grid = clean_columns(pd.read_csv(GRID_FILE))
    grid["i"] = grid["i"].astype(int)
    grid["j"] = grid["j"].astype(int)

    print("Matching Longitude/Latitude to grid i,j...")
    tree_ll = cKDTree(grid[["lat", "lon"]].to_numpy())
    _, idx = tree_ll.query(df[["Latitude", "Longitude"]].to_numpy())
    df["i"] = grid.iloc[idx]["i"].to_numpy(dtype=int)
    df["j"] = grid.iloc[idx]["j"].to_numpy(dtype=int)

    print("Filling Combine variables...")
    combine = clean_columns(pd.read_excel(COMBINE_FILE))
    combine["Date"] = normalize_date(combine["Date"])
    combine = combine.set_index("Date")
    combine_cols = [c for c in combine.columns if c != "Date"]
    for col in combine_cols:
        df[col] = df["Date"].map(combine[col])

    print("Adding bathymetry and grid coordinates...")
    key_grid = grid["i"] * 1000 + grid["j"]
    key_df = df["i"] * 1000 + df["j"]
    loc = pd.Series(np.arange(len(grid)), index=key_grid)
    idx_match = key_df.map(loc)

    df["Bathymetry_depth"] = np.nan
    valid = idx_match.notna()
    if valid.any():
        matched = idx_match[valid].astype(int).to_numpy()
        df.loc[valid, "Bathymetry_depth"] = grid.iloc[matched]["depth"].to_numpy()
        df.loc[valid, "lat"] = grid.iloc[matched]["lat"].to_numpy()
        df.loc[valid, "lon"] = grid.iloc[matched]["lon"].to_numpy()
        df.loc[valid, "depth"] = grid.iloc[matched]["depth"].to_numpy()
        # Preserve the legacy typo present in the historical filled file for backward compatibility.
        df.loc[valid, "dpeth"] = grid.iloc[matched]["depth"].to_numpy()

    if "Water_elevation" in df.columns:
        df["Water_depth"] = df["Water_elevation"] - df["Bathymetry_depth"]

    print("Adding substrate variables...")
    joined = clean_columns(pd.read_csv(JOINED_FILE))
    joined["i"] = joined["i"].astype(int)
    joined["j"] = joined["j"].astype(int)
    soil_cols = [
        c for c in joined.columns
        if any(k in c.upper() for k in [
            "BLOCK", "BOULDER", "COBBLE", "GRAVEL", "SAND", "SILT", "CLAY"
        ])
    ]
    joined_small = joined[["i", "j"] + soil_cols].drop_duplicates(subset=["i", "j"])
    df = df.merge(joined_small, on=["i", "j"], how="left", suffixes=("", "_soil"))
    for col in soil_cols:
        soil_col = f"{col}_soil"
        if soil_col in df.columns:
            df[col] = df[soil_col]
            df = df.drop(columns=[soil_col])

    print("Joining satellite temperature...")
    sat = load_satellite_table(SAT_FILE)
    df = df.drop(columns=["Satellite_temp"], errors="ignore")
    df = df.merge(sat, on=["i", "j", "Date"], how="left")

    preferred_order = original_cols + [
        "In_TP_GreatLakes", "In_TP_OttawaRiver", "In_TP_RichelieuRiver",
        "In_TP_YamaskaRiver", "In_TP_Saint_FrancoisRiver", "In_TP_NicoletRiver",
        "In_TP_MaskinongeRiver", "In_TP_DuLoupRiver", "In_TP_YamachicheRiver",
        "In_TN_GreatLakes", "In_TN_OttawaRiver", "In_TN_RichelieuRiver",
        "In_TN_YamaskaRiver", "In_TN_Saint_FrancoisRiver", "In_TN_NicoletRiver",
        "In_TN_MaskinongeRiver", "In_TN_DuLoupRiver", "In_TN_YamachicheRiver",
        "In_TSS_GreatLakes", "In_TSS_OttawaRiver", "In_TSS_RichelieuRiver",
        "In_TSS_YamaskaRiver", "In_TSS_Saint_FrancoisRiver", "In_TSS_NicoletRiver",
        "In_TSS_MaskinongeRiver", "In_TSS_DuLoupRiver", "In_TSS_YamachicheRiver",
        "Water_temp", "Wind_speed", "Wind_direction", "lat", "lon", "depth", "dpeth",
    ]
    final_cols = [c for c in preferred_order if c in df.columns] + [c for c in df.columns if c not in preferred_order]
    df = df[final_cols]

    print(f"Saving filled table to {OUTPUT_FILE} ...")
    df.to_excel(OUTPUT_FILE, index=False)
    print("DONE")


if __name__ == "__main__":
    main()
