"""Assemble annual SAV cascade inputs from packaged daily prediction outputs.

This script rebuilds the annual predictor grids (``cascade_grid_YYYY.parquet``)
from the packaged daily temperature, TSS, TP, TN, and water-depth branches.

If the unpublished raw annual SAV observation workbook
``Data/04_sav_annual/references/All_year_previous_2.xlsx`` is available, the
script can also rebuild the annual SAV training table and optional diagnostic
outputs. If that workbook is not present, the script falls back to the packaged
``cascade_training_table.parquet`` and still allows regeneration of the annual
predictor grids.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.path import Path as MplPath
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, fbeta_score, roc_curve

PKG = Path(__file__).resolve().parents[1]
DATA_DIR = PKG / "Data"
ANNUAL_DIR = DATA_DIR / "04_sav_annual"
CORE_DIR = ANNUAL_DIR / "core_data"
GRID_DIR = CORE_DIR / "annual_grids"
RESULT_DIR = ANNUAL_DIR / "results" / "build_cascade_annual_inputs"
MAP_DIR = RESULT_DIR / "maps"
REF_DIR = ANNUAL_DIR / "references"
SPATIAL_DIR = DATA_DIR / "05_spatial_constraints"

GRID_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
MAP_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_OBS = REF_DIR / "All_year_previous_2.xlsx"
TEMP_DIR = (
    DATA_DIR
    / "01_temperature"
    / "Temp_prediction_LSP"
    / "Prediction"
    / "Temp_prediction_expanded"
    / "Final_Temp_Output_monthly_latlon"
)
TSS_RES_DIR = (
    DATA_DIR
    / "02_tss"
    / "TSS_prediction_LSP"
    / "Prediction"
    / "TSS_prediction_results"
)
TSS_FILL_DIR = (
    DATA_DIR
    / "02_tss"
    / "TSS_prediction_LSP"
    / "Prediction"
    / "TSS_prediction_filled_parquet"
)
TP_RES_DIR = (
    DATA_DIR
    / "03_tp_tn"
    / "TP_prediction_LSP"
    / "Prediction"
    / "TP_prediction_results"
)
TN_RES_DIR = (
    DATA_DIR
    / "03_tp_tn"
    / "TN_prediction_LSP"
    / "Prediction"
    / "TN_prediction_results"
)
NORTH_POLY_FILE = SPATIAL_DIR / "north_polygon.json"
FIX_FILE = SPATIAL_DIR / "Lakesides_mainchannel_southarea.xlsx"
PACKAGED_TRAINING = CORE_DIR / "cascade_training_table.parquet"
PACKAGED_TRAINING_CSV = CORE_DIR / "cascade_training_table.csv"

YEARS = list(range(2002, 2025))
TRAIN_YEARS = [2007, 2012, 2013, 2014, 2015, 2016, 2017, 2019, 2021]
MONTHS = {8, 9}
GRID_KM2 = 0.04
FEATURES = ["Water_temp", "Water_depth", "TSS_pred", "TP_pred", "TN_pred", "Year_norm"]


def north_poly() -> MplPath:
    return MplPath(json.loads(NORTH_POLY_FILE.read_text(encoding="utf-8")))


def load_fix_constraints() -> pd.DataFrame:
    df = pd.read_excel(FIX_FILE)
    lower = {c.lower(): c for c in df.columns}
    veg_col = next(lower[k] for k in ["vegetation", "sav", "veg"] if k in lower)
    out = df[[lower["i"], lower["j"], veg_col]].copy()
    out.columns = ["i", "j", "fix"]
    out = out.dropna()
    out["i"] = out["i"].astype(int)
    out["j"] = out["j"].astype(int)
    if not np.issubdtype(out["fix"].dtype, np.number):
        out["fix"] = out["fix"].astype(str).str.lower().map({"1": 1, "0": 0, "sav": 1, "yes": 1})
    out["fix"] = out["fix"].astype(int)
    return out


def threshold_from_youden(y: np.ndarray, p: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


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
        build_annual_grid(year, overwrite=overwrite)
        print(f"Built annual grid for {year}")


def load_or_build_training_table(overwrite: bool = False) -> pd.DataFrame:
    if TRAIN_OBS.exists():
        if PACKAGED_TRAINING.exists() and not overwrite:
            return pd.read_parquet(PACKAGED_TRAINING)

        obs = pd.read_excel(TRAIN_OBS)[["Year", "i", "j", "SAV"]].copy()
        obs["SAV"] = pd.to_numeric(obs["SAV"], errors="coerce")
        obs = obs[obs["SAV"].isin([0, 1])].copy()

        annuals = [build_annual_grid(year, overwrite=overwrite) for year in TRAIN_YEARS]
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


def train_loyo(df: pd.DataFrame):
    years = sorted(df["Year"].unique())
    probs = np.full(len(df), np.nan)
    rows = []
    for yr in years:
        tr = df["Year"] != yr
        te = ~tr
        imp = SimpleImputer(strategy="median")
        xtr = imp.fit_transform(df.loc[tr, FEATURES])
        xte = imp.transform(df.loc[te, FEATURES])
        ytr = df.loc[tr, "SAV"].astype(int).to_numpy()
        yte = df.loc[te, "SAV"].astype(int).to_numpy()
        clf = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced_subsample",
            n_jobs=1,
        )
        clf.fit(xtr, ytr)
        p = clf.predict_proba(xte)[:, 1]
        probs[np.where(te.to_numpy())[0]] = p
        thr_fold = threshold_from_youden(yte, p) if len(np.unique(yte)) > 1 else 0.5
        pred = (p >= thr_fold).astype(int)
        rows.append(
            {
                "Year": yr,
                "Accuracy": accuracy_score(yte, pred),
                "BAcc": balanced_accuracy_score(yte, pred),
                "F2": fbeta_score(yte, pred, beta=2, zero_division=0),
                "Threshold_fold": thr_fold,
            }
        )
        print(f"LOYO {yr}: BAcc={rows[-1]['BAcc']:.3f}, F2={rows[-1]['F2']:.3f}")

    thr = threshold_from_youden(df["SAV"].astype(int).to_numpy(), probs)
    imp = SimpleImputer(strategy="median")
    x = imp.fit_transform(df[FEATURES])
    y = df["SAV"].astype(int).to_numpy()
    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=1,
    )
    clf.fit(x, y)

    pd.DataFrame(rows).to_csv(RESULT_DIR / "cascade_loyo_per_year.csv", index=False)
    pd.Series(clf.feature_importances_, index=FEATURES).sort_values(ascending=False).to_csv(
        RESULT_DIR / "cascade_feature_importance.csv", header=["importance"]
    )
    return clf, imp, thr


def apply_constraints(df: pd.DataFrame) -> pd.DataFrame:
    fix = load_fix_constraints()
    out = df.merge(fix, on=["i", "j"], how="left")
    has_fix = out["fix"].notna()
    out.loc[has_fix, "binary"] = out.loc[has_fix, "fix"].astype(int)
    deep = pd.to_numeric(out["Bathymetry_depth"], errors="coerce") <= -8
    out.loc[deep.fillna(False), "binary"] = 0
    return out.drop(columns=["fix"])


def predict_baseline(clf, imp, thr):
    poly = north_poly()
    rows = []
    maps = {}
    for year in YEARS:
        annual = build_annual_grid(year)
        annual["Year_norm"] = (annual["Year"] - min(YEARS)) / (max(YEARS) - min(YEARS))
        x = imp.transform(annual[FEATURES])
        p = clf.predict_proba(x)[:, 1]
        out = annual[["i", "j", "Year", "Bathymetry_depth"]].copy()
        out["prob"] = p
        out["binary"] = (p >= thr).astype(int)
        out = apply_constraints(out)
        mask = poly.contains_points(out[["j", "i"]].to_numpy())
        rows.append(
            {
                "Year": year,
                "Total_cells": int(out["binary"].sum()),
                "North_cells": int(out.loc[mask, "binary"].sum()),
                "Total_km2": float(out["binary"].sum() * GRID_KM2),
                "North_km2": float(out.loc[mask, "binary"].sum() * GRID_KM2),
            }
        )
        maps[year] = out

    pd.DataFrame(rows).to_csv(RESULT_DIR / "cascade_baseline_north_timeseries.csv", index=False)
    return pd.DataFrame(rows), maps


def panel(maps: dict[int, pd.DataFrame], out_path: Path, title: str) -> None:
    years = sorted(maps)
    cmap = ListedColormap(["#1565c0", "#17a34a"])
    ncol = 6
    nrow = int(np.ceil(len(years) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(18, 10))
    axes = np.atleast_1d(axes).ravel()
    for ax, yr in zip(axes, years):
        annual = maps[yr]
        ax.scatter(annual["j"], annual["i"], c=annual["binary"], s=1, cmap=cmap, vmin=0, vmax=1)
        ax.invert_yaxis()
        ax.axis("off")
        ax.set_title(str(yr), fontsize=8)
    for ax in axes[len(years):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    rebuild_all_annual_grids(overwrite=False)
    train = load_or_build_training_table(overwrite=False)
    clf, imp, thr = train_loyo(train)
    _, maps = predict_baseline(clf, imp, thr)
    panel(maps, MAP_DIR / "SAV_ALL_YEARS_cascade_baseline.png", "Cascade-only SAV baseline")


if __name__ == "__main__":
    main()
