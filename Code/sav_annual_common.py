from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.path import Path as MplPath
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parent
DATA_ROOT = PACKAGE_ROOT / "Data"
PIPELINE_ROOT = DATA_ROOT / "04_sav_annual"
SRC = PIPELINE_ROOT / "core_data"
GRID_SRC = SRC / "annual_grids"
OUT_DIR = PIPELINE_ROOT / "results"
MAP_DIR = OUT_DIR / "maps"
SCENARIO_MAP_DIR = OUT_DIR / "scenario_maps"
MODEL_BUNDLE = OUT_DIR / "cascade_v3_model_bundle.pkl"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MAP_DIR.mkdir(parents=True, exist_ok=True)
SCENARIO_MAP_DIR.mkdir(parents=True, exist_ok=True)

NORTH_POLY_FILE = DATA_ROOT / "05_spatial_constraints" / "north_polygon.json"
FIX_FILE = DATA_ROOT / "05_spatial_constraints" / "Lakesides_mainchannel_southarea.xlsx"
GRID_KM2 = 0.04
YEARS = list(range(2002, 2025))
FEATURES = [
    "Water_temp",
    "Water_depth",
    "TSS_pred",
    "TP_pred",
    "TN_pred",
    "Year_norm",
    "north_flag",
    "year_x_north",
    "tss_x_north",
    "depth_x_north",
    "temp_x_north",
    "tp_x_north",
    "tn_x_north",
    "tss_x_year",
]


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


def train_baseline_model() -> tuple[RandomForestClassifier, SimpleImputer, float, pd.DataFrame]:
    train = pd.read_parquet(SRC / "cascade_training_table.parquet").copy()
    train = add_model_features(train, FEATURES)
    years = sorted(train["Year"].dropna().astype(int).unique())
    probs = np.full(len(train), np.nan)
    rows = []

    for yr in years:
        tr = train["Year"] != yr
        te = ~tr
        imp = SimpleImputer(strategy="median")
        xtr = imp.fit_transform(train.loc[tr, FEATURES])
        xte = imp.transform(train.loc[te, FEATURES])
        ytr = train.loc[tr, "SAV"].astype(int).to_numpy()
        yte = train.loc[te, "SAV"].astype(int).to_numpy()

        clf = RandomForestClassifier(
            n_estimators=320,
            random_state=42,
            class_weight="balanced_subsample",
            n_jobs=1,
        )
        clf.fit(xtr, ytr)
        p = clf.predict_proba(xte)[:, 1]
        probs[np.where(te.to_numpy())[0]] = p

        thr_fold = threshold_from_youden(yte, p) if len(np.unique(yte)) > 1 else 0.5
        pred = (p >= thr_fold).astype(int)
        tn, fp, fn, tp = confusion_matrix(yte, pred, labels=[0, 1]).ravel()

        rows.append(
            {
                "Year": int(yr),
                "Accuracy": accuracy_score(yte, pred),
                "BAcc": balanced_accuracy_score(yte, pred),
                "Precision": precision_score(yte, pred, zero_division=0),
                "Recall": recall_score(yte, pred, zero_division=0),
                "Specificity": tn / (tn + fp) if (tn + fp) > 0 else np.nan,
                "F1": f1_score(yte, pred, zero_division=0),
                "ROC_AUC": roc_auc_score(yte, p) if len(np.unique(yte)) > 1 else np.nan,
                "PR_AUC": average_precision_score(yte, p) if len(np.unique(yte)) > 1 else np.nan,
                "Threshold_fold": thr_fold,
            }
        )

    thr = threshold_from_youden(train["SAV"].astype(int).to_numpy(), probs)
    imp = SimpleImputer(strategy="median")
    x = imp.fit_transform(train[FEATURES])
    y = train["SAV"].astype(int).to_numpy()
    clf = RandomForestClassifier(
        n_estimators=320,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=1,
    )
    clf.fit(x, y)
    return clf, imp, thr, pd.DataFrame(rows)


def load_model_bundle() -> tuple[RandomForestClassifier, SimpleImputer, float, list[str]]:
    with MODEL_BUNDLE.open("rb") as f:
        bundle = pickle.load(f)
    features = bundle.get("features", FEATURES)
    return bundle["model"], bundle["imputer"], float(bundle["threshold"]), features


def add_model_features(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    out = df.copy()
    if "Year_norm" not in out.columns:
        out["Year_norm"] = (out["Year"] - min(YEARS)) / (max(YEARS) - min(YEARS))

    need_north = any(
        name in feature_names
        for name in [
            "north_flag",
            "year_x_north",
            "tss_x_north",
            "depth_x_north",
            "temp_x_north",
            "tp_x_north",
            "tn_x_north",
        ]
    )
    if need_north:
        poly = north_poly()
        out["north_flag"] = poly.contains_points(out[["j", "i"]].to_numpy()).astype(int)

    if "year_x_north" in feature_names:
        out["year_x_north"] = out["Year_norm"] * out["north_flag"]
    if "tss_x_north" in feature_names:
        out["tss_x_north"] = out["TSS_pred"] * out["north_flag"]
    if "depth_x_north" in feature_names:
        out["depth_x_north"] = out["Water_depth"] * out["north_flag"]
    if "temp_x_north" in feature_names:
        out["temp_x_north"] = out["Water_temp"] * out["north_flag"]
    if "tp_x_north" in feature_names:
        out["tp_x_north"] = out["TP_pred"] * out["north_flag"]
    if "tn_x_north" in feature_names:
        out["tn_x_north"] = out["TN_pred"] * out["north_flag"]
    if "tss_x_year" in feature_names:
        out["tss_x_year"] = out["TSS_pred"] * out["Year_norm"]

    return out


def apply_constraints(df: pd.DataFrame) -> pd.DataFrame:
    fix = load_fix_constraints()
    out = df.merge(fix, on=["i", "j"], how="left")
    has_fix = out["fix"].notna()
    out.loc[has_fix, "binary"] = out.loc[has_fix, "fix"].astype(int)
    deep = pd.to_numeric(out["Bathymetry_depth"], errors="coerce") <= -8
    out.loc[deep.fillna(False), "binary"] = 0
    return out.drop(columns=["fix"])


def annual_prediction(clf, imp, thr, annual: pd.DataFrame, feature_names: list[str] | None = None) -> pd.DataFrame:
    feature_names = feature_names or FEATURES
    annual = add_model_features(annual, feature_names)
    xp = imp.transform(annual[feature_names])
    p = clf.predict_proba(xp)[:, 1]
    out = annual[["i", "j", "Year", "Bathymetry_depth"]].copy()
    out["prob"] = p
    out["binary"] = (p >= thr).astype(int)
    return apply_constraints(out)


def summarize_maps(maps: dict[int, pd.DataFrame], scenario: str | None = None) -> pd.DataFrame:
    poly = north_poly()
    rows = []
    for year in sorted(maps):
        annual = maps[year]
        mask = poly.contains_points(annual[["j", "i"]].to_numpy())
        row = {
            "Year": int(year),
            "Total_cells": int(annual["binary"].sum()),
            "North_cells": int(annual.loc[mask, "binary"].sum()),
            "Total_km2": float(annual["binary"].sum() * GRID_KM2),
            "North_km2": float(annual.loc[mask, "binary"].sum() * GRID_KM2),
        }
        if scenario is not None:
            row["Scenario"] = scenario
        rows.append(row)
    cols = ["Scenario", "Year", "Total_cells", "North_cells", "Total_km2", "North_km2"] if scenario is not None else ["Year", "Total_cells", "North_cells", "Total_km2", "North_km2"]
    return pd.DataFrame(rows)[cols]


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


def save_feature_importance(clf, out_path: Path) -> None:
    pd.Series(clf.feature_importances_, index=FEATURES).sort_values(ascending=False).to_csv(
        out_path,
        header=["importance"],
    )


def run_scenarios(variable: str, scenarios: dict[str, float]) -> tuple[pd.DataFrame, dict[str, dict[int, pd.DataFrame]]]:
    clf, imp, thr, feature_names = load_model_bundle()
    area_frames = []
    all_maps: dict[str, dict[int, pd.DataFrame]] = {}
    for name, factor in scenarios.items():
        print("Running", name, factor)
        maps: dict[int, pd.DataFrame] = {}
        scenario_dir = SCENARIO_MAP_DIR / name
        scenario_dir.mkdir(parents=True, exist_ok=True)
        for year in YEARS:
            annual = pd.read_parquet(GRID_SRC / f"cascade_grid_{year}.parquet")
            annual[variable] = annual[variable] * factor
            maps[year] = annual_prediction(clf, imp, thr, annual, feature_names)
            maps[year].to_parquet(scenario_dir / f"{name}_{year}.parquet", index=False)
        area_frames.append(summarize_maps(maps, name))
        all_maps[name] = maps
    return pd.concat(area_frames, ignore_index=True), all_maps


def run_custom_scenarios(scenario_transforms: dict) -> tuple[pd.DataFrame, dict[str, dict[int, pd.DataFrame]]]:
    clf, imp, thr, feature_names = load_model_bundle()
    area_frames = []
    all_maps: dict[str, dict[int, pd.DataFrame]] = {}
    for name, transform in scenario_transforms.items():
        print("Running", name)
        maps: dict[int, pd.DataFrame] = {}
        scenario_dir = SCENARIO_MAP_DIR / name
        scenario_dir.mkdir(parents=True, exist_ok=True)
        for year in YEARS:
            annual = pd.read_parquet(GRID_SRC / f"cascade_grid_{year}.parquet")
            annual = transform(annual.copy())
            maps[year] = annual_prediction(clf, imp, thr, annual, feature_names)
            maps[year].to_parquet(scenario_dir / f"{name}_{year}.parquet", index=False)
        area_frames.append(summarize_maps(maps, name))
        all_maps[name] = maps
    return pd.concat(area_frames, ignore_index=True), all_maps
