from __future__ import annotations

import pickle
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from sav_annual_common import GRID_SRC, MAP_DIR, OUT_DIR, YEARS, annual_prediction, panel, summarize_maps

HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parent
DATA_ROOT = PACKAGE_ROOT / "Data"
SAV_BUNDLE = DATA_ROOT / "04_sav_annual" / "results" / "cascade_v3_model_bundle.pkl"

NORTH_TRIBUTARY_RIVERS = [
    "OttawaRiver",
    "MaskinongeRiver",
    "DuLoupRiver",
    "YamachicheRiver",
]


def _load_sav_bundle():
    with SAV_BUNDLE.open("rb") as f:
        return pickle.load(f)


def _late_summer_mean(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    work = df.copy()
    work["Date"] = pd.to_datetime(work["Date"])
    work = work[work["Date"].dt.month.isin([8, 9])].copy()
    return work.groupby(["i", "j"], as_index=False)[value_col].mean()


def _keep_late_summer_rows(df: pd.DataFrame) -> pd.DataFrame:
    work = df
    if not np.issubdtype(work["Date"].dtype, np.datetime64):
        work["Date"] = pd.to_datetime(work["Date"])
    return work.loc[work["Date"].dt.month.isin([8, 9])].copy()


def _build_baseline_maps() -> dict[int, pd.DataFrame]:
    bundle = _load_sav_bundle()
    clf = bundle["model"]
    imp = bundle["imputer"]
    thr = bundle["threshold"]
    maps: dict[int, pd.DataFrame] = {}
    for year in YEARS:
        annual = pd.read_parquet(GRID_SRC / f"cascade_grid_{year}.parquet")
        maps[year] = annual_prediction(clf, imp, thr, annual)
    return maps


def _predict_daily_from_inputs(
    df: pd.DataFrame,
    channel_bundle: dict,
    nonchannel_bundle: dict,
    pred_col: str,
) -> pd.DataFrame:
    features = channel_bundle["features"]
    med = channel_bundle["median"]
    model_channel = channel_bundle["model"]
    model_nonchannel = nonchannel_bundle["model"]

    work = df
    work["Date"] = pd.to_datetime(work["Date"])
    work["DOY"] = work["Date"].dt.dayofyear
    work["Month"] = work["Date"].dt.month
    work["sin_DOY"] = np.sin(2 * np.pi * work["DOY"] / 365)
    work["cos_DOY"] = np.cos(2 * np.pi * work["DOY"] / 365)
    work["depth_inv"] = 1 / (np.abs(work["Bathymetry_depth"]) + 1)
    work["flow_effect"] = (
        (work["In_discharge_GreatLakes"] + work["In_discharge_OttawaRiver"])
        / (np.abs(work["Bathymetry_depth"]) + 1)
    )

    channel_mask = work["Bathymetry_depth"] <= -8
    if channel_mask.sum() > 0:
        tree = cKDTree(work.loc[channel_mask, ["i", "j"]].values)
        dist, _ = tree.query(work[["i", "j"]].values)
    else:
        dist = np.zeros(len(work))
    work["dist_to_channel"] = dist
    work["dist_to_channel_norm"] = dist / (dist.max() + 1e-6)
    work["is_channel"] = channel_mask.astype(int)
    work[pred_col] = np.nan

    mask_c = work["is_channel"] == 1
    if mask_c.any():
        xc = work.loc[mask_c, features].to_numpy(dtype=np.float32)
        xc = np.where(np.isfinite(xc), xc, med)
        work.loc[mask_c, pred_col] = np.expm1(model_channel.predict(xc))

    mask_n = work["is_channel"] == 0
    if mask_n.any():
        xn = work.loc[mask_n, features].to_numpy(dtype=np.float32)
        xn = np.where(np.isfinite(xn), xn, med)
        work.loc[mask_n, pred_col] = np.expm1(model_nonchannel.predict(xn))

    return work.loc[:, ["Date", "i", "j", pred_col]].copy()


def _scale_selected_river_columns(df: pd.DataFrame, prefixes: list[str], factor: float) -> pd.DataFrame:
    for prefix in prefixes:
        for river in NORTH_TRIBUTARY_RIVERS:
            col = f"{prefix}{river}"
            if col in df.columns:
                df[col] = df[col] * factor
    return df


def run_north_tributary_scenarios(
    *,
    variable: str,
    pred_col: str,
    filled_input_dir: Path,
    channel_model_path: Path,
    nonchannel_model_path: Path,
    scaled_prefixes: list[str],
    scenarios: dict[str, float],
    output_prefix: str,
) -> None:
    channel_bundle = joblib.load(channel_model_path)
    nonchannel_bundle = joblib.load(nonchannel_model_path)
    sav_bundle = _load_sav_bundle()
    sav_clf = sav_bundle["model"]
    sav_imp = sav_bundle["imputer"]
    sav_thr = sav_bundle["threshold"]

    baseline_maps = _build_baseline_maps()
    area_frames = [summarize_maps(baseline_maps, "baseline")]
    map_sets: dict[str, dict[int, pd.DataFrame]] = {"baseline": baseline_maps}

    for label, factor in scenarios.items():
        yearly_maps: dict[int, pd.DataFrame] = {}
        for year in YEARS:
            input_path = filled_input_dir / f"{variable}_{year}.parquet"
            daily_input = pd.read_parquet(input_path)
            daily_input = _keep_late_summer_rows(daily_input)
            daily_input = _scale_selected_river_columns(daily_input, scaled_prefixes, factor)
            daily_pred = _predict_daily_from_inputs(
                daily_input,
                channel_bundle=channel_bundle,
                nonchannel_bundle=nonchannel_bundle,
                pred_col=pred_col,
            )
            annual_delta = daily_pred.groupby(["i", "j"], as_index=False)[pred_col].mean()
            annual_base = pd.read_parquet(GRID_SRC / f"cascade_grid_{year}.parquet")
            annual = annual_base.merge(annual_delta, on=["i", "j"], how="left", suffixes=("", "_scenario"))
            annual[pred_col] = annual[f"{pred_col}_scenario"].combine_first(annual[pred_col])
            annual = annual.drop(columns=[f"{pred_col}_scenario"])
            yearly_maps[year] = annual_prediction(sav_clf, sav_imp, sav_thr, annual)

        map_sets[label] = yearly_maps
        area_frames.append(summarize_maps(yearly_maps, label))

    area_df = pd.concat(area_frames, ignore_index=True)
    area_df.to_csv(OUT_DIR / f"{output_prefix}_area_summary.csv", index=False)

    base = area_df[area_df["Scenario"] == "baseline"][["Year", "Total_km2", "North_km2"]].rename(
        columns={"Total_km2": "base_total", "North_km2": "base_north"}
    )
    delta_frames = []
    for scen in scenarios:
        sub = area_df[area_df["Scenario"] == scen][["Year", "Total_km2", "North_km2"]].merge(base, on="Year")
        sub["Scenario"] = scen
        sub["Delta_total_km2"] = sub["Total_km2"] - sub["base_total"]
        sub["Delta_north_km2"] = sub["North_km2"] - sub["base_north"]
        delta_frames.append(sub[["Scenario", "Year", "Delta_total_km2", "Delta_north_km2"]])
    pd.concat(delta_frames, ignore_index=True).to_csv(OUT_DIR / f"{output_prefix}_delta_vs_baseline.csv", index=False)

    panel(map_sets["baseline"], MAP_DIR / f"SAV_ALL_YEARS_{output_prefix}_baseline_reference.png", f"Annual SAV {output_prefix} baseline")
    for scen in scenarios:
        panel(map_sets[scen], MAP_DIR / f"SAV_ALL_YEARS_{scen}.png", f"Annual SAV {scen}")

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    for scen in ["baseline", *scenarios.keys()]:
        sub = area_df[area_df["Scenario"] == scen].sort_values("Year")
        axes[0].plot(sub["Year"], sub["Total_km2"], marker="o", label=scen)
        axes[1].plot(sub["Year"], sub["North_km2"], marker="o", label=scen)
    axes[0].set_ylabel("Total SAV area (km$^2$)")
    axes[1].set_ylabel("North SAV area (km$^2$)")
    axes[1].set_xlabel("Year")
    axes[0].grid(True, alpha=0.3)
    axes[1].grid(True, alpha=0.3)
    axes[0].legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{output_prefix}_timeseries.png", dpi=300)
    plt.close(fig)

    print(area_df[area_df["Year"] == 2024].to_string(index=False))
