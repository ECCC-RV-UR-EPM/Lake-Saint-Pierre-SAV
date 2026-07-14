# -*- coding: utf-8 -*-
"""Train the final daily TN models with LOYO validation."""

import os
HERE = os.path.dirname(os.path.abspath(__file__))
import numpy as np
import pandas as pd
import joblib

from xgboost import XGBRegressor
from sklearn.inspection import permutation_importance

# ============================================================
# PATH
# ============================================================
DATA_FILE = os.path.abspath(os.path.join(HERE, "..", "Data", "03_tp_tn", "TN_prediction_LSP", "TN_training_daily_filled.xlsx"))
OUTPUT_DIR = os.path.join(os.path.dirname(DATA_FILE), "Final_Model_TN_noTotal")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LOAD
# ============================================================
print("Reading data...")
df = pd.read_excel(DATA_FILE)

df["Date"] = pd.to_datetime(df["Date"])
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month

# ============================================================
# ============================================================
df["DOY"] = df["Date"].dt.dayofyear
df["sin_DOY"] = np.sin(2*np.pi*df["DOY"]/365)
df["cos_DOY"] = np.cos(2*np.pi*df["DOY"]/365)

year_min = df["Year"].min()
year_max = df["Year"].max()

df["Year_norm"] = (df["Year"] - year_min) / (year_max - year_min)

# ============================================================
# ============================================================

if "Bathymetry_depth" in df.columns:
    df["depth_inv"] = 1 / (np.abs(df["Bathymetry_depth"]) + 1)
else:
    raise ValueError("  Bathymetry_depth  hydrodynamic features")

df["flow_effect"] = (
    (df["In_discharge_GreatLakes"] + df["In_discharge_OttawaRiver"])
    / (np.abs(df["Bathymetry_depth"]) + 1)
)

from scipy.spatial import cKDTree

channel_mask = df["Bathymetry_depth"] <= -8
channel_points = df.loc[channel_mask, ["i","j"]].values.astype(np.float32)
all_points = df[["i","j"]].values.astype(np.float32)

tree = cKDTree(channel_points)
dist, _ = tree.query(all_points, k=1)

df["dist_to_channel"] = dist.astype(np.float32)

df["dist_to_channel_norm"] = (
    df["dist_to_channel"] / (df["dist_to_channel"].max() + 1e-6)
).astype(np.float32)

# ============================================================
# ============================================================
df["is_channel"] = (df["Bathymetry_depth"] <= -8).astype(int)

# ============================================================
# ============================================================
FEATURES = [
    "i","j","sin_DOY","cos_DOY","Month",

    "Water_temp","Water_depth",
    "TSS_LSP",

    "Bathymetry_depth","depth_inv","flow_effect",
    "dist_to_channel","dist_to_channel_norm",

    "In_distance_GreatLakes","In_distance_OttawaRiver","In_distance_RichelieuRiver",
    "In_distance_YamaskaRiver","In_distance_Saint_FrancoisRiver",
    "In_distance_NicoletRiver","In_distance_MaskinongeRiver",
    "In_distance_DuLoupRiver","In_distance_YamachicheRiver",

    "In_discharge_GreatLakes","In_discharge_OttawaRiver","In_discharge_RichelieuRiver",
    "In_discharge_YamaskaRiver","In_discharge_Saint_FrancoisRiver",
    "In_discharge_NicoletRiver","In_discharge_MaskinongeRiver",
    "In_discharge_DuLoupRiver","In_discharge_YamachicheRiver",

    "In_TSS_GreatLakes","In_TSS_OttawaRiver","In_TSS_RichelieuRiver",
    "In_TSS_YamaskaRiver","In_TSS_Saint_FrancoisRiver",
    "In_TSS_NicoletRiver","In_TSS_MaskinongeRiver",
    "In_TSS_DuLoupRiver","In_TSS_YamachicheRiver",

    "In_TN_GreatLakes", "In_TN_OttawaRiver", "In_TN_RichelieuRiver",
    "In_TN_YamaskaRiver", "In_TN_Saint_FrancoisRiver",
    "In_TN_NicoletRiver", "In_TN_MaskinongeRiver",
    "In_TN_DuLoupRiver", "In_TN_YamachicheRiver",

    "BLOCKS","BOULDERS","COBBLES","GRAVEL","SAND","SILT","CLAY",
    "BLOCKSIZE","BOULDERSIZE","COBBLESIZE","GRAVELSIZE",
    "SANDSIZE","SILTSIZE","CLAYSIZE"
]

TARGET = "TN_observation"

# ============================================================
# CLEAN
# ============================================================
df = df[df[TARGET].notna()]

missing = [c for c in FEATURES if c not in df.columns]
if len(missing) > 0:
    print(" ", missing)
    raise ValueError("Feature")

# ============================================================
# DATA
# ============================================================
X = df[FEATURES].to_numpy()
Y_raw = df[TARGET].to_numpy()

Y_raw = Y_raw.clip(0, np.percentile(Y_raw, 99))
Y = np.log1p(Y_raw)

# ============================================================
# IMPUTE
# ============================================================
med = np.nanmedian(X, axis=0)
med = np.where(np.isfinite(med), med, 0)

mask = ~np.isfinite(X)
X[mask] = med[np.where(mask)[1]]

# ============================================================
# MODEL
# ============================================================
print("Training XGBoost model (LOYO + channel)...")

years = np.sort(df["Year"].unique())

models_channel = []
models_nonchannel = []

rmse_all = []
mae_all = []
r2_all = []

for y in years:
    print(f"\n LOYO year: {y}")

    train_idx = df["Year"] != y
    test_idx  = df["Year"] == y

    # ========= channel =========
    train_ch = train_idx & (df["is_channel"] == 1)
    test_ch  = test_idx  & (df["is_channel"] == 1)

    if train_ch.sum() > 0 and test_ch.sum() > 0:

        X_tr = X[train_ch]
        Y_tr = Y[train_ch]

        X_te = X[test_ch]
        Y_te_raw = Y_raw[test_ch]

        model_ch = XGBRegressor(
            n_estimators=900,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=2,
            reg_lambda=2,
            n_jobs=-1,
            random_state=42
        )

        model_ch.fit(X_tr, Y_tr)

        Y_pred = np.expm1(model_ch.predict(X_te))

        rmse = np.sqrt(np.mean((Y_pred - Y_te_raw)**2))
        mae  = np.mean(np.abs(Y_pred - Y_te_raw))

        ss_res = np.sum((Y_te_raw - Y_pred) ** 2)
        ss_tot = np.sum((Y_te_raw - np.mean(Y_te_raw)) ** 2)

        if ss_tot == 0:
            r2 = np.nan
        else:
            r2 = 1 - ss_res / ss_tot

        rmse_all.append(rmse)
        mae_all.append(mae)
        r2_all.append(r2)

        models_channel.append(model_ch)

    # ========= non-channel =========
    train_nc = train_idx & (df["is_channel"] == 0)
    test_nc  = test_idx  & (df["is_channel"] == 0)

    if train_nc.sum() > 0 and test_nc.sum() > 0:

        X_tr = X[train_nc]
        Y_tr = Y[train_nc]

        X_te = X[test_nc]
        Y_te_raw = Y_raw[test_nc]

        model_nc = XGBRegressor(
            n_estimators=900,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=2,
            reg_lambda=2,
            n_jobs=-1,
            random_state=42
        )

        model_nc.fit(X_tr, Y_tr)

        Y_pred = np.expm1(model_nc.predict(X_te))

        rmse = np.sqrt(np.mean((Y_pred - Y_te_raw)**2))
        mae  = np.mean(np.abs(Y_pred - Y_te_raw))


        ss_res = np.sum((Y_te_raw - Y_pred) ** 2)
        ss_tot = np.sum((Y_te_raw - np.mean(Y_te_raw)) ** 2)

        if ss_tot == 0:
            r2 = np.nan
        else:
            r2 = 1 - ss_res / ss_tot

        rmse_all.append(rmse)
        mae_all.append(mae)
        r2_all.append(r2)

        models_nonchannel.append(model_nc)

# ============================================================
# ============================================================
print("\n===== LOYO Performance =====")
print(f"RMSE: {np.mean(rmse_all):.3f}")
print(f"MAE : {np.mean(mae_all):.3f}")
print(f"R2  : {np.mean(r2_all):.3f}")

# ============================================================
# ============================================================
print("\n Training FINAL model on ALL data...")

# ========= channel =========
mask_c = df["is_channel"] == 1

model_channel_final = XGBRegressor(
    n_estimators=900,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=2,
    reg_lambda=2,
    n_jobs=-1,
    random_state=42
)

model_channel_final.fit(X[mask_c], Y[mask_c])


# ========= non-channel =========
mask_n = df["is_channel"] == 0

model_nonchannel_final = XGBRegressor(
    n_estimators=900,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=2,
    reg_lambda=2,
    n_jobs=-1,
    random_state=42
)

model_nonchannel_final.fit(X[mask_n], Y[mask_n])

# ============================================================
# PERFORMANCE
# ============================================================
print("Evaluating model...")

# ============================================================
# SAVE
# ============================================================
# ============================================================
# ============================================================

model_channel_path = os.path.join(OUTPUT_DIR, "model_channel.pkl")
model_nonchannel_path = os.path.join(OUTPUT_DIR, "model_nonchannel.pkl")

# ========= channel =========
joblib.dump({
    "model": model_channel_final,
    "features": FEATURES,
    "median": med,
    "year_min": year_min,
    "year_max": year_max
}, model_channel_path)

# ========= non-channel =========
joblib.dump({
    "model": model_nonchannel_final,
    "features": FEATURES,
    "median": med,
    "year_min": year_min,
    "year_max": year_max
}, model_nonchannel_path)

print(" Saved channel model:", model_channel_path)
print(" Saved non-channel model:", model_nonchannel_path)

print("\n FINAL TN MODEL READY")
