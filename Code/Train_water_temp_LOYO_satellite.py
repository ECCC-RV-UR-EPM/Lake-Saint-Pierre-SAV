# -*- coding: utf-8 -*-
import os
HERE = os.path.dirname(os.path.abspath(__file__))
import numpy as np
import pandas as pd
import joblib

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================
# PATH
# ============================================
data_file = os.path.abspath(os.path.join(HERE, "..", "Data", "01_temperature", "Temp_prediction_LSP", "Temp_training_daily_observe_to_train_clip_filled.xlsx"))

out_dir = os.path.join(os.path.dirname(data_file), "Train_results_residual_split")
os.makedirs(out_dir, exist_ok=True)

# ============================================
# ============================================
print(" Loading dataset...")
df = pd.read_excel(data_file)

df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
df = df.dropna(subset=["Date"])

# ============================================
# ============================================
df["DOY"] = df["Date"].dt.dayofyear
df["Month"] = df["Date"].dt.month
df["Year"] = df["Date"].dt.year

df["sin_DOY"] = np.sin(2*np.pi*df["DOY"]/365)
df["cos_DOY"] = np.cos(2*np.pi*df["DOY"]/365)

# ============================================
# ============================================
df["Satellite_temp_raw"] = df["Satellite_temp"].copy()

# ============================================
# ============================================
df["is_channel"] = (df["Bathymetry_depth"] <= -8).astype(int)
df["depth_inv"] = 1 / (np.abs(df["Water_depth"]) + 1)

# ============================================
# ============================================
df["flow_effect"] = (
    (df["In_discharge_GreatLakes"] + df["In_discharge_OttawaRiver"])
    / (np.abs(df["Water_depth"]) + 1)
)

# ============================================
# ============================================
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

df["depth_x_dist"] = df["Water_depth"] * df["dist_to_channel_norm"]

# ============================================
# ============================================
df["channel_cooling"] = (
    df["flow_effect"] * df["dist_to_channel_norm"]
)

# ============================================
# ============================================
df["Residual"] = df["Temp_observation"] - df["Satellite_temp_raw"]

print("\nBefore cleaning:")
print(df["Residual"].describe())

# ============================================
# ============================================

df = df[np.isfinite(df["Residual"])]

df = df[(df["Residual"] > -10) & (df["Residual"] < 10)]

print("\nAfter cleaning:")
print(df["Residual"].describe())

# ============================================
# ============================================
features = [
    "i","j",
    "sin_DOY","cos_DOY","Month",
    "Air_temp",
    "Water_depth","depth_inv",
    "flow_effect",
    "BLOCKS","BOULDERS","COBBLES","GRAVEL","SAND","SILT","CLAY",
    "BLOCKSIZE","BOULDERSIZE","COBBLESIZE","GRAVELSIZE","SANDSIZE","SILTSIZE","CLAYSIZE",
    "In_discharge_GreatLakes","In_discharge_OttawaRiver",
    "In_discharge_RichelieuRiver","In_discharge_YamaskaRiver",
    "In_discharge_Saint_FrancoisRiver","In_discharge_NicoletRiver",
    "In_discharge_MaskinongeRiver","In_discharge_DuLoupRiver","In_discharge_YamachicheRiver",
    "In_distance_GreatLakes","In_distance_OttawaRiver",
    "In_distance_RichelieuRiver","In_distance_YamaskaRiver",
    "In_distance_Saint_FrancoisRiver","In_distance_NicoletRiver",
    "In_distance_MaskinongeRiver","In_distance_DuLoupRiver","In_distance_YamachicheRiver",
    "dist_to_channel","dist_to_channel_norm",
    "depth_x_dist",
    "channel_cooling"
]

df[features] = df[features].astype(np.float32)
df["Residual"] = df["Residual"].astype(np.float32)

# ============================================
# ============================================
df_channel = df[df["is_channel"]==1].copy()
df_nonchannel = df[df["is_channel"]==0].copy()

def train_model(df_sub, name):

    print(f"\n Training {name}")
    metrics_list = []  #

    X_all = df_sub[features].values
    Y_all = df_sub["Residual"].values

    years = np.sort(df_sub["Year"].unique())
    all_rows = np.arange(len(df_sub))

    global_med = np.nanmedian(X_all, axis=0)

    for test_year in years:

        print(f"\n--- {name} | Test Year {test_year}")

        test_idx_main = np.where(df_sub["Year"]==test_year)[0]
        train_idx = np.setdiff1d(all_rows, test_idx_main)

        Xtrain = X_all[train_idx]
        Ytrain = Y_all[train_idx]
        Xtest = X_all[test_idx_main]
        Ytest = Y_all[test_idx_main]

        Xtrain = np.where(np.isfinite(Xtrain), Xtrain, global_med)
        Xtest  = np.where(np.isfinite(Xtest), Xtest, global_med)

        model = XGBRegressor(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            random_state=42,
            n_jobs=-1
        )

        model.fit(Xtrain, Ytrain)

        pred = model.predict(Xtest)

        rmse = np.sqrt(mean_squared_error(Ytest, pred))
        mae = mean_absolute_error(Ytest, pred)
        r2 = r2_score(Ytest, pred)

        print(f"{name} RMSE={rmse:.3f} | MAE={mae:.3f} | R2={r2:.3f}")

        metrics_list.append({
            "Year": test_year,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        })

    # ======================================
    # ======================================
    df_metrics = pd.DataFrame(metrics_list)

    out_csv = os.path.join(out_dir, f"metrics_{name}.csv")
    df_metrics.to_csv(out_csv, index=False)

    print(f" Saved metrics: {out_csv}")

    joblib.dump(model, os.path.join(out_dir, f"model_{name}.pkl"))

    return model

# ============================================
# ============================================
model_channel = train_model(df_channel, "channel")
model_nonchannel = train_model(df_nonchannel, "nonchannel")


# ============================================================
# ============================================================
print("\n Training FINAL models on ALL data...")

# ========= channel =========
model_channel_final = XGBRegressor(
    n_estimators=120,
    max_depth=4,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    random_state=42,
    n_jobs=-1
)

Xc = df_channel[features].values
Yc = df_channel["Residual"].values

med_c = np.nanmedian(Xc, axis=0)
Xc = np.where(np.isfinite(Xc), Xc, med_c)

model_channel_final.fit(Xc, Yc)

joblib.dump(
    model_channel_final,
    os.path.join(out_dir, "model_channel.pkl")
)

print(" Saved FINAL channel model")


# ========= non-channel =========
model_nonchannel_final = XGBRegressor(
    n_estimators=120,
    max_depth=4,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    random_state=42,
    n_jobs=-1
)

Xn = df_nonchannel[features].values
Yn = df_nonchannel["Residual"].values

med_n = np.nanmedian(Xn, axis=0)
Xn = np.where(np.isfinite(Xn), Xn, med_n)

model_nonchannel_final.fit(Xn, Yn)

joblib.dump(
    model_nonchannel_final,
    os.path.join(out_dir, "model_nonchannel.pkl")
)

print(" Saved FINAL non-channel model")

print("\n=================================")
print(" DONE (Residual Model Stable)")
print("=================================")
