# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import joblib

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# ============================================
# PATH
# ============================================
data_file = r"E:\SAV_map_PNAS\Submission_package\Data\01_temperature\Temp_training_daily_observe_to_train_clip_filled.xlsx"
grid_file = r"E:\SAV_map_PNAS\Submission_package\Data\01_temperature\lat_lon_UMT_i_j.csv"

out_dir = os.path.join(os.path.dirname(data_file), "Train_results_residual_split")
os.makedirs(out_dir, exist_ok=True)

# ============================================
# 1️⃣ 数据读取
# ============================================
print("📥 Loading dataset...")
df = pd.read_excel(data_file)

df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
df = df.dropna(subset=["Date"])

# ============================================
# 2️⃣ 时间特征
# ============================================
df["DOY"] = df["Date"].dt.dayofyear
df["Month"] = df["Date"].dt.month
df["Year"] = df["Date"].dt.year

df["sin_DOY"] = np.sin(2*np.pi*df["DOY"]/365)
df["cos_DOY"] = np.cos(2*np.pi*df["DOY"]/365)

# ============================================
# ⭐ Satellite（保留原值）
# ============================================
df["Satellite_temp_raw"] = df["Satellite_temp"].copy()

# ============================================
# ⭐ Channel + Depth
# ============================================
df["depth_inv"] = 1 / (np.abs(df["Water_depth"]) + 1)

# ============================================
# ⭐ flow_effect
# ============================================
df["flow_effect"] = (
    (df["In_discharge_GreatLakes"] + df["In_discharge_OttawaRiver"])
    / (np.abs(df["Water_depth"]) + 1)
)

# ============================================
# ⭐ dist_to_channel
# ============================================
from scipy.spatial import cKDTree

grid = pd.read_csv(grid_file, usecols=["i", "j", "depth"]).drop_duplicates(["i", "j"])
grid["is_channel"] = (grid["depth"] <= -8).astype(np.int8)
channel_points = grid.loc[grid["is_channel"] == 1, ["i", "j"]].to_numpy(np.float32)
if len(channel_points) == 0:
    raise ValueError("Static grid contains no channel cells at the -8 m threshold")

tree = cKDTree(channel_points)
grid_dist, _ = tree.query(grid[["i", "j"]].to_numpy(np.float32), k=1)
distance_scale = float(np.max(grid_dist))
if not np.isfinite(distance_scale) or distance_scale <= 0:
    raise ValueError(f"Invalid channel-distance scale: {distance_scale}")

grid["dist_to_channel"] = grid_dist.astype(np.float32)
grid["dist_to_channel_norm"] = (grid_dist / distance_scale).astype(np.float32)

df = df.drop(columns=["is_channel", "dist_to_channel", "dist_to_channel_norm"], errors="ignore")
df = df.merge(
    grid[["i", "j", "is_channel", "dist_to_channel", "dist_to_channel_norm"]],
    on=["i", "j"], how="left", validate="many_to_one",
)
if df[["is_channel", "dist_to_channel", "dist_to_channel_norm"]].isna().any().any():
    raise ValueError("Some training i/j cells were not found in the static grid")

df["depth_x_dist"] = df["Water_depth"] * df["dist_to_channel_norm"]

# ============================================
# ⭐ channel_cooling
# ============================================
df["channel_cooling"] = (
    df["flow_effect"] * df["dist_to_channel_norm"]
)

# ============================================
# 🚀 ⭐ Residual（核心）
# ============================================
df["Residual"] = df["Temp_observation"] - df["Satellite_temp_raw"]

print("\nBefore cleaning:")
print(df["Residual"].describe())

# ============================================
# 🔥 ⭐⭐ FIX：清洗异常值（关键修复）
# ============================================

# 1️⃣ 去除 NaN / inf
df = df[np.isfinite(df["Residual"])]

# 2️⃣ 去掉极端异常（物理不合理）
df = df[(df["Residual"] > -10) & (df["Residual"] < 10)]

print("\nAfter cleaning:")
print(df["Residual"].describe())

# ============================================
# 🚀 Features（不含 Satellite）
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
# 🚀 分模型
# ============================================
df_channel = df[df["is_channel"]==1].copy()
df_nonchannel = df[df["is_channel"]==0].copy()

def train_model(df_sub, name):

    print(f"\n🚀 Training {name}")
    metrics_list = []  # ⭐ 加这一行

    X_all = df_sub[features].values
    Y_all = df_sub["Residual"].values

    years = np.sort(df_sub["Year"].unique())
    all_rows = np.arange(len(df_sub))

    for test_year in years:

        print(f"\n--- {name} | Test Year {test_year}")

        test_idx_main = np.where(df_sub["Year"]==test_year)[0]
        train_idx = np.setdiff1d(all_rows, test_idx_main)

        Xtrain = X_all[train_idx]
        Ytrain = Y_all[train_idx]
        Xtest = X_all[test_idx_main]
        Ytest = Y_all[test_idx_main]

        fold_med = np.nanmedian(Xtrain, axis=0)
        fold_med = np.where(np.isfinite(fold_med), fold_med, 0.0)
        Xtrain = np.where(np.isfinite(Xtrain), Xtrain, fold_med)
        Xtest  = np.where(np.isfinite(Xtest), Xtest, fold_med)

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

        from sklearn.metrics import r2_score
        r2 = r2_score(Ytest, pred)

        print(f"{name} RMSE={rmse:.3f} | R2={r2:.3f}")

        # ⭐ 保存每年结果
        metrics_list.append({
            "Year": test_year,
            "RMSE": rmse,
            "R2": r2
        })

    # ======================================
    # ⭐ 保存 metrics（新增）
    # ======================================
    df_metrics = pd.DataFrame(metrics_list)

    out_csv = os.path.join(out_dir, f"metrics_{name}.csv")
    df_metrics.to_csv(out_csv, index=False)

    print(f"✔ Saved metrics: {out_csv}")

    return model

# ============================================
# 🚀 训练
# ============================================
model_channel = train_model(df_channel, "channel")
model_nonchannel = train_model(df_nonchannel, "nonchannel")


# ============================================================
# 🔥 FINAL MODEL（用于预测，必须新增）
# ============================================================
print("\n🔥 Training FINAL models on ALL data...")

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
med_c = np.where(np.isfinite(med_c), med_c, 0.0).astype(np.float32)
Xc = np.where(np.isfinite(Xc), Xc, med_c)

model_channel_final.fit(Xc, Yc)

joblib.dump(
    model_channel_final,
    os.path.join(out_dir, "model_channel.pkl")
)

print("✔ Saved FINAL channel model")


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
med_n = np.where(np.isfinite(med_n), med_n, 0.0).astype(np.float32)
Xn = np.where(np.isfinite(Xn), Xn, med_n)

model_nonchannel_final.fit(Xn, Yn)

joblib.dump(
    model_nonchannel_final,
    os.path.join(out_dir, "model_nonchannel.pkl")
)

print("✔ Saved FINAL non-channel model")

joblib.dump(
    {
        "features": features,
        "channel_threshold_m": -8.0,
        "distance_scale": distance_scale,
        "impute_median_channel": med_c,
        "impute_median_nonchannel": med_n,
        "grid_file": grid_file,
    },
    os.path.join(out_dir, "temperature_preprocessing.pkl"),
)
print("✔ Saved preprocessing metadata")

print("\n=================================")
print("✅ DONE (Residual Model Stable)")
print("=================================")
