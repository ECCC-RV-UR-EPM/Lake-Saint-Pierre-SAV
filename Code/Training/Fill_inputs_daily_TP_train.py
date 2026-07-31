import pandas as pd
import numpy as np
import os
from scipy.spatial import cKDTree
import glob

# =========================================================
# PATH
# =========================================================
base_dir = r"E:\SAV_map_PNAS\Submission_package\Data\03_tp_tn\TP_prediction_LSP"

tss_file = os.path.join(base_dir, "TP_training_daily.xlsx")
grid_file = os.path.join(base_dir, "lat_lon_UMT_i_j.csv")
combine_file = os.path.join(base_dir, "Combine.xlsx")
joined_file = os.path.join(base_dir, "JOINED_with_ij.csv")

temp_parquet_dir = r"E:\SAV_map_PNAS\Submission_package\Data\01_temperature\Prediction\Final_Temp_Output_monthly_latlon"

out_file = os.path.join(base_dir, "TP_training_daily_filled.xlsx")

# =========================================================
# 工具函数
# =========================================================
def clean_columns(df):
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("(", "")
        .str.replace(")", "")
    )
    return df

# =========================================================
# 1️⃣ Load
# =========================================================
df = pd.read_excel(tss_file)
orig_cols = df.columns.tolist()

df = clean_columns(df)
df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()

print("✔ Load done")

# =========================================================
# 2️⃣ 经纬度 → i,j
# =========================================================
G = pd.read_csv(grid_file)
G = clean_columns(G)

tree = cKDTree(G[["lat","lon"]].values)
_, idx = tree.query(df[["Latitude","Longitude"]].values)

df["i"] = G.iloc[idx]["i"].values.astype(int)
df["j"] = G.iloc[idx]["j"].values.astype(int)

print("✔ i,j assigned")

# =========================================================
# 3️⃣ Combine（时间映射）
# =========================================================
T1 = pd.read_excel(combine_file)
T1 = clean_columns(T1)
T1["Date"] = pd.to_datetime(T1["Date"]).dt.normalize()

needed_cols = [c for c in T1.columns if (
    "In_discharge" in c or
    "In_TSS" in c or
    "In_TP" in c or
    "Water_elevation" in c
)]

T1 = T1[["Date"] + needed_cols].set_index("Date")

print("✔ Filling Combine variables...")

for col in needed_cols:
    df[col] = df["Date"].map(T1[col])

# =========================================================
# 4️⃣ Bathymetry（KDTree）
# =========================================================
print("🔄 Bathymetry...")

depth_col = [c for c in G.columns if "depth" in c.lower()][0]

tree_depth = cKDTree(G[["i","j"]].values)
_, idx = tree_depth.query(df[["i","j"]].values)

df["Bathymetry_depth"] = G.iloc[idx][depth_col].values

print("✔ Bathymetry assigned")

# =========================================================
# 5️⃣ Soil（KDTree + 最近邻补）
# =========================================================
print("🔄 Soil...")

J = pd.read_csv(joined_file)
J = clean_columns(J)

soil_cols = [c for c in J.columns if any(k in c.upper() for k in [
    "BLOCK","BOULDER","COBBLE","GRAVEL","SAND","SILT","CLAY"
])]

print("Detected soil columns:", soil_cols)

df["i"] = df["i"].astype(int)
df["j"] = df["j"].astype(int)
J["i"] = J["i"].astype(int)
J["j"] = J["j"].astype(int)

# KDTree匹配
tree_soil = cKDTree(J[["i","j"]].values)
_, idx = tree_soil.query(df[["i","j"]].values)

for col in soil_cols:
    df[col] = J.iloc[idx][col].values

print("✔ Initial soil assigned")

# 最近邻补缺
print("🔄 Filling missing soil...")

valid = df[soil_cols].notna().all(axis=1)

if valid.sum() > 0:
    tree_valid = cKDTree(df.loc[valid, ["i","j"]].values)

    missing_idx = np.where(~valid)[0]

    for idx in missing_idx:
        point = df.loc[idx, ["i","j"]].values
        _, nn = tree_valid.query(point)

        nearest = df.loc[valid].iloc[nn]

        for col in soil_cols:
            df.at[idx, col] = nearest[col]

print("✔ Soil fully filled")

# =========================================================
# 6️⃣ Water_depth
# =========================================================
df = df.loc[:, ~df.columns.duplicated()]

df["Water_depth"] = df["Water_elevation"] - df["Bathymetry_depth"]

print("✔ Water_depth computed")

# =========================================================
# 7️⃣ Water_temp（🔥最终修复版）
# =========================================================
print("🌡 Filling Water_temp...")

df["Water_temp"] = np.nan
df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()

files = glob.glob(os.path.join(temp_parquet_dir, "*.parquet"))

for f in files:
    temp_df = pd.read_parquet(f)
    temp_df = clean_columns(temp_df)

    # 🔥 关键：统一到“只有日期”
    temp_df["Date"] = pd.to_datetime(temp_df["Date"]).dt.normalize()

    # ==============================
    # 🔍 DEBUG 检查（关键）
    # ==============================
    check = temp_df[
        (temp_df["i"] == 118) &
        (temp_df["j"] == 80) &
        (temp_df["Date"] == pd.Timestamp("2001-06-18"))
    ]

    print(f"\nChecking file: {os.path.basename(f)}")
    print(check)

    temp_df["i"] = temp_df["i"].astype(int)
    temp_df["j"] = temp_df["j"].astype(int)

    temp_col = [c for c in temp_df.columns if "temp" in c.lower()][0]
    temp_df = temp_df.rename(columns={temp_col: "Water_temp"})

    merge_df = pd.merge(
        df[["i","j","Date"]],
        temp_df[["i","j","Date","Water_temp"]],
        on=["i","j","Date"],
        how="left"
    )

    df["Water_temp"] = df["Water_temp"].combine_first(merge_df["Water_temp"])

print("✔ Water_temp filled")

# =========================================================
# 8️⃣ Distance
# =========================================================
print("📏 Distance...")

rivers = {
    "GreatLakes": (218, 8),
    "OttawaRiver": (218, 3),
    "RichelieuRiver": (167, 40),
    "YamaskaRiver": (128, 111),
    "Saint_FrancoisRiver": (128, 116),
    "NicoletRiver": (55, 220),
    "MaskinongeRiver": (103, 78),
    "DuLoupRiver": (70, 115),
    "YamachicheRiver": (47, 157)
}

for name, (ri, rj) in rivers.items():
    df[f"In_distance_{name}"] = np.sqrt(
        (df["i"] - ri)**2 + (df["j"] - rj)**2
    )

print("✔ Distance computed")

# =========================================================
# 9️⃣ 填补 TSS_LSP（🔥新增）
# =========================================================
print("🌊 Filling TSS_LSP...")

tss_dir = r"E:\SAV_map_PNAS\Submission_package\Data\02_tss\Prediction\TSS_prediction_results"

df["TSS_LSP"] = np.nan

# 👉 获取当前年份
years = df["Date"].dt.year.unique()

for year in years:
    tss_file = os.path.join(tss_dir, f"TSS_{year}_daily.parquet")

    if not os.path.exists(tss_file):
        print(f"❌ 缺少文件: {tss_file}")
        continue

    print(f"读取: TSS_{year}_daily.parquet")

    tss_df = pd.read_parquet(tss_file)
    tss_df = clean_columns(tss_df)

    # 统一格式（关键）
    tss_df["Date"] = pd.to_datetime(tss_df["Date"]).dt.normalize()
    tss_df["i"] = tss_df["i"].astype(int)
    tss_df["j"] = tss_df["j"].astype(int)

    # 找TSS列（自动识别）
    tss_cols = [c for c in tss_df.columns if "tss" in c.lower()]

    if len(tss_cols) == 0:
        print(f"❌ 没有 TSS 列: {tss_file}")
        continue

    tss_col = tss_cols[0]
    tss_df = tss_df.rename(columns={tss_col: "TSS_LSP"})

    # merge
    merge_df = pd.merge(
        df[["i","j","Date"]],
        tss_df[["i","j","Date","TSS_LSP"]],
        on=["i","j","Date"],
        how="left"
    )

    # 填补
    df["TSS_LSP"] = df["TSS_LSP"].combine_first(merge_df["TSS_LSP"])

print("✔ TSS_LSP filled")

# =========================================================
# 保持原表头
# =========================================================
for col in orig_cols:
    if col not in df.columns:
        df[col] = np.nan

df = df[orig_cols]

# =========================================================
# 保存
# =========================================================
df.to_excel(out_file, index=False)

print("\n🎉 DONE — Water_temp已彻底修复")
print("Saved:", out_file)