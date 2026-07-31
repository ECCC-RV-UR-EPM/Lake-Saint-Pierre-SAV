# =========================================================
# 批量填补 TN prediction（TXT → parquet）【FINAL STABLE FAST】
# =========================================================

import pandas as pd
import numpy as np
import os
from scipy.spatial import cKDTree
import glob

base_dir = r"E:\SAV_map_PNAS\Submission_package\Data\03_tp_tn\TN_prediction_LSP"

prediction_dir = os.path.join(base_dir, "Prediction")
out_dir = os.path.join(base_dir, "Prediction","TN_prediction_filled_parquet")
os.makedirs(out_dir, exist_ok=True)

grid_file = os.path.join(base_dir, "lat_lon_UMT_i_j.csv")
combine_file = os.path.join(base_dir, "Combine.xlsx")
joined_file = os.path.join(base_dir, "JOINED_with_ij.csv")

temp_parquet_dir = r"E:\SAV_map_PNAS\Submission_package\Data\01_temperature\Prediction\Final_Temp_Output_monthly_latlon"

# =========================================================
# 工具
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
# 预加载（一次）
# =========================================================
print("📥 Loading static data...")

G = clean_columns(pd.read_csv(grid_file))
J = clean_columns(pd.read_csv(joined_file))
T1 = clean_columns(pd.read_excel(combine_file))

T1["Date"] = pd.to_datetime(T1["Date"]).dt.normalize()
T1 = T1.set_index("Date")

tree_depth = cKDTree(G[["i","j"]].values)
tree_soil = cKDTree(J[["i","j"]].values)

depth_col = [c for c in G.columns if "depth" in c.lower()][0]

soil_cols = [c for c in J.columns if any(k in c.upper() for k in [
    "BLOCK","BOULDER","COBBLE","GRAVEL","SAND","SILT","CLAY"
])]

# =========================================================
# 批量处理
# =========================================================
files = glob.glob(os.path.join(prediction_dir, "TN_prediction_*.txt"))

files = sorted(
    files,
    key=lambda x: int(os.path.basename(x).split("_")[-1].split(".")[0])
)

for file in files:

    year = os.path.basename(file).split("_")[-1].split(".")[0]
    year_int = int(year)

    print(f"\n🚀 Processing {year}")

    # =====================================================
    # 稳定读取 TXT（只读3列）
    # =====================================================
    df = pd.read_csv(
        file,
        usecols=[0,1,2],
        names=["i","j","Date"],
        header=0,
        engine="python",
        on_bad_lines="skip"
    )

    df = clean_columns(df)

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce",
        format="mixed"
    ).dt.normalize()

    df = df.dropna(subset=["Date"])

    df["i"] = df["i"].astype(int)
    df["j"] = df["j"].astype(int)

    # =====================================================
    # Combine（一次 merge）
    # =====================================================
    df = df.merge(T1, left_on="Date", right_index=True, how="left")

    # =====================================================
    # Bathymetry
    # =====================================================
    _, idx = tree_depth.query(df[["i","j"]].values)
    df["Bathymetry_depth"] = G.iloc[idx][depth_col].values

    # =====================================================
    # Soil
    # =====================================================
    _, idx = tree_soil.query(df[["i","j"]].values)
    for col in soil_cols:
        df[col] = J.iloc[idx][col].values

    # =====================================================
    # Water_depth
    # =====================================================
    df["Water_depth"] = df["Water_elevation"] - df["Bathymetry_depth"]

    # =====================================================
    # 🔥 正确 + 加速 Water_temp（核心）
    # =====================================================
    print("🌡 Filling Water_temp...")

    df["Water_temp"] = np.nan

    temp_files = [os.path.join(temp_parquet_dir, f"Temp_{year}_daily.parquet")]

    for f in temp_files:
        temp_df = pd.read_parquet(f)

        # 🔥 关键加速：只取当前年份
        temp_df = temp_df[
            pd.to_datetime(temp_df["Date"]).dt.year == year_int
        ]

        if len(temp_df) == 0:
            continue

        temp_df = clean_columns(temp_df)

        temp_df["Date"] = pd.to_datetime(temp_df["Date"]).dt.normalize()
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

        # 🔥 核心逻辑（必须保留）
        df["Water_temp"] = df["Water_temp"].combine_first(merge_df["Water_temp"])

    # =====================================================
    # 🔥 NEW：Fill TSS_LSP（完全独立，不影响原逻辑）
    # =====================================================
    print("🟤 Filling TSS_LSP...")

    df["TSS_LSP"] = np.nan

    tss_parquet_dir = r"E:\SAV_map_PNAS\Submission_package\Data\02_tss\Prediction\TSS_prediction_results"

    tss_file = os.path.join(tss_parquet_dir, f"TSS_{year}_daily.parquet")

    if os.path.exists(tss_file):

        tss_df = pd.read_parquet(tss_file)

        tss_df = clean_columns(tss_df)

        # 时间 & 坐标统一
        tss_df["Date"] = pd.to_datetime(tss_df["Date"]).dt.normalize()
        tss_df["i"] = tss_df["i"].astype(int)
        tss_df["j"] = tss_df["j"].astype(int)

        # 🔥 自动识别 TSS 列（你说你不确定列名）
        tss_col = [c for c in tss_df.columns if "tss" in c.lower()][0]

        tss_df = tss_df.rename(columns={tss_col: "TSS_LSP"})

        merge_df = pd.merge(
            df[["i", "j", "Date"]],
            tss_df[["i", "j", "Date", "TSS_LSP"]],
            on=["i", "j", "Date"],
            how="left"
        )

        # 🔥 保持和 Water_temp 完全一致逻辑
        df["TSS_LSP"] = df["TSS_LSP"].combine_first(merge_df["TSS_LSP"])

    else:
        print(f"⚠ TSS file not found for {year}")

    # =====================================================
    # Distance
    # =====================================================
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

    # =====================================================
    # 保存 parquet
    # =====================================================
    out_path = os.path.join(out_dir, f"TN_{year}.parquet")
    df.to_parquet(out_path, index=False)

    print(f"✔ Saved: {out_path}")

print("\n🎉 ALL DONE (FINAL STABLE VERSION)")