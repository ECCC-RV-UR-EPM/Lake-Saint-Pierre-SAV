# =========================================================
# Predict daily TP fields, auto-building yearly input parquet
# files from raw yearly text inputs when needed.
# =========================================================

import glob
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))

# =========================================================
# PATH
# =========================================================
base_dir = os.path.abspath(os.path.join(HERE, "..", "Data", "03_tp_tn", "TP_prediction_LSP"))

input_dir = os.path.join(base_dir, "Prediction", "TP_prediction_filled_parquet")
raw_txt_dir = os.path.join(base_dir, "Prediction", "TP_prediction_expanded")
output_dir = os.path.join(base_dir, "Prediction", "TP_prediction_results")
fig_dir = os.path.join(base_dir, "Prediction", "TP_maps")

model_channel_path = os.path.join(base_dir, "Final_Model_TP_noTotal", "model_channel.pkl")
model_nonchannel_path = os.path.join(base_dir, "Final_Model_TP_noTotal", "model_nonchannel.pkl")

grid_file = os.path.join(base_dir, "lat_lon_UMT_i_j.csv")
combine_file = os.path.join(base_dir, "Combine.xlsx")
joined_file = os.path.join(base_dir, "JOINED_with_ij.csv")

temp_parquet_dir = os.path.abspath(
    os.path.join(
        HERE, "..", "Data", "01_temperature", "Temp_prediction_LSP",
        "Prediction", "Temp_prediction_expanded", "Final_Temp_Output_monthly_latlon"
    )
)
tss_parquet_dir = os.path.abspath(
    os.path.join(
        HERE, "..", "Data", "02_tss", "TSS_prediction_LSP",
        "Prediction", "TSS_prediction_results"
    )
)

START_YEAR = 2002

os.makedirs(input_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)


def clean_columns(df):
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("(", "")
        .str.replace(")", "")
    )
    return df


print(" Loading models...")

bundle_c = joblib.load(model_channel_path)
bundle_n = joblib.load(model_nonchannel_path)

model_channel = bundle_c["model"]
model_nonchannel = bundle_n["model"]
features = bundle_c["features"]
med = bundle_c["median"]

# =========================================================
# STATIC DATA FOR ON-THE-FLY INPUT BUILDING
# =========================================================
print(" Loading static TP support tables...")

G = clean_columns(pd.read_csv(grid_file))
J = clean_columns(pd.read_csv(joined_file))
T1 = clean_columns(pd.read_excel(combine_file))
T1["Date"] = pd.to_datetime(T1["Date"]).dt.normalize()
T1 = T1.set_index("Date")

tree_depth = cKDTree(G[["i", "j"]].values)
tree_soil = cKDTree(J[["i", "j"]].values)

depth_col = [c for c in G.columns if "depth" in c.lower()][0]
soil_cols = [c for c in J.columns if any(k in c.upper() for k in [
    "BLOCK", "BOULDER", "COBBLE", "GRAVEL", "SAND", "SILT", "CLAY"
])]


def build_tp_input_for_year(year: int) -> str:
    txt_file = os.path.join(raw_txt_dir, f"TP_prediction_{year}.txt")
    out_path = os.path.join(input_dir, f"TP_{year}.parquet")

    if os.path.exists(out_path):
        return out_path

    if not os.path.exists(txt_file):
        raise FileNotFoundError(
            f"Neither cached parquet nor raw TP text input was found for {year}. "
            f"Missing file: {txt_file}"
        )

    print(f" Building TP input parquet for {year} from raw text...")

    df = pd.read_csv(
        txt_file,
        usecols=[0, 1, 2],
        names=["i", "j", "Date"],
        header=0,
        engine="python",
        on_bad_lines="skip",
    )
    df = clean_columns(df)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", format="mixed").dt.normalize()
    df = df.dropna(subset=["Date"])
    df["i"] = df["i"].astype(int)
    df["j"] = df["j"].astype(int)

    df = df.merge(T1, left_on="Date", right_index=True, how="left")

    _, idx = tree_depth.query(df[["i", "j"]].values)
    df["Bathymetry_depth"] = G.iloc[idx][depth_col].values

    _, idx = tree_soil.query(df[["i", "j"]].values)
    for col in soil_cols:
        df[col] = J.iloc[idx][col].values

    df["Water_depth"] = df["Water_elevation"] - df["Bathymetry_depth"]

    print("  Filling Water_temp...")
    temp_file = os.path.join(temp_parquet_dir, f"Temp_{year}_daily.parquet")
    if not os.path.exists(temp_file):
        raise FileNotFoundError(f"Missing required temperature file: {temp_file}")

    temp_df = pd.read_parquet(temp_file)
    temp_df = clean_columns(temp_df)
    temp_df["Date"] = pd.to_datetime(temp_df["Date"]).dt.normalize()
    temp_df["i"] = temp_df["i"].astype(int)
    temp_df["j"] = temp_df["j"].astype(int)
    temp_col = [c for c in temp_df.columns if "temp" in c.lower()][0]
    temp_df = temp_df.rename(columns={temp_col: "Water_temp"})

    df = df.merge(
        temp_df[["i", "j", "Date", "Water_temp"]],
        on=["i", "j", "Date"],
        how="left",
    )

    print("  Filling TSS_LSP...")
    tss_file = os.path.join(tss_parquet_dir, f"TSS_{year}_daily.parquet")
    if not os.path.exists(tss_file):
        raise FileNotFoundError(f"Missing required TSS file: {tss_file}")

    tss_df = pd.read_parquet(tss_file)
    tss_df = clean_columns(tss_df)
    tss_df["Date"] = pd.to_datetime(tss_df["Date"]).dt.normalize()
    tss_df["i"] = tss_df["i"].astype(int)
    tss_df["j"] = tss_df["j"].astype(int)
    tss_col = [c for c in tss_df.columns if "tss" in c.lower()][0]
    tss_df = tss_df.rename(columns={tss_col: "TSS_LSP"})

    df = df.merge(
        tss_df[["i", "j", "Date", "TSS_LSP"]],
        on=["i", "j", "Date"],
        how="left",
    )

    rivers = {
        "GreatLakes": (218, 8),
        "OttawaRiver": (218, 3),
        "RichelieuRiver": (167, 40),
        "YamaskaRiver": (128, 111),
        "Saint_FrancoisRiver": (128, 116),
        "NicoletRiver": (55, 220),
        "MaskinongeRiver": (103, 78),
        "DuLoupRiver": (70, 115),
        "YamachicheRiver": (47, 157),
    }

    for name, (ri, rj) in rivers.items():
        df[f"In_distance_{name}"] = np.sqrt((df["i"] - ri) ** 2 + (df["j"] - rj) ** 2)

    df.to_parquet(out_path, index=False)
    print(f"  Saved input parquet: {out_path}")
    return out_path


# =========================================================
# DETERMINE YEARS TO RUN
# =========================================================
parquet_files = sorted(
    f for f in glob.glob(os.path.join(input_dir, "TP_*.parquet"))
    if int(os.path.basename(f).split("_")[1].split(".")[0]) >= START_YEAR
)
txt_files = sorted(
    f for f in glob.glob(os.path.join(raw_txt_dir, "TP_prediction_*.txt"))
    if int(os.path.basename(f).split("_")[-1].split(".")[0]) >= START_YEAR
)

years_from_parquet = {
    int(os.path.basename(f).split("_")[1].split(".")[0]) for f in parquet_files
}
years_from_txt = {
    int(os.path.basename(f).split("_")[-1].split(".")[0]) for f in txt_files
}
years = sorted(years_from_parquet | years_from_txt)

print(f" Found {len(years_from_parquet)} cached TP parquet files")
print(f" Found {len(years_from_txt)} raw TP text files")

if len(years) == 0:
    raise FileNotFoundError(
        f"No TP inputs were found in either {input_dir} or {raw_txt_dir}."
    )

# =========================================================
# LOOP YEARS
# =========================================================
for year in years:
    print(f"\n Processing TP year {year}")

    parquet_file = build_tp_input_for_year(year)
    df = pd.read_parquet(parquet_file)

    df["Date"] = pd.to_datetime(df["Date"])
    df["DOY"] = df["Date"].dt.dayofyear
    df["Month"] = df["Date"].dt.month

    df["sin_DOY"] = np.sin(2 * np.pi * df["DOY"] / 365)
    df["cos_DOY"] = np.cos(2 * np.pi * df["DOY"] / 365)

    df["depth_inv"] = 1 / (np.abs(df["Bathymetry_depth"]) + 1)
    df["flow_effect"] = (
        (df["In_discharge_GreatLakes"] + df["In_discharge_OttawaRiver"])
        / (np.abs(df["Bathymetry_depth"]) + 1)
    )

    channel_mask = df["Bathymetry_depth"] <= -8
    if channel_mask.sum() > 0:
        tree = cKDTree(df.loc[channel_mask, ["i", "j"]].values)
        dist, _ = tree.query(df[["i", "j"]].values)
    else:
        dist = np.zeros(len(df))

    df["dist_to_channel"] = dist
    df["dist_to_channel_norm"] = dist / (dist.max() + 1e-6)
    df["is_channel"] = (df["Bathymetry_depth"] <= -8).astype(int)

    df["TP_pred"] = np.nan

    mask_c = df["is_channel"] == 1
    if mask_c.any():
        Xc = df.loc[mask_c, features].to_numpy(dtype=np.float32)
        Xc = np.where(np.isfinite(Xc), Xc, med)
        df.loc[mask_c, "TP_pred"] = np.expm1(model_channel.predict(Xc))

    mask_n = df["is_channel"] == 0
    if mask_n.any():
        Xn = df.loc[mask_n, features].to_numpy(dtype=np.float32)
        Xn = np.where(np.isfinite(Xn), Xn, med)
        df.loc[mask_n, "TP_pred"] = np.expm1(model_nonchannel.predict(Xn))

    out_file = os.path.join(output_dir, f"TP_{year}_daily.parquet")
    df[["Date", "i", "j", "TP_pred"]].to_parquet(out_file, index=False)
    print(f" Saved TP daily prediction: {out_file}")

    print(" Generating TP map...")
    months = [5, 6, 7, 8, 9, 10]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    vmin = 0
    vmax = 0.1

    for idx, m in enumerate(months):
        ax = axes[idx]
        df_m = df[df["Month"] == m]
        df_m = df_m.groupby(["i", "j"])["TP_pred"].median().reset_index()

        sc = ax.scatter(
            df_m["j"],
            df_m["i"],
            c=df_m["TP_pred"],
            s=1,
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
        )
        ax.set_title(f"{year}-{m:02d}")
        ax.invert_yaxis()
        ax.axis("off")

    plt.tight_layout()
    plt.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label("TP (mg/L)")

    fig_path = os.path.join(fig_dir, f"TP_{year}.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f" TP map saved: {fig_path}")

print("\n===================================")
print(" ALL TP YEARS DONE")
print("===================================")
