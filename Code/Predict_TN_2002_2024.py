# =========================================================
# =========================================================

import pandas as pd
import numpy as np
import os
HERE = os.path.dirname(os.path.abspath(__file__))
import glob
import joblib
import matplotlib.pyplot as plt

# =========================================================
# PATH
# =========================================================
base_dir = os.path.abspath(os.path.join(HERE, "..", "Data", "03_tp_tn", "TN_prediction_LSP"))

input_dir = os.path.join(base_dir, "Prediction", "TN_prediction_filled_parquet")
output_dir = os.path.join(base_dir, "Prediction", "TN_prediction_results")
fig_dir = os.path.join(base_dir, "Prediction", "TN_maps")

START_YEAR = 2002

os.makedirs(output_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)

# =========================================================
# LOAD MODEL
# =========================================================
# =========================================================
# =========================================================
print(" Loading models...")

model_channel_path = os.path.join(base_dir, "Final_Model_TN_noTotal", "model_channel.pkl")
model_nonchannel_path = os.path.join(base_dir, "Final_Model_TN_noTotal", "model_nonchannel.pkl")

bundle_c = joblib.load(model_channel_path)
bundle_n = joblib.load(model_nonchannel_path)

model_channel = bundle_c["model"]
model_nonchannel = bundle_n["model"]

features = bundle_c["features"]
med = bundle_c["median"]

# =========================================================
# FILES
# =========================================================
files = sorted(
    f for f in glob.glob(os.path.join(input_dir, "TN_*.parquet"))
    if int(os.path.basename(f).split("_")[1].split(".")[0]) >= START_YEAR
)

print(f" Found {len(files)} TN input parquet files")
if len(files) == 0:
    raise FileNotFoundError(
        f"No TN input parquet files were found in {input_dir}. "
        "Run Fill_inputs_daily_TN_prediction.py first."
    )

# =========================================================
# LOOP YEARS
# =========================================================
for f in files:

    year = int(os.path.basename(f).split("_")[1].split(".")[0])
    print(f"\n Processing TN year {year}")

    df = pd.read_parquet(f)

    # =====================================================
    # =====================================================
    df["Date"] = pd.to_datetime(df["Date"])
    df["DOY"] = df["Date"].dt.dayofyear
    df["Month"] = df["Date"].dt.month

    df["sin_DOY"] = np.sin(2*np.pi*df["DOY"]/365)
    df["cos_DOY"] = np.cos(2*np.pi*df["DOY"]/365)

    # =====================================================
    # =====================================================
    df["depth_inv"] = 1 / (np.abs(df["Bathymetry_depth"]) + 1)

    df["flow_effect"] = (
        (df["In_discharge_GreatLakes"] + df["In_discharge_OttawaRiver"])
        / (np.abs(df["Bathymetry_depth"]) + 1)
    )

    from scipy.spatial import cKDTree

    channel_mask = df["Bathymetry_depth"] <= -8

    if channel_mask.sum() > 0:
        tree = cKDTree(df.loc[channel_mask, ["i","j"]].values)
        dist, _ = tree.query(df[["i","j"]].values)
    else:
        dist = np.zeros(len(df))

    df["dist_to_channel"] = dist
    df["dist_to_channel_norm"] = dist / (dist.max() + 1e-6)
    df["is_channel"] = (df["Bathymetry_depth"] <= -8).astype(int)

    # =====================================================
    # =====================================================
    X = df[features].to_numpy(dtype=np.float32)
    X = np.where(np.isfinite(X), X, med)

    # =====================================================
    # =====================================================
    df["TN_pred"] = np.nan

    # ========= channel =========
    mask_c = df["is_channel"] == 1

    if mask_c.any():
        Xc = df.loc[mask_c, features].to_numpy(dtype=np.float32)
        Xc = np.where(np.isfinite(Xc), Xc, med)

        df.loc[mask_c, "TN_pred"] = np.expm1(model_channel.predict(Xc))

    # ========= non-channel =========
    mask_n = df["is_channel"] == 0

    if mask_n.any():
        Xn = df.loc[mask_n, features].to_numpy(dtype=np.float32)
        Xn = np.where(np.isfinite(Xn), Xn, med)

        df.loc[mask_n, "TN_pred"] = np.expm1(model_nonchannel.predict(Xn))

    # =====================================================
    # =====================================================
    out_file = os.path.join(output_dir, f"TN_{year}_daily.parquet")

    df[["Date","i","j","TN_pred"]].to_parquet(out_file, index=False)

    print(f" Saved TN daily prediction: {out_file}")

    # =====================================================
    # =====================================================
    print(" Generating TN map...")

    months = [5,6,7,8,9,10]

    fig, axes = plt.subplots(2, 3, figsize=(14,8))
    axes = axes.flatten()

    vmin = 0
    vmax = 2

    for idx, m in enumerate(months):

        ax = axes[idx]

        df_m = df[df["Month"] == m]

        df_m = df_m.groupby(["i", "j"])["TN_pred"].median().reset_index()

        sc = ax.scatter(
            df_m["j"],   #
            df_m["i"],   #
            c=df_m["TN_pred"],
            s=1,
            vmin=vmin,
            vmax=vmax,
            cmap="viridis"  #
        )

        ax.set_title(f"{year}-{m:02d}")

        ax.invert_yaxis()

        ax.axis("off")

    plt.tight_layout()
    plt.subplots_adjust(right=0.88)

    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])

    cbar = fig.colorbar(sc, cax=cbar_ax)

    cbar.set_label("TN (mg/L)")

    fig_path = os.path.join(fig_dir, f"TN_{year}.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print(f" TN map saved: {fig_path}")

print("\n===================================")
print(" ALL TN YEARS DONE")
print("===================================")
