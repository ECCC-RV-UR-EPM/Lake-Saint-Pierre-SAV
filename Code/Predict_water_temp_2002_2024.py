import os
HERE = os.path.dirname(os.path.abspath(__file__))
import glob
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# ============================================
# PATH
# ============================================
base_dir = os.path.abspath(os.path.join(HERE, "..", "Data", "01_temperature", "Temp_prediction_LSP"))

input_dir = os.path.join(base_dir, "Prediction", "Temp_prediction_expanded")
cache_dir = os.path.join(input_dir, "parquet_cache")
output_dir = os.path.join(input_dir, "Final_Temp_Output_monthly_latlon")
intermediate_dir = os.path.join(output_dir, "_intermediate_pred")
figure_dir = os.path.join(output_dir, "Figures_obs")
START_YEAR = 2002

model_channel_path = os.path.join(base_dir, "Train_results_residual_split", "model_channel.pkl")
model_nonchannel_path = os.path.join(base_dir, "Train_results_residual_split", "model_nonchannel.pkl")
obs_file = os.path.join(base_dir, "Temp_training_daily_observe_to_train_clip_filled.xlsx")

os.makedirs(output_dir, exist_ok=True)
os.makedirs(intermediate_dir, exist_ok=True)
os.makedirs(figure_dir, exist_ok=True)

# ============================================
# LOAD MODELS
# ============================================
print("Loading models...")
model_channel = joblib.load(model_channel_path)
model_nonchannel = joblib.load(model_nonchannel_path)

# ============================================
# FEATURES
# ============================================
model_features = [
    "i", "j", "sin_DOY", "cos_DOY", "Month",
    "Air_temp", "Water_depth", "depth_inv", "flow_effect",
    "BLOCKS", "BOULDERS", "COBBLES", "GRAVEL", "SAND", "SILT", "CLAY",
    "BLOCKSIZE", "BOULDERSIZE", "COBBLESIZE", "GRAVELSIZE", "SANDSIZE", "SILTSIZE", "CLAYSIZE",
    "In_discharge_GreatLakes", "In_discharge_OttawaRiver",
    "In_discharge_RichelieuRiver", "In_discharge_YamaskaRiver",
    "In_discharge_Saint_FrancoisRiver", "In_discharge_NicoletRiver",
    "In_discharge_MaskinongeRiver", "In_discharge_DuLoupRiver", "In_discharge_YamachicheRiver",
    "In_distance_GreatLakes", "In_distance_OttawaRiver",
    "In_distance_RichelieuRiver", "In_distance_YamaskaRiver",
    "In_distance_Saint_FrancoisRiver", "In_distance_NicoletRiver",
    "In_distance_MaskinongeRiver", "In_distance_DuLoupRiver", "In_distance_YamachicheRiver",
    "dist_to_channel", "dist_to_channel_norm",
    "depth_x_dist", "channel_cooling",
]


def year_from_file(path):
    return os.path.basename(path).split("_")[1].split(".")[0]


def build_prediction_features(df):
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")

    df["Date"] = pd.to_datetime(df["Date"])
    df["DOY"] = df["Date"].dt.dayofyear
    df["Month"] = df["Date"].dt.month
    df["Year"] = df["Date"].dt.year

    df["sin_DOY"] = np.sin(2 * np.pi * df["DOY"] / 365).astype("float32")
    df["cos_DOY"] = np.cos(2 * np.pi * df["DOY"] / 365).astype("float32")
    df["Satellite_temp_raw"] = df["Satellite_temp"]

    df["is_channel"] = (df["Bathymetry_depth"] <= -8).astype("int8")
    df["depth_inv"] = 1 / (np.abs(df["Water_depth"]) + 1)
    df["flow_effect"] = (
        (df["In_discharge_GreatLakes"] + df["In_discharge_OttawaRiver"])
        / (np.abs(df["Water_depth"]) + 1)
    )

    channel_mask = df["Bathymetry_depth"] <= -8
    if channel_mask.sum() > 0:
        tree = cKDTree(df.loc[channel_mask, ["i", "j"]].values)
        dist, _ = tree.query(df[["i", "j"]].values)
    else:
        dist = np.zeros(len(df), dtype=np.float32)

    df["dist_to_channel"] = dist
    df["dist_to_channel_norm"] = dist / (dist.max() + 1e-6)
    df["depth_x_dist"] = df["Water_depth"] * df["dist_to_channel_norm"]
    df["channel_cooling"] = df["flow_effect"] * df["dist_to_channel_norm"]
    return df


def predict_one_year(cache_file):
    year = year_from_file(cache_file)
    print(f"\nProcessing year {year}")

    df = pd.read_parquet(cache_file)
    df = build_prediction_features(df)
    df["Water_temp"] = np.nan

    mask_c = df["is_channel"] == 1
    if mask_c.any():
        x = df.loc[mask_c, model_features].to_numpy(dtype=np.float32)
        med = np.nanmean(x, axis=0)
        x = np.where(np.isfinite(x), x, med)
        pred = model_channel.predict(x)
        df.loc[mask_c, "Water_temp"] = df.loc[mask_c, "Satellite_temp_raw"] + pred

    mask_n = df["is_channel"] == 0
    if mask_n.any():
        x = df.loc[mask_n, model_features].to_numpy(dtype=np.float32)
        med = np.nanmean(x, axis=0)
        x = np.where(np.isfinite(x), x, med)
        pred = model_nonchannel.predict(x)
        df.loc[mask_n, "Water_temp"] = df.loc[mask_n, "Satellite_temp_raw"] + pred

    interm_file = os.path.join(intermediate_dir, f"Temp_{year}_predicted.parquet")
    df.to_parquet(interm_file, index=False)
    print(f"Intermediate saved: {interm_file}")


def build_doy_climatology(intermediate_files):
    print("\nStreaming DOY climatology...")
    sum_dict = {}
    count_dict = {}

    for f in intermediate_files:
        df = pd.read_parquet(f, columns=["i", "j", "DOY", "Water_temp"])
        vals = df[["i", "j", "DOY", "Water_temp"]].to_numpy()
        for i, j, doy, temp in vals:
            if not np.isnan(temp):
                key = (int(i), int(j), int(doy))
                if key in sum_dict:
                    sum_dict[key] += temp
                    count_dict[key] += 1
                else:
                    sum_dict[key] = temp
                    count_dict[key] = 1

    print("DOY stats done")
    return sum_dict, count_dict


def finalize_one_year(intermediate_file, sum_dict, count_dict):
    year = year_from_file(intermediate_file.replace("_predicted", ""))
    df = pd.read_parquet(intermediate_file)
    vals = df[["i", "j", "DOY", "Water_temp"]].to_numpy()

    for idx in range(len(vals)):
        if np.isnan(vals[idx, 3]):
            key = (int(vals[idx, 0]), int(vals[idx, 1]), int(vals[idx, 2]))
            if key in sum_dict:
                vals[idx, 3] = sum_dict[key] / count_dict[key]

    df["Water_temp"] = vals[:, 3]
    df["Water_temp"] = df["Water_temp"].fillna(df["Water_temp"].mean())

    out_parquet = os.path.join(output_dir, f"Temp_{year}_daily.parquet")
    df[["Date", "i", "j", "Water_temp"]].to_parquet(out_parquet, index=False)
    print(f"Saved: {out_parquet}")


def plot_one_year(obs, year):
    parquet_file = os.path.join(output_dir, f"Temp_{year}_daily.parquet")
    df = pd.read_parquet(parquet_file)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Month"] = df["Date"].dt.month

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    vmin = 0
    vmax = 35
    months = [5, 6, 7, 8, 9, 10]
    obs_year = obs[(obs["Year"] == year) & (obs["Month"].between(5, 10))]

    for idx, month in enumerate(months):
        ax = axes[idx]
        df_m = df[df["Month"] == month].groupby(["i", "j"])["Water_temp"].median().reset_index()

        sc = ax.scatter(
            df_m["j"],
            df_m["i"],
            c=df_m["Water_temp"],
            s=1,
            vmin=vmin,
            vmax=vmax,
            cmap="turbo",
        )

        obs_m = obs_year[obs_year["Month"] == month]
        if len(obs_m) > 0:
            ax.scatter(
                obs_m["j"],
                obs_m["i"],
                c=obs_m["Temp_observation"],
                cmap="turbo",
                vmin=vmin,
                vmax=vmax,
                marker="^",
                s=40,
                edgecolors="white",
                linewidths=0.8,
            )

        ax.set_title(f"{year}-{month:02d}")
        ax.invert_yaxis()
        ax.axis("off")

    plt.tight_layout()
    plt.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label("Water Temp (C)")

    fig_path = os.path.join(figure_dir, f"Temp_{year}_obs.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Map saved: {fig_path}")


def main():
    cache_files = sorted(
        f for f in glob.glob(os.path.join(cache_dir, "Temp_*.parquet"))
        if int(year_from_file(f)) >= START_YEAR
    )
    years = [int(year_from_file(f)) for f in cache_files]

    for cache_file in cache_files:
        predict_one_year(cache_file)

    intermediate_files = sorted(glob.glob(os.path.join(intermediate_dir, "Temp_*_predicted.parquet")))
    sum_dict, count_dict = build_doy_climatology(intermediate_files)

    for intermediate_file in intermediate_files:
        finalize_one_year(intermediate_file, sum_dict, count_dict)

    print("\n===================================")
    print("FINAL: All parquet have NO NaN")
    print("===================================")

    print("\nLoading observation...")
    obs = pd.read_excel(obs_file)
    obs["Date"] = pd.to_datetime(obs["Date"])
    obs["Month"] = obs["Date"].dt.month
    obs["Year"] = obs["Date"].dt.year

    print("\nGenerating temperature maps (with obs)...")
    for year in years:
        plot_one_year(obs, year)

    print("\n===================================")
    print("FINAL WITH OBS DONE")
    print("===================================")


if __name__ == "__main__":
    main()
