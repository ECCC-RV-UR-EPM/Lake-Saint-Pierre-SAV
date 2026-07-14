import pandas as pd
import numpy as np
import os
HERE = os.path.dirname(os.path.abspath(__file__))
from scipy.spatial import cKDTree
import glob

# =========================================================
# PATH
# =========================================================
base_dir = os.path.abspath(os.path.join(HERE, "..", "Data", "03_tp_tn", "TN_prediction_LSP"))

tss_file = os.path.join(base_dir, "TN_training_daily.xlsx")
grid_file = os.path.join(base_dir, "lat_lon_UMT_i_j.csv")
combine_file = os.path.join(base_dir, "Combine.xlsx")
joined_file = os.path.join(base_dir, "JOINED_with_ij.csv")

temp_parquet_dir = os.path.abspath(os.path.join(HERE, "..", "Data", "01_temperature", "Temp_prediction_LSP", "Prediction", "Temp_prediction_expanded", "Final_Temp_Output_monthly_latlon"))

out_file = os.path.join(base_dir, "TN_training_daily_filled.xlsx")

# =========================================================
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
# =========================================================
df = pd.read_excel(tss_file)
orig_cols = df.columns.tolist()

df = clean_columns(df)
df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()

print(" Load done")

# =========================================================
# =========================================================
G = pd.read_csv(grid_file)
G = clean_columns(G)

tree = cKDTree(G[["lat","lon"]].values)
_, idx = tree.query(df[["Latitude","Longitude"]].values)

df["i"] = G.iloc[idx]["i"].values.astype(int)
df["j"] = G.iloc[idx]["j"].values.astype(int)

print(" i,j assigned")

# =========================================================
# =========================================================
T1 = pd.read_excel(combine_file)
T1 = clean_columns(T1)
T1["Date"] = pd.to_datetime(T1["Date"]).dt.normalize()

needed_cols = [c for c in T1.columns if (
    "In_discharge" in c or
    "In_TSS" in c or
    "In_TN" in c or
    "Water_elevation" in c
)]

T1 = T1[["Date"] + needed_cols].set_index("Date")

print(" Filling Combine variables...")

for col in needed_cols:
    df[col] = df["Date"].map(T1[col])

# =========================================================
# =========================================================
print(" Bathymetry...")

depth_col = [c for c in G.columns if "depth" in c.lower()][0]

tree_depth = cKDTree(G[["i","j"]].values)
_, idx = tree_depth.query(df[["i","j"]].values)

df["Bathymetry_depth"] = G.iloc[idx][depth_col].values

print(" Bathymetry assigned")

# =========================================================
# =========================================================
print(" Soil...")

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

tree_soil = cKDTree(J[["i","j"]].values)
_, idx = tree_soil.query(df[["i","j"]].values)

for col in soil_cols:
    df[col] = J.iloc[idx][col].values

print(" Initial soil assigned")

print(" Filling missing soil...")

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

print(" Soil fully filled")

# =========================================================
# =========================================================
df = df.loc[:, ~df.columns.duplicated()]

df["Water_depth"] = df["Water_elevation"] - df["Bathymetry_depth"]

print(" Water_depth computed")

# =========================================================
# =========================================================
print(" Filling Water_temp...")

df["Water_temp"] = np.nan
df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()

files = glob.glob(os.path.join(temp_parquet_dir, "*.parquet"))

for f in files:
    temp_df = pd.read_parquet(f)
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

    df["Water_temp"] = df["Water_temp"].combine_first(merge_df["Water_temp"])

print(" Water_temp filled")

# =========================================================
# =========================================================
print(" Distance...")

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

print(" Distance computed")

# =========================================================
# =========================================================
print(" Filling TSS_LSP...")

tss_dir = os.path.abspath(os.path.join(HERE, "..", "Data", "02_tss", "TSS_prediction_LSP", "Prediction", "TSS_prediction_results"))

df["TSS_LSP"] = np.nan

years = df["Date"].dt.year.unique()

for year in years:
    tss_file = os.path.join(tss_dir, f"TSS_{year}_daily.parquet")

    if not os.path.exists(tss_file):
        print(f" : {tss_file}")
        continue

    print(f": TSS_{year}_daily.parquet")

    tss_df = pd.read_parquet(tss_file)
    tss_df = clean_columns(tss_df)

    tss_df["Date"] = pd.to_datetime(tss_df["Date"]).dt.normalize()
    tss_df["i"] = tss_df["i"].astype(int)
    tss_df["j"] = tss_df["j"].astype(int)

    tss_cols = [c for c in tss_df.columns if "tss" in c.lower()]

    if len(tss_cols) == 0:
        print(f"  TSS : {tss_file}")
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

    df["TSS_LSP"] = df["TSS_LSP"].combine_first(merge_df["TSS_LSP"])

print(" TSS_LSP filled")

# =========================================================
# =========================================================
for col in orig_cols:
    if col not in df.columns:
        df[col] = np.nan

df = df[orig_cols]

# =========================================================
# =========================================================
df.to_excel(out_file, index=False)

print("\n DONE TN")
print("Saved:", out_file)
