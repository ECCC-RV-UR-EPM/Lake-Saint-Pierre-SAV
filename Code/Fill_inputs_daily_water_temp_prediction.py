import pandas as pd
import numpy as np
import os
HERE = os.path.dirname(os.path.abspath(__file__))
import glob
from scipy.spatial import cKDTree

# ============================================
# PATH
# ============================================
base_dir = os.path.abspath(os.path.join(HERE, "..", "Data", "01_temperature", "Temp_prediction_LSP"))

input_dir = os.path.join(base_dir, "Prediction", "Temp_prediction_expanded")
cache_dir = os.path.join(input_dir, "parquet_cache")

os.makedirs(cache_dir, exist_ok=True)

combine_file = os.path.join(base_dir, "Combine.xlsx")
depth_file = os.path.join(base_dir, "lat_lon_UMT_i_j.csv")
sediment_file = os.path.join(base_dir, "JOINED_with_ij.csv")
sat_csv = os.path.join(base_dir, "LSP_LSWT_200m_rev02_2000-2024_satellite.csv")
sat_parquet = os.path.join(cache_dir, "satellite.parquet")
START_YEAR = 2002

# ============================================
# ============================================
print(" Loading static data...")

combine_df = pd.read_excel(combine_file)
combine_df.columns = combine_df.columns.str.strip()
combine_df["Date"] = pd.to_datetime(combine_df["Date"]).dt.normalize()

depth_full = pd.read_csv(depth_file)
depth_df = depth_full[["i","j","depth"]].rename(columns={"depth":"Bathymetry_depth"})
latlon_df = depth_full[["i","j","lat","lon"]]

sed_cols = [
    "i","j","BLOCKS","BOULDERS","COBBLES","GRAVEL","SAND","SILT","CLAY",
    "BLOCKSIZE","BOULDERSIZE","COBBLESIZE","GRAVELSIZE","SANDSIZE","SILTSIZE","CLAYSIZE"
]
sed_df = pd.read_csv(sediment_file)[sed_cols]

valid_sed = sed_df.dropna()
tree_sed = cKDTree(valid_sed[["i","j"]].values)

# ============================================
# ============================================
print(" Loading Satellite (cache)...")

if not os.path.exists(sat_parquet):

    print(" Creating satellite parquet...")

    Sat = pd.read_csv(sat_csv)
    Sat.columns = Sat.columns.str.strip()

    col_lat = [c for c in Sat.columns if "lat" in c.lower()][0]
    col_lon = [c for c in Sat.columns if "lon" in c.lower()][0]
    col_date = [c for c in Sat.columns if "date" in c.lower()][0]
    col_temp = [c for c in Sat.columns if "temp" in c.lower()][0]

    Sat = Sat.rename(columns={
        col_lat:"Latitude",
        col_lon:"Longitude",
        col_temp:"Satellite_temp"
    })

    Sat["Date"] = pd.to_datetime(Sat[col_date]).dt.normalize()
    Sat = Sat.dropna(subset=["Latitude","Longitude","Satellite_temp"])

    Sat.to_parquet(sat_parquet, index=False)

else:
    print(" Using cached satellite parquet")

Sat = pd.read_parquet(sat_parquet)

tree_sat = cKDTree(Sat[["Latitude","Longitude"]].values)

print(" Satellite ready (fast mode)")

# ============================================
# fix merge
# ============================================
def fix_merge(df):
    for col in list(df.columns):
        if col.endswith("_x"):
            base = col[:-2]
            ycol = base + "_y"
            if ycol in df.columns:
                df[base] = df[ycol]
            else:
                df[base] = df[col]
    df = df.drop(columns=[c for c in df.columns if c.endswith("_x") or c.endswith("_y")])
    return df

# ============================================
# ============================================
print("\n START FAST FILLING...")

files = sorted(
    f for f in glob.glob(os.path.join(input_dir, "Temp_prediction_*.txt"))
    if int(os.path.basename(f).split("_")[2].split(".")[0]) >= START_YEAR
)

for f in files:

    year = os.path.basename(f).split("_")[2].split(".")[0]

    cache_file = os.path.join(cache_dir, f"Temp_{year}.parquet")

    if os.path.exists(cache_file):
        print(f" Skip (cached): {year}")
        continue

    print(f"\n Processing: {year}")

    df = pd.read_csv(f)
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()

    # Combine
    df = df.merge(combine_df, on="Date", how="left")
    df = fix_merge(df)

    # depth
    df = df.merge(depth_df, on=["i","j"], how="left")
    df = fix_merge(df)

    # lat/lon
    df = df.merge(latlon_df, on=["i","j"], how="left")

    # sediment
    df = df.merge(sed_df, on=["i","j"], how="left")
    df = fix_merge(df)

    missing_mask = df["BLOCKS"].isna()
    if missing_mask.any():
        _, idx = tree_sed.query(df.loc[missing_mask, ["i","j"]].values)
        for col in sed_cols[2:]:
            df.loc[missing_mask, col] = valid_sed.iloc[idx][col].values

    # Water_depth
    df["Water_depth"] = df["Water_elevation"] - df["Bathymetry_depth"]

    # ======================================
    # ======================================
    df["Satellite_temp"] = np.nan

    df["Satellite_temp"] = np.nan

    unique_dates = df["Date"].dropna().unique()

    for d in unique_dates:

        mask = df["Date"] == d

        Sat_sub = Sat[Sat["Date"] == d]

        if len(Sat_sub) == 0:
            Sat_sub = Sat[
                (Sat["Date"] >= d - pd.Timedelta(days=3)) &
                (Sat["Date"] <= d + pd.Timedelta(days=3))
                ]

        if len(Sat_sub) == 0:
            continue

        tree = cKDTree(Sat_sub[["Latitude", "Longitude"]].values)

        pts = df.loc[mask, ["lat", "lon"]].values
        _, idx = tree.query(pts)

        df.loc[mask, "Satellite_temp"] = Sat_sub.iloc[idx]["Satellite_temp"].values

    # ======================================
    # distance
    # ======================================
    for river, (ri, rj) in {
        "GreatLakes": (218, 8),
        "OttawaRiver": (218, 3),
        "RichelieuRiver": (167, 40),
        "YamaskaRiver": (128, 111),
        "Saint_FrancoisRiver": (128, 116),
        "NicoletRiver": (55, 220),
        "MaskinongeRiver": (103, 78),
        "DuLoupRiver": (70, 115),
        "YamachicheRiver": (47, 157)
    }.items():
        df[f"In_distance_{river}"] = np.sqrt((df["i"] - ri)**2 + (df["j"] - rj)**2)

    # ======================================
    # ======================================
    df.to_parquet(cache_file, index=False)

    print(f" Saved parquet: {cache_file}")
    print("Satellite ratio:", df["Satellite_temp"].notna().mean())

print("\n===================================")
print(" FAST PIPELINE DONE")
print("===================================")
