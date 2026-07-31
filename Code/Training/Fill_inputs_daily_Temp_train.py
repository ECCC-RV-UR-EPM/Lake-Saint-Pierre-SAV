import os
import sys

PAPER_PYTHON = r"C:\Users\qiwan\miniconda3\envs\sentinel\python.exe"
if os.environ.get("TEMP_PAPER_ENV_ACTIVE") != "1":
    try:
        import pandas as _pd_check
        import scipy as _scipy_check
        versions_ok = (
            _pd_check.__version__ == "2.3.3"
            and _scipy_check.__version__ == "1.15.2"
        )
    except ImportError:
        versions_ok = False
    if not versions_ok:
        if not os.path.exists(PAPER_PYTHON):
            raise RuntimeError(
                "The historical workflow requires pandas 2.3.3 and SciPy 1.15.2"
            )
        os.environ["TEMP_PAPER_ENV_ACTIVE"] = "1"
        os.execv(PAPER_PYTHON, [PAPER_PYTHON, __file__])

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


base_dir = r"E:\SAV_map_PNAS\Submission_package\Data\01_temperature"

input_file = os.path.join(
    base_dir, "Temp_training_daily_observe_to_train_clip.xlsx"
)
grid_file = os.path.join(base_dir, "lat_lon_UMT_i_j.csv")
combine_file = os.path.join(base_dir, "Combine.xlsx")
joined_file = os.path.join(base_dir, "JOINED_with_ij.csv")
satellite_file = os.path.join(
    base_dir, "LSP_LSWT_200m_rev02_2000-2024_satellite.csv"
)

output_dir = os.environ.get("TEMP_FILL_OUTPUT_DIR", base_dir)
output_file = os.path.join(
    output_dir, "Temp_training_daily_observe_to_train_clip_filled.xlsx"
)


def clean_columns(frame):
    frame = frame.copy()
    frame.columns = (
        frame.columns.astype(str)
        .str.strip()
        .str.replace(" ", "_", regex=False)
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
    )
    return frame


def normalize_date(values):
    return pd.to_datetime(values, errors="coerce").dt.normalize()


def fill_only_missing(target, column, values):
    if column not in target.columns:
        return
    candidate = pd.Series(values, index=target.index)
    missing = target[column].isna() & candidate.notna()
    target.loc[missing, column] = candidate.loc[missing]


# =========================================================
# 1. Template
# =========================================================
df = clean_columns(pd.read_excel(input_file))
original_columns = df.columns.tolist()
df["Date"] = normalize_date(df["Date"])


# =========================================================
# 2. Latitude/longitude -> nearest model i/j
# =========================================================
grid = clean_columns(pd.read_csv(grid_file))
grid_tree = cKDTree(grid[["lat", "lon"]].to_numpy())
_, grid_idx = grid_tree.query(df[["Latitude", "Longitude"]].to_numpy())
nearest_grid = grid.iloc[grid_idx].reset_index(drop=True)

fill_only_missing(df, "i", nearest_grid["i"].astype(int))
fill_only_missing(df, "j", nearest_grid["j"].astype(int))

i_key = pd.to_numeric(df["i"], errors="coerce").round().astype("Int64")
j_key = pd.to_numeric(df["j"], errors="coerce").round().astype("Int64")


# =========================================================
# 3. Exact-date variables from Combine.xlsx
# =========================================================
combine = clean_columns(pd.read_excel(combine_file))
combine["Date"] = normalize_date(combine["Date"])
combine = combine.drop_duplicates("Date").set_index("Date")

combine_columns = [
    column
    for column in original_columns
    if column in combine.columns and column not in ("Date", "Water_temp")
]
for column in combine_columns:
    fill_only_missing(df, column, df["Date"].map(combine[column]))


# =========================================================
# 4. Bathymetry
# =========================================================
depth_column = next(column for column in grid.columns if "depth" in column.lower())
fill_only_missing(df, "Bathymetry_depth", nearest_grid[depth_column])


# =========================================================
# 5. Soil — exact i/j join, matching the original MATLAB logic
# =========================================================
joined = clean_columns(pd.read_csv(joined_file))
joined["i"] = pd.to_numeric(joined["i"], errors="coerce").round().astype("Int64")
joined["j"] = pd.to_numeric(joined["j"], errors="coerce").round().astype("Int64")
joined = joined.dropna(subset=["i", "j"]).drop_duplicates(["i", "j"])
joined = joined.set_index(["i", "j"])

soil_columns = [
    column
    for column in original_columns
    if column in joined.columns
    and any(
        token in column.upper()
        for token in ("BLOCK", "BOULDER", "COBBLE", "GRAVEL", "SAND", "SILT", "CLAY")
    )
]
ij_keys = pd.MultiIndex.from_arrays([i_key, j_key], names=["i", "j"])
for column in soil_columns:
    values = pd.Series(joined[column].reindex(ij_keys).to_numpy(), index=df.index)
    fill_only_missing(df, column, values)


# =========================================================
# 6. Satellite temperature — historical paper workflow.
# Use the exact date first. Only when that date has no valid pixels, pool all
# valid pixels within +/-3 days and select the spatial nearest neighbour.
# =========================================================
satellite = clean_columns(pd.read_csv(satellite_file))
satellite = satellite.rename(
    columns={
        "lat": "Latitude",
        "long": "Longitude",
        "temperature_C": "Satellite_temp",
    }
)
satellite["Date"] = pd.to_datetime(
    satellite["date"], errors="coerce"
).dt.normalize()
satellite["Satellite_temp"] = pd.to_numeric(
    satellite["Satellite_temp"], errors="coerce"
)
satellite = satellite.dropna(
    subset=["Latitude", "Longitude", "Date", "Satellite_temp"]
)

satellite_by_date = {
    date: group for date, group in satellite.groupby("Date", sort=False)
}
available_dates = pd.DatetimeIndex(sorted(satellite_by_date))
satellite_values = pd.Series(np.nan, index=df.index, dtype=float)

# The historical routine queried model-grid coordinates, not the raw
# observation coordinates.
target_coordinates = df[["Latitude", "Longitude"]].rename(
    columns={"Latitude": "lat", "Longitude": "lon"}
).reset_index(drop=True)

for observation_date, row_indices in df.groupby("Date").groups.items():
    candidates = satellite_by_date.get(observation_date)

    if candidates is None or candidates.empty:
        window_dates = available_dates[
            (available_dates >= observation_date - pd.Timedelta(days=3))
            & (available_dates <= observation_date + pd.Timedelta(days=3))
        ]
        if len(window_dates) == 0:
            continue
        candidates = pd.concat(
            [satellite_by_date[date] for date in window_dates],
            ignore_index=True,
        )

    satellite_tree = cKDTree(
        candidates[["Latitude", "Longitude"]].to_numpy()
    )
    _, nearest_indices = satellite_tree.query(
        target_coordinates.loc[row_indices, ["lat", "lon"]].to_numpy()
    )
    satellite_values.loc[row_indices] = candidates.iloc[nearest_indices][
        "Satellite_temp"
    ].to_numpy()

fill_only_missing(df, "Satellite_temp", satellite_values)

# =========================================================
# 7. Water depth and river distances
# =========================================================
if {"Water_depth", "Water_elevation", "Bathymetry_depth"}.issubset(df.columns):
    water_depth = (
        pd.to_numeric(df["Water_elevation"], errors="coerce")
        - pd.to_numeric(df["Bathymetry_depth"], errors="coerce")
    )
    fill_only_missing(df, "Water_depth", water_depth)

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
for name, (river_i, river_j) in rivers.items():
    distances = np.sqrt(
        (i_key.astype(float) - river_i) ** 2
        + (j_key.astype(float) - river_j) ** 2
    )
    fill_only_missing(df, f"In_distance_{name}", distances)


# =========================================================
# 8. Save the original 43-column structure
# =========================================================
os.makedirs(output_dir, exist_ok=True)
df = df[original_columns]
df.to_excel(output_file, index=False)

print("DONE")
print("Rows, columns:", df.shape)
print("Satellite_temp non-null:", int(df["Satellite_temp"].notna().sum()))
print("Total blank cells:", int(df.isna().sum().sum()))
print("Saved:", output_file)
