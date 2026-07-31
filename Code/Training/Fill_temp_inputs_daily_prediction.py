"""Reproduce the paper-version temperature input parquet files.

The historical cache was generated with SciPy 1.15.2. Its cKDTree
tie-breaking differs from SciPy 1.14.1 when several observations are at
exactly the same distance.

Generated parquet files are saved by default in:

Data/01_temperature/Prediction/parquet_cache
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow
import scipy
from scipy.spatial import cKDTree


# ============================================================
# Required historical environment
# ============================================================

REQUIRED_VERSIONS = {
    "scipy": "1.15.2",
    "pandas": "2.3.3",
    "pyarrow": "18.1.0",
}


# ============================================================
# Paths
# ============================================================

# Submission_package can be placed anywhere.
PACKAGE_DIR = Path(__file__).resolve().parents[1]

BASE_DIR = PACKAGE_DIR / "Data" / "01_temperature"
INPUT_DIR = BASE_DIR / "Prediction"

# All generated annual parquet files are saved here.
CACHE_DIR = INPUT_DIR / "parquet_cache"

COMBINE_FILE = BASE_DIR / "Combine.xlsx"
DEPTH_FILE = BASE_DIR / "lat_lon_UMT_i_j.csv"
SEDIMENT_FILE = BASE_DIR / "JOINED_with_ij.csv"

SATELLITE_CSV = (
    BASE_DIR
    / "LSP_LSWT_200m_rev02_2000-2024_satellite.csv"
)

SATELLITE_PARQUET = CACHE_DIR / "satellite.parquet"


# ============================================================
# Sediment variables
# ============================================================

SEDIMENT_COLUMNS = [
    "i",
    "j",
    "BLOCKS",
    "BOULDERS",
    "COBBLES",
    "GRAVEL",
    "SAND",
    "SILT",
    "CLAY",
    "BLOCKSIZE",
    "BOULDERSIZE",
    "COBBLESIZE",
    "GRAVELSIZE",
    "SANDSIZE",
    "SILTSIZE",
    "CLAYSIZE",
]


# ============================================================
# River coordinates
# ============================================================

RIVER_COORDINATES = {
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


# ============================================================
# Command-line arguments
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--year",
        type=int,
        action="append",
        help=(
            "Process only this year. Repeat this option "
            "to process multiple years."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CACHE_DIR,
        help=(
            "Directory for generated parquet files. "
            "Default: Data/01_temperature/Prediction/"
            "parquet_cache"
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing Temp_YYYY.parquet files.",
    )

    return parser.parse_args()


# ============================================================
# Environment verification
# ============================================================

def require_historical_environment() -> None:
    actual_versions = {
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "pyarrow": pyarrow.__version__,
    }

    wrong_versions = {
        package: (
            actual_versions[package],
            required_version,
        )
        for package, required_version
        in REQUIRED_VERSIONS.items()
        if actual_versions[package] != required_version
    }

    if wrong_versions:
        details = ", ".join(
            (
                f"{package}={found_version} "
                f"(required {required_version})"
            )
            for package, (
                found_version,
                required_version,
            )
            in wrong_versions.items()
        )

        raise RuntimeError(
            "Historical nearest-neighbour results require "
            "the paper environment: "
            + details
            + ". Please run this script using the Python "
              "environment containing the required versions."
        )

    print("Environment verified:")
    print(f"  SciPy:   {actual_versions['scipy']}")
    print(f"  pandas:  {actual_versions['pandas']}")
    print(f"  PyArrow: {actual_versions['pyarrow']}")


# ============================================================
# Merge helper
# ============================================================

def fix_merge(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    for column in list(frame.columns):
        if column.endswith("_x"):
            base_column = column[:-2]
            other_column = base_column + "_y"

            if other_column in frame.columns:
                frame[base_column] = frame[other_column]
            else:
                frame[base_column] = frame[column]

    duplicate_columns = [
        column
        for column in frame.columns
        if column.endswith(("_x", "_y"))
    ]

    return frame.drop(
        columns=duplicate_columns
    )


# ============================================================
# Satellite data
# ============================================================

def load_satellite(
    output_dir: Path,
) -> pd.DataFrame:
    output_satellite_parquet = (
        output_dir / "satellite.parquet"
    )

    # Normally this is:
    # Prediction/parquet_cache/satellite.parquet
    if output_satellite_parquet.exists():
        print(
            "Using satellite parquet cache:"
        )
        print(f"  {output_satellite_parquet}")

        satellite = pd.read_parquet(
            output_satellite_parquet
        )

    elif SATELLITE_PARQUET.exists():
        print(
            "Using packaged satellite parquet cache:"
        )
        print(f"  {SATELLITE_PARQUET}")

        satellite = pd.read_parquet(
            SATELLITE_PARQUET
        )

    else:
        print(
            "Satellite parquet was not found."
        )
        print(
            "Creating satellite parquet from CSV..."
        )

        satellite = pd.read_csv(
            SATELLITE_CSV
        )

        satellite.columns = (
            satellite.columns.str.strip()
        )

        latitude_column = next(
            column
            for column in satellite.columns
            if "lat" in column.lower()
        )

        longitude_column = next(
            column
            for column in satellite.columns
            if "lon" in column.lower()
        )

        date_column = next(
            column
            for column in satellite.columns
            if "date" in column.lower()
        )

        temperature_column = next(
            column
            for column in satellite.columns
            if "temp" in column.lower()
        )

        satellite = satellite.rename(
            columns={
                latitude_column: "Latitude",
                longitude_column: "Longitude",
                temperature_column: (
                    "Satellite_temp"
                ),
            }
        )

        satellite["Date"] = pd.to_datetime(
            satellite[date_column]
        ).dt.normalize()

        satellite = satellite.dropna(
            subset=[
                "Latitude",
                "Longitude",
                "Satellite_temp",
            ]
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        satellite.to_parquet(
            output_satellite_parquet,
            index=False,
        )

        print(
            "Satellite parquet saved:"
        )
        print(
            f"  {output_satellite_parquet}"
        )

    satellite["Date"] = pd.to_datetime(
        satellite["Date"]
    ).dt.normalize()

    print(
        f"Satellite rows: {len(satellite):,}"
    )

    return satellite


# ============================================================
# Process one year
# ============================================================

def process_year(
    source_file: Path,
    output_file: Path,
    combine: pd.DataFrame,
    depth: pd.DataFrame,
    latlon: pd.DataFrame,
    sediment: pd.DataFrame,
    valid_sediment: pd.DataFrame,
    sediment_tree: cKDTree,
    satellite: pd.DataFrame,
) -> None:
    frame = pd.read_csv(
        source_file
    )

    frame["Date"] = pd.to_datetime(
        frame["Date"]
    ).dt.normalize()

    # --------------------------------------------------------
    # Daily environmental variables
    # --------------------------------------------------------

    frame = fix_merge(
        frame.merge(
            combine,
            on="Date",
            how="left",
        )
    )

    # --------------------------------------------------------
    # Bathymetry
    # --------------------------------------------------------

    frame = fix_merge(
        frame.merge(
            depth,
            on=["i", "j"],
            how="left",
        )
    )

    # --------------------------------------------------------
    # Latitude and longitude
    # --------------------------------------------------------

    frame = frame.merge(
        latlon,
        on=["i", "j"],
        how="left",
    )

    # --------------------------------------------------------
    # Sediment
    # --------------------------------------------------------

    frame = fix_merge(
        frame.merge(
            sediment,
            on=["i", "j"],
            how="left",
        )
    )

    missing_sediment = (
        frame["BLOCKS"].isna()
    )

    if missing_sediment.any():
        print(
            "Filling missing sediment rows: "
            f"{missing_sediment.sum():,}"
        )

        _, nearest_indices = (
            sediment_tree.query(
                frame.loc[
                    missing_sediment,
                    ["i", "j"],
                ].values
            )
        )

        for column in SEDIMENT_COLUMNS[2:]:
            frame.loc[
                missing_sediment,
                column,
            ] = (
                valid_sediment
                .iloc[nearest_indices][column]
                .values
            )

    # --------------------------------------------------------
    # Water depth
    # --------------------------------------------------------

    frame["Water_depth"] = (
        frame["Water_elevation"]
        - frame["Bathymetry_depth"]
    )

    # --------------------------------------------------------
    # Satellite temperature
    # --------------------------------------------------------

    frame["Satellite_temp"] = np.nan

    unique_dates = (
        frame["Date"]
        .dropna()
        .unique()
    )

    for date in unique_dates:
        date_mask = (
            frame["Date"] == date
        )

        # Use the exact date first.
        candidates = satellite[
            satellite["Date"] == date
        ]

        # If the exact date is unavailable,
        # use observations within plus/minus 3 days.
        if candidates.empty:
            candidates = satellite[
                satellite["Date"].between(
                    date - pd.Timedelta(days=3),
                    date + pd.Timedelta(days=3),
                )
            ]

        if candidates.empty:
            continue

        satellite_tree = cKDTree(
            candidates[
                ["Latitude", "Longitude"]
            ].values
        )

        target_points = frame.loc[
            date_mask,
            ["lat", "lon"],
        ].values

        _, nearest_indices = (
            satellite_tree.query(
                target_points
            )
        )

        frame.loc[
            date_mask,
            "Satellite_temp",
        ] = (
            candidates
            .iloc[nearest_indices][
                "Satellite_temp"
            ]
            .values
        )

    # --------------------------------------------------------
    # Distance to river inputs
    # --------------------------------------------------------

    for river, (
        river_i,
        river_j,
    ) in RIVER_COORDINATES.items():
        frame[
            f"In_distance_{river}"
        ] = np.sqrt(
            (frame["i"] - river_i) ** 2
            + (frame["j"] - river_j) ** 2
        )

    # --------------------------------------------------------
    # Save annual parquet
    # --------------------------------------------------------

    frame.to_parquet(
        output_file,
        index=False,
    )

    satellite_coverage = (
        frame["Satellite_temp"]
        .notna()
        .mean()
    )

    print(f"Saved: {output_file}")
    print(
        "Satellite coverage: "
        f"{satellite_coverage:.6f}"
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()

    require_historical_environment()

    output_dir = (
        args.output_dir.resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Annual parquet output directory:"
    )
    print(f"  {output_dir}")

    # --------------------------------------------------------
    # Load daily environmental data
    # --------------------------------------------------------

    print("Loading Combine.xlsx...")

    combine = pd.read_excel(
        COMBINE_FILE
    )

    combine.columns = (
        combine.columns.str.strip()
    )

    combine["Date"] = pd.to_datetime(
        combine["Date"]
    ).dt.normalize()

    # --------------------------------------------------------
    # Load bathymetry and coordinates
    # --------------------------------------------------------

    print(
        "Loading bathymetry and coordinates..."
    )

    depth_full = pd.read_csv(
        DEPTH_FILE
    )

    depth = (
        depth_full[
            ["i", "j", "depth"]
        ]
        .rename(
            columns={
                "depth": "Bathymetry_depth"
            }
        )
    )

    latlon = depth_full[
        ["i", "j", "lat", "lon"]
    ]

    # --------------------------------------------------------
    # Load sediment data
    # --------------------------------------------------------

    print("Loading sediment data...")

    sediment = pd.read_csv(
        SEDIMENT_FILE
    )[SEDIMENT_COLUMNS]

    valid_sediment = (
        sediment.dropna()
    )

    sediment_tree = cKDTree(
        valid_sediment[
            ["i", "j"]
        ].values
    )

    # --------------------------------------------------------
    # Load satellite data
    # --------------------------------------------------------

    satellite = load_satellite(
        output_dir
    )

    # --------------------------------------------------------
    # Find annual source files
    # --------------------------------------------------------

    source_pattern = (
        INPUT_DIR
        / "Temp_prediction_*.txt"
    )

    source_files = [
        Path(path)
        for path in sorted(
            glob.glob(
                str(source_pattern)
            )
        )
    ]

    # Process selected years only,
    # when --year is supplied.
    if args.year:
        selected_years = set(
            args.year
        )

        source_files = [
            path
            for path in source_files
            if int(
                path.stem.split("_")[2]
            )
            in selected_years
        ]

    if not source_files:
        raise FileNotFoundError(
            "No matching "
            "Temp_prediction_YYYY.txt "
            "files were found."
        )

    print(
        "Years to process:",
        [
            int(
                path.stem.split("_")[2]
            )
            for path in source_files
        ],
    )

    # --------------------------------------------------------
    # Process annual files
    # --------------------------------------------------------

    for source_file in source_files:
        year = int(
            source_file
            .stem
            .split("_")[2]
        )

        output_file = (
            output_dir
            / f"Temp_{year}.parquet"
        )

        if (
            output_file.exists()
            and not args.overwrite
        ):
            print(
                "Skip existing output: "
                f"{output_file}"
            )
            continue

        print()
        print(
            "==================================="
        )
        print(f"Processing {year}...")
        print(
            "==================================="
        )

        process_year(
            source_file=source_file,
            output_file=output_file,
            combine=combine,
            depth=depth,
            latlon=latlon,
            sediment=sediment,
            valid_sediment=valid_sediment,
            sediment_tree=sediment_tree,
            satellite=satellite,
        )

    print()
    print(
        "==================================="
    )
    print(
        "Annual temperature input "
        "construction completed."
    )
    print(
        "==================================="
    )
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()