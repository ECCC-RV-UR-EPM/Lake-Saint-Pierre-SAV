from __future__ import annotations

import pickle

import pandas as pd

from sav_annual_common import (
    FEATURES,
    GRID_SRC,
    MAP_DIR,
    OUT_DIR,
    YEARS,
    annual_prediction,
    panel,
    save_feature_importance,
    summarize_maps,
    train_baseline_model,
)


def main() -> None:
    clf, imp, thr, loyo_df = train_baseline_model()
    loyo_df.to_csv(OUT_DIR / "cascade_v3_loyo_per_year.csv", index=False)
    save_feature_importance(clf, OUT_DIR / "cascade_v3_feature_importance.csv")
    with (OUT_DIR / "cascade_v3_model_bundle.pkl").open("wb") as f:
        pickle.dump(
            {
                "model": clf,
                "imputer": imp,
                "threshold": thr,
                "features": FEATURES,
            },
            f,
        )

    maps = {}
    for year in YEARS:
        annual = pd.read_parquet(GRID_SRC / f"cascade_grid_{year}.parquet")
        maps[year] = annual_prediction(clf, imp, thr, annual)
        maps[year].to_parquet(MAP_DIR / f"cascade_v3_baseline_map_{year}.parquet", index=False)

    north_df = summarize_maps(maps)
    north_df.to_csv(OUT_DIR / "cascade_v3_north_timeseries.csv", index=False)
    panel(maps, MAP_DIR / "SAV_ALL_YEARS_cascade_v3_baseline.png", "Annual SAV baseline")

    summary = pd.DataFrame(
        [
            {
                "Model": "cascade_single_model_v3",
                "North_2002_km2": float(north_df.loc[north_df.Year == 2002, "North_km2"].iloc[0]),
                "North_2024_km2": float(north_df.loc[north_df.Year == 2024, "North_km2"].iloc[0]),
                "North_decline_pct_2002_2024": float(
                    100
                    * (
                        1
                        - north_df.loc[north_df.Year == 2024, "North_km2"].iloc[0]
                        / north_df.loc[north_df.Year == 2002, "North_km2"].iloc[0]
                    )
                ),
            }
        ]
    )
    summary.to_csv(OUT_DIR / "cascade_v3_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
