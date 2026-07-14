from __future__ import annotations

import pickle

import pandas as pd

from sav_annual_common import GRID_SRC, MAP_DIR, OUT_DIR, YEARS, annual_prediction


MODEL_BUNDLE = OUT_DIR / "cascade_v3_model_bundle.pkl"


def main() -> None:
    with MODEL_BUNDLE.open("rb") as f:
        bundle = pickle.load(f)

    clf = bundle["model"]
    imp = bundle["imputer"]
    thr = float(bundle["threshold"])
    features = bundle.get("features")

    for year in YEARS:
        annual = pd.read_parquet(GRID_SRC / f"cascade_grid_{year}.parquet")
        pred = annual_prediction(clf, imp, thr, annual, features)
        pred.to_parquet(MAP_DIR / f"cascade_v3_baseline_map_{year}.parquet", index=False)

    print(f"Exported baseline map parquet files to: {MAP_DIR}")


if __name__ == "__main__":
    main()
