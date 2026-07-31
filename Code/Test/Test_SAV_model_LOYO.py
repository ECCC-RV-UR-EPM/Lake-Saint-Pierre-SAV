"""Evaluate the annual SAV classifier using leave-one-year-out cross-validation."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parent.parent
INPUT_DIR = PACKAGE_ROOT / "Data" / "Test"
OUTPUT_DIR = HERE / "outputs"

TRAIN_FILE = INPUT_DIR / "cascade_training_table.parquet"
NORTH_POLYGON_FILE = INPUT_DIR / "north_polygon_paper.json"
MODEL_BUNDLE_FILE = INPUT_DIR / "cascade_v3_model_bundle.pkl"


def load_north_polygon() -> MplPath:
    vertices = json.loads(NORTH_POLYGON_FILE.read_text(encoding="utf-8"))
    return MplPath(vertices)


def load_model_bundle() -> dict:
    with MODEL_BUNDLE_FILE.open("rb") as file:
        bundle = pickle.load(file)

    required_keys = {"model", "imputer", "threshold", "features"}
    missing_keys = required_keys.difference(bundle)
    if missing_keys:
        raise ValueError(
            "Model bundle is missing keys: " + ", ".join(sorted(missing_keys))
        )
    return bundle


def add_model_features(
    df: pd.DataFrame,
    polygon: MplPath,
    features: list[str],
) -> pd.DataFrame:
    out = df.copy()
    if "north_flag" in features:
        out["north_flag"] = polygon.contains_points(
            out[["j", "i"]].to_numpy()
        ).astype(int)
    if "tss_x_year" in features:
        out["tss_x_year"] = out["TSS_pred"] * out["Year_norm"]
    return out


def threshold_from_youden(y_true: np.ndarray, probability: np.ndarray) -> float:
    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        y_true, probability
    )
    return float(
        thresholds[np.argmax(true_positive_rate - false_positive_rate)]
    )


def calculate_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> tuple[dict[str, float | int], np.ndarray]:
    prediction = (probability >= threshold).astype(int)
    matrix = confusion_matrix(y_true, prediction, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    specificity = (
        true_negative / (true_negative + false_positive)
        if (true_negative + false_positive)
        else np.nan
    )

    metrics: dict[str, float | int] = {
        "Threshold": float(threshold),
        "Accuracy": float(accuracy_score(y_true, prediction)),
        "Balanced_accuracy": float(
            balanced_accuracy_score(y_true, prediction)
        ),
        "Precision": float(
            precision_score(y_true, prediction, zero_division=0)
        ),
        "Recall": float(recall_score(y_true, prediction, zero_division=0)),
        "Specificity": float(specificity),
        "F1": float(f1_score(y_true, prediction, zero_division=0)),
        "ROC_AUC": float(roc_auc_score(y_true, probability)),
        "PR_AUC": float(average_precision_score(y_true, probability)),
        "TN": int(true_negative),
        "FP": int(false_positive),
        "FN": int(false_negative),
        "TP": int(true_positive),
        "Samples": int(len(y_true)),
    }
    return metrics, matrix


def run_loyo(
    training: pd.DataFrame,
    features: list[str],
    model_template: RandomForestClassifier,
    imputer_template: SimpleImputer,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    probabilities = np.full(len(training), np.nan, dtype=float)
    fold_records: list[dict[str, int]] = []
    fold_metric_records: list[dict[str, float | int]] = []

    for test_year in sorted(training["Year"].unique()):
        train_mask = training["Year"] != test_year
        test_mask = ~train_mask

        imputer = clone(imputer_template)
        x_train = imputer.fit_transform(
            training.loc[train_mask, features]
        )
        x_test = imputer.transform(training.loc[test_mask, features])
        y_train = (
            training.loc[train_mask, "SAV"].astype(int).to_numpy()
        )
        y_test = (
            training.loc[test_mask, "SAV"].astype(int).to_numpy()
        )

        classifier = clone(model_template)
        classifier.set_params(n_jobs=1)
        classifier.fit(x_train, y_train)
        fold_probability = classifier.predict_proba(x_test)[:, 1]
        probabilities[np.flatnonzero(test_mask.to_numpy())] = fold_probability

        fold_threshold = (
            threshold_from_youden(y_test, fold_probability)
            if len(np.unique(y_test)) > 1
            else 0.5
        )
        fold_metrics, _ = calculate_metrics(
            y_test,
            fold_probability,
            fold_threshold,
        )
        fold_metric_records.append(
            {
                "Year": int(test_year),
                "Accuracy": fold_metrics["Accuracy"],
                "BAcc": fold_metrics["Balanced_accuracy"],
                "Precision": fold_metrics["Precision"],
                "Recall": fold_metrics["Recall"],
                "Specificity": fold_metrics["Specificity"],
                "F1": fold_metrics["F1"],
                "ROC_AUC": fold_metrics["ROC_AUC"],
                "PR_AUC": fold_metrics["PR_AUC"],
                "Threshold_fold": float(fold_threshold),
            }
        )

        fold_records.append(
            {
                "Test_year": int(test_year),
                "Training_samples": int(train_mask.sum()),
                "Test_samples": int(test_mask.sum()),
            }
        )
        print(f"Completed LOYO test year: {int(test_year)}")

    if np.isnan(probabilities).any():
        raise RuntimeError("Some LOYO predictions were not generated.")

    return (
        probabilities,
        pd.DataFrame(fold_records),
        pd.DataFrame(fold_metric_records),
    )


def main() -> None:
    for required_file in (
        TRAIN_FILE,
        NORTH_POLYGON_FILE,
        MODEL_BUNDLE_FILE,
    ):
        if not required_file.exists():
            raise FileNotFoundError(f"Required input not found: {required_file}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    bundle = load_model_bundle()
    features = list(bundle["features"])
    model_template = bundle["model"]
    imputer_template = bundle["imputer"]

    print(f"Model features ({len(features)}): {', '.join(features)}")

    training = pd.read_parquet(TRAIN_FILE)
    training = add_model_features(
        training,
        load_north_polygon(),
        features,
    )

    missing_columns = [
        column
        for column in [*features, "Year", "SAV"]
        if column not in training.columns
    ]
    if missing_columns:
        raise ValueError(
            "Required columns are missing: " + ", ".join(missing_columns)
        )

    probabilities, fold_summary, fold_metrics = run_loyo(
        training,
        features,
        model_template,
        imputer_template,
    )
    truth = training["SAV"].astype(int).to_numpy()

    pooled_threshold = threshold_from_youden(truth, probabilities)
    pooled_metrics, pooled_matrix = calculate_metrics(
        truth, probabilities, pooled_threshold
    )

    pd.DataFrame([pooled_metrics]).to_csv(
        OUTPUT_DIR / "sav_loyo_performance_metrics.csv", index=False
    )
    pd.DataFrame(
        pooled_matrix,
        index=["Observed_absence", "Observed_presence"],
        columns=["Predicted_absence", "Predicted_presence"],
    ).to_csv(OUTPUT_DIR / "sav_loyo_confusion_matrix.csv")

    fold_metrics.to_csv(
        OUTPUT_DIR / "sav_loyo_metrics_by_year.csv", index=False
    )
    fold_summary.to_csv(
        OUTPUT_DIR / "sav_loyo_fold_summary.csv", index=False
    )

    print("\nPooled LOYO performance")
    print(pd.DataFrame([pooled_metrics]).to_string(index=False))
    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
