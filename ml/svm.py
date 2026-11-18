"""Train and evaluate an RBF SVM for the UCI Internet Advertisements data.

Run from the repository root with: python ml/svm.py
The dataset remains at the repository root; generated files go to ml/artifacts/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "ad.data"
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
CONTINUOUS_FEATURES = ["height", "width", "aratio"]
RANDOM_STATE = 42
VALIDATION_FPR_LIMIT = 0.05
# This minimum makes the precision-first threshold choice guard against a
# threshold that labels almost nothing as an ad. Relaxed only if no candidate
# meeting it can satisfy the strict validation FPR constraint.
MIN_VALIDATION_RECALL = 0.50


def load_dataset(path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Read all source rows, retaining duplicates and encoding ad as 1."""
    raw = pd.read_csv(
        path, header=None, skipinitialspace=True, na_values="?", low_memory=False
    )
    if raw.shape != (3279, 1559):
        raise ValueError(f"Expected 3279 rows and 1559 columns; found {raw.shape}.")

    X = raw.iloc[:, :-1].copy()
    X.columns = CONTINUOUS_FEATURES + [
        f"binary_{index}" for index in range(1, X.shape[1] - 2)
    ]
    y_text = raw.iloc[:, -1].astype("string").str.strip().str.rstrip(".")
    labels = {"ad": 1, "nonad": 0}
    unexpected = set(y_text.dropna().unique()) - set(labels)
    if unexpected or y_text.isna().any():
        raise ValueError(f"Unexpected or missing target labels: {unexpected}")
    y = y_text.map(labels).astype("int8")
    for column in X.columns:
        X[column] = pd.to_numeric(X[column], errors="coerce")
    return X, y


def build_pipeline(feature_names: list[str]) -> Pipeline:
    """Impute and scale only continuous columns; impute binary columns only."""
    binary_features = [name for name in feature_names if name not in CONTINUOUS_FEATURES]
    preprocess = ColumnTransformer(
        transformers=[
            (
                "continuous",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="mean")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                CONTINUOUS_FEATURES,
            ),
            (
                "binary",
                SimpleImputer(strategy="most_frequent"),
                binary_features,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline(
        [
            ("preprocessor", preprocess),
            (
                "classifier",
                SVC(kernel="rbf", class_weight="balanced", decision_function_shape="ovr"),
            ),
        ]
    )


def false_positive_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, _, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(fp / (fp + tn)) if fp + tn else 0.0


def choose_threshold(
    y_true: np.ndarray, scores: np.ndarray
) -> tuple[float, dict[str, float]]:
    """Favor precision subject to strict FPR and a useful recall floor."""
    candidates = np.r_[np.nextafter(scores.max(), np.inf), np.unique(scores)[::-1]]
    eligible: list[tuple[float, float, float, float]] = []
    fpr_eligible: list[tuple[float, float, float, float]] = []
    for threshold in candidates:
        pred = (scores >= threshold).astype(int)
        fpr = false_positive_rate(y_true, pred)
        if fpr < VALIDATION_FPR_LIMIT:
            row = (
                precision_score(y_true, pred, zero_division=0),
                recall_score(y_true, pred, zero_division=0),
                float(threshold),
                fpr,
            )
            fpr_eligible.append(row)
            if row[1] >= MIN_VALIDATION_RECALL:
                eligible.append(row)

    if eligible:
        # Primary objective is precision; recall then threshold break ties.
        best = max(eligible, key=lambda item: (item[0], item[1], item[2]))
    elif fpr_eligible:
        # Preserve the FPR guarantee if recall floor proves unattainable.
        best = max(fpr_eligible, key=lambda item: (item[1], item[0], item[2]))
        print(
            "Warning: no threshold met the requested recall floor while keeping "
            "validation FPR below 5%; selected the eligible threshold with "
            "highest recall."
        )
    else:
        raise RuntimeError("No threshold achieved validation FPR strictly below 5%.")

    precision, recall, threshold, fpr = best
    return threshold, {
        "precision_ad": precision,
        "recall_ad": recall,
        "false_positive_rate": fpr,
    }


def main() -> None:
    X, y = load_dataset()
    # 60% train, 20% validation, 20% test. All splits preserve class balance.
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y, test_size=0.40, random_state=RANDOM_STATE, stratify=y
    )
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_holdout,
        y_holdout,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_holdout,
    )

    model = build_pipeline(list(X.columns))
    model.fit(X_train, y_train)

    validation_scores = model.decision_function(X_validation)
    threshold, validation_metrics = choose_threshold(
        y_validation.to_numpy(), validation_scores
    )

    test_scores = model.decision_function(X_test)
    test_pred = (test_scores >= threshold).astype(int)
    matrix = confusion_matrix(y_test, test_pred, labels=[0, 1])
    metrics = {
        "dataset": str(DATA_PATH.relative_to(ROOT)),
        "rows_preserved": int(len(X)),
        "duplicates_removed": 0,
        "feature_count": int(X.shape[1]),
        "class_mapping": {"nonad": 0, "ad": 1},
        "split_sizes": {
            "train": int(len(X_train)),
            "validation": int(len(X_validation)),
            "test": int(len(X_test)),
        },
        "model": "RBF-kernel SVC with class_weight='balanced'",
        "preprocessing": {
            "continuous": "mean imputation then standard scaling",
            "binary": "most-frequent imputation; unscaled",
        },
        "threshold_selection": {
            "threshold": threshold,
            "validation_fpr_strictly_below": VALIDATION_FPR_LIMIT,
            "minimum_validation_ad_recall_target": MIN_VALIDATION_RECALL,
            "validation_metrics": validation_metrics,
        },
        "test_metrics": {
            "advertisement_precision": float(
                precision_score(y_test, test_pred, zero_division=0)
            ),
            "advertisement_recall": float(
                recall_score(y_test, test_pred, zero_division=0)
            ),
            "f1_score": float(f1_score(y_test, test_pred, zero_division=0)),
            "accuracy": float(accuracy_score(y_test, test_pred)),
            "roc_auc": float(roc_auc_score(y_test, test_scores)),
            "false_positive_rate": false_positive_rate(y_test.to_numpy(), test_pred),
        },
        "confusion_matrix": {
            "labels": ["nonad (0)", "ad (1)"],
            "rows_actual_columns_predicted": matrix.tolist(),
        },
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    dump({"pipeline": model, "threshold": threshold}, ARTIFACT_DIR / "svm_model.joblib")
    (ARTIFACT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix, display_labels=["nonad", "ad"]
    )
    display.plot(cmap="Blues", values_format="d")
    plt.title("RBF SVM Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

    print(json.dumps(metrics, indent=2))
    print(f"\nSaved artifacts to: {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
