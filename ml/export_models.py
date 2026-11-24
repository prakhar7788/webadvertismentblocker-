"""OPTIONAL helper to save the missing trained models as .joblib artifacts.

This script is NOT required to run the demo. The demo already works with the
saved SVM model. Use this only if you want all four models loaded and you do not
mind retraining the three that were never persisted.

WHAT IT DOES
------------
* Leaves the existing `ml/artifacts/svm_model.joblib` untouched (the SVM is reused,
  never retrained or replaced).
* For each missing model (Logistic Regression, Bernoulli Naive Bayes, Random
  Forest) it trains that model with the same leak-free ColumnTransformer pipeline
  and the best hyperparameters found in the notebooks, selects a precision-focused
  threshold (validation FPR <= 5 %, recall floor 0.50) and saves it to
  `ml/artifacts/` in the format the backend expects.

NOTE: this RETRAINS the three models (random_state=42, same pipeline as the
notebooks, so the resulting models match the notebooks). If you would rather not
retrain, open each notebook and add a one-line `joblib.dump(...)` cell instead —
see the README.

Run from the project root:   python ml/export_models.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
MAX_FPR = 0.05
MIN_RECALL = 0.50
CONTINUOUS = ["height", "width", "aratio"]

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "ml" / "dataset" / "ad.data"
ART = ROOT / "ml" / "artifacts"

# (filename, label, score_func, builder-factory)
MODELS = {
    "logreg_model.joblib": dict(
        name="Logistic Regression", score_func="predict_proba",
        build=lambda: LogisticRegression(class_weight="balanced", max_iter=2000,
                                         random_state=RANDOM_STATE)),
    "nb_model.joblib": dict(
        name="Bernoulli Naive Bayes", score_func="predict_proba",
        build=lambda: BernoulliNB(alpha=1.0, binarize=0.5)),
    "rf_model.joblib": dict(
        name="Random Forest", score_func="predict_proba",
        build=lambda: RandomForestClassifier(n_estimators=100, max_depth=20,
                                             min_samples_leaf=1,
                                             class_weight="balanced",
                                             random_state=RANDOM_STATE, n_jobs=-1)),
}


def load_data():
    df = pd.read_csv(DATA, header=None, skipinitialspace=True, na_values="?")
    df = df.drop_duplicates().reset_index(drop=True)
    X = df.iloc[:, :-1].copy()
    X.columns = CONTINUOUS + [f"binary_{i}" for i in range(1, X.shape[1] - 2)]
    y = (df.iloc[:, -1].astype("string").str.strip().str.rstrip(".")
         .map({"nonad": 0, "ad": 1}).astype("int8"))
    X_fit, X_test, y_fit, y_test = train_test_split(X, y, test_size=0.20,
        random_state=RANDOM_STATE, stratify=y)
    X_tr, X_val, y_tr, y_val = train_test_split(X_fit, y_fit, test_size=0.25,
        random_state=RANDOM_STATE, stratify=y_fit)
    return X_tr, X_val, y_tr, y_val


def preprocessor(X):
    binary = [c for c in X.columns if c not in CONTINUOUS]
    return ColumnTransformer([
        ("continuous", Pipeline([("imputer", SimpleImputer(strategy="mean")),
                                  ("scaler", StandardScaler())]), CONTINUOUS),
        ("binary", SimpleImputer(strategy="most_frequent"), binary),
    ], remainder="drop", verbose_feature_names_out=False)


def select_threshold(scores, y_val):
    nonad = int((y_val == 0).sum())
    candidates = np.unique(np.concatenate(([0.0], scores)))
    fpr_eligible = []
    for t in candidates:
        pred = (scores >= t).astype(int)
        fpr = ((pred == 1) & (y_val == 0)).sum() / nonad
        if fpr <= MAX_FPR:
            fpr_eligible.append((t, fpr,
                                 precision_score(y_val, pred, zero_division=0),
                                 recall_score(y_val, pred, zero_division=0)))
    if not fpr_eligible:
        return None
    eligible = [r for r in fpr_eligible if r[3] >= MIN_RECALL]
    pool = eligible if eligible else fpr_eligible
    return max(pool, key=lambda r: (r[2], r[3], r[0]))


def main():
    print(f"Data: {DATA}")
    X_tr, X_val, y_tr, y_val = load_data()
    pre = preprocessor(X_tr)
    ART.mkdir(parents=True, exist_ok=True)

    for fname, spec in MODELS.items():
        out = ART / fname
        if out.exists():
            print(f"  [skip] {fname} already exists (not replaced)")
            continue
        pipe = Pipeline([("preprocessor", pre), ("classifier", spec["build"]())])
        pipe.fit(X_tr, y_tr)
        scores = pipe.predict_proba(X_val)[:, list(pipe.classes_).index(1)]
        chosen = select_threshold(scores, y_val.to_numpy())
        if chosen is None:
            print(f"  [warn] {fname}: no threshold met FPR<=5% — saving threshold 0.5")
            thr = 0.5
        else:
            thr = float(chosen[0])
        joblib.dump({"name": spec["name"], "pipeline": pipe, "threshold": thr,
                     "score_func": spec["score_func"]}, out)
        print(f"  [ok]   {fname}  threshold={thr:.4f}  -> {out}")

    print("\nDone. Restart the backend to load the new models.")


if __name__ == "__main__":
    main()
