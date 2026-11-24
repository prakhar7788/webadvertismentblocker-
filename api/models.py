"""Load the trained models from the artifacts directory and run predictions.

Expected saved-model file format (a pickled dict):

    {
        "pipeline": <sklearn Pipeline ending in a classifier>,
        "threshold": <float>,                      # decision threshold
        "name": "<human-readable name>",            # optional
        "score_func": "predict_proba" | "decision_function",  # optional
    }

The existing `svm_model.joblib` only has "pipeline" and "threshold"; for it we
infer the scoring function from the classifier type (SVC -> decision_function).
For every other model we default to predict_proba (correct for Logistic
Regression, Bernoulli NB and Random Forest).
"""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
from sklearn.svm import SVC

from .features import url_to_features

log = logging.getLogger("adblocker.models")

# Filename -> human-readable label. These are the files the backend looks for.
EXPECTED_MODELS = {
    "svm_model.joblib": "SVM (RBF)",
    "logreg_model.joblib": "Logistic Regression",
    "nb_model.joblib": "Bernoulli Naive Bayes",
    "rf_model.joblib": "Random Forest",
}


class ModelBundle:
    """A loaded trained model plus its threshold and scoring function."""

    def __init__(self, name: str, pipeline, threshold: float, score_func: str):
        self.name = name
        self.pipeline = pipeline
        self.threshold = float(threshold)
        self.score_func = score_func

    def predict(self, url: str) -> dict:
        X = url_to_features(url)
        clf = self.pipeline.named_steps["classifier"]

        if self.score_func == "decision_function":
            score = float(self.pipeline.decision_function(X)[0])
        else:
            proba = self.pipeline.predict_proba(X)
            ad_index = list(clf.classes_).index(1)
            score = float(proba[0, ad_index])

        is_ad = bool(score >= self.threshold)
        return {
            "name": self.name,
            "label": "ad" if is_ad else "nonad",
            "ad": is_ad,
            "score": round(score, 6),
            "threshold": round(self.threshold, 6),
            "score_func": self.score_func,
            "source": "trained",
        }


def _infer_score_func(pipeline) -> str:
    clf = pipeline.named_steps["classifier"]
    if isinstance(clf, SVC):
        return "decision_function"
    return "predict_proba"


def load_models(artifacts_dir: Path):
    """Load every expected model file that exists.

    Returns (bundles, missing) where `missing` is a list of human-readable names
    (with a reason) for models that are absent or failed to load.
    """
    bundles: list[ModelBundle] = []
    missing: list[str] = []

    for filename, label in EXPECTED_MODELS.items():
        path = Path(artifacts_dir) / filename
        if not path.exists():
            missing.append(label)
            continue
        try:
            blob = joblib.load(path)
            pipeline = blob["pipeline"]
            threshold = blob["threshold"]
            name = blob.get("name", label)
            score_func = blob.get("score_func") or _infer_score_func(pipeline)
            bundles.append(ModelBundle(name, pipeline, threshold, score_func))
            log.info("loaded model %s from %s (score_func=%s)", name, filename, score_func)
        except Exception as exc:  # noqa: BLE001 - report any load failure clearly
            missing.append(f"{label} (load failed: {exc})")
            log.warning("could not load %s: %s", filename, exc)

    return bundles, missing
