"""Stock-level classifier on top of (VLM probs, embeddings, geometry).

Trained on the team's hand labels (audit set) + high-agreement VLM
pseudo-labels. Outputs calibrated empty/half/full probabilities that feed
straight into the EV decision rule.

sklearn is optional at import time; train/predict require it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

LEVELS = ("empty", "half", "full")

FEATURE_NAMES = (
    "p_empty",
    "p_half",
    "p_full",
    "coverage_percent",
    "visible_fraction",
    "dist_empty_ref",
    "dist_full_ref",
)


def build_feature_vector(
    probs: dict[str, float] | None,
    coverage_percent: float | None,
    visible_fraction: float,
    dist_empty_ref: float | None = None,
    dist_full_ref: float | None = None,
) -> list[float]:
    p = probs or {"empty": 1 / 3, "half": 1 / 3, "full": 1 / 3}
    return [
        float(p.get("empty", 0.0)),
        float(p.get("half", 0.0)),
        float(p.get("full", 0.0)),
        float(coverage_percent if coverage_percent is not None else -1.0),
        float(visible_fraction),
        float(dist_empty_ref if dist_empty_ref is not None else -1.0),
        float(dist_full_ref if dist_full_ref is not None else -1.0),
    ]


def train_classifier(
    features: np.ndarray,
    labels: list[str],
    calibrate: bool = False,
):
    """Train a multinomial logistic regression (optionally isotonic-calibrated)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y = np.array([LEVELS.index(l) for l in labels])
    model = LogisticRegression(max_iter=2000, C=1.0)
    if calibrate:
        from sklearn.calibration import CalibratedClassifierCV

        model = CalibratedClassifierCV(model, cv=3, method="isotonic")
    else:
        model = make_pipeline(StandardScaler(), model)
    model.fit(features, y)
    return model


def predict_probs(model, feature_vector: list[float]) -> dict[str, float]:
    x = np.asarray(feature_vector, dtype=float).reshape(1, -1)
    raw = model.predict_proba(x)[0]
    # ensure alignment with LEVELS when classes are sparse
    classes = list(model.classes_)
    probs = {"empty": 0.0, "half": 0.0, "full": 0.0}
    for cls, p in zip(classes, raw):
        probs[LEVELS[int(cls)]] = float(p)
    total = sum(probs.values())
    if total <= 0:
        return {"empty": 1 / 3, "half": 1 / 3, "full": 1 / 3}
    return {k: v / total for k, v in probs.items()}


def save_model(model, path: str | Path) -> None:
    import joblib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str | Path):
    import joblib

    return joblib.load(path)


def dataset_from_audit(
    labels_path: str | Path,
    feature_store_path: str | Path,
) -> tuple[np.ndarray, list[str]]:
    """Load (features, labels) for training.

    The feature store is a JSON file mapping
    "<camera>/<image>/<area>" -> feature vector, produced during a pipeline
    run (see scripts/dump_features.py once data is available).
    """
    labels = json.loads(Path(labels_path).read_text(encoding="utf-8"))
    store = json.loads(Path(feature_store_path).read_text(encoding="utf-8"))

    X: list[list[float]] = []
    y: list[str] = []
    for cam, images in labels.items():
        for img, areas in images.items():
            for area_id, truth in areas.items():
                key = f"{cam}/{img}/{area_id}"
                if key in store and truth.get("stock") in LEVELS:
                    X.append(store[key])
                    y.append(truth["stock"])
    if not X:
        raise ValueError("No overlapping audit labels / features to train on")
    return np.asarray(X, dtype=float), y
