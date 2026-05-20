"""scikit-learn model wrapper for PhishGuard.

The model is a small ``RandomForestClassifier`` trained on the canonical
feature vector produced by :mod:`phishguard.features`. We deliberately wrap
sklearn behind a thin class so:

* the rest of the codebase has a stable import surface, and
* it is easy to swap the estimator (gradient boosting, logistic regression,
  etc.) without touching the CLI or the heuristics.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np

from .features import FEATURE_NAMES, extract_features, features_to_vector


# Default location the CLI looks at if the user does not pass ``--model``.
DEFAULT_MODEL_PATH = Path(os.environ.get(
    "PHISHGUARD_MODEL",
    str(Path.home() / ".phishguard" / "model.joblib"),
))


@dataclass
class TrainReport:
    accuracy: float
    precision: float
    recall: float
    f1: float
    n_train: int
    n_test: int
    feature_importances: List[Tuple[str, float]]

    def pretty(self) -> str:
        lines = [
            f"Trained on {self.n_train} samples, evaluated on {self.n_test}.",
            f"  accuracy : {self.accuracy:.3f}",
            f"  precision: {self.precision:.3f}",
            f"  recall   : {self.recall:.3f}",
            f"  f1       : {self.f1:.3f}",
            "Top features:",
        ]
        for name, imp in self.feature_importances[:10]:
            lines.append(f"  {imp:6.3f}  {name}")
        return "\n".join(lines)


class PhishingModel:
    """Thin wrapper around a scikit-learn classifier.

    Use :meth:`train` to fit a fresh model, :meth:`save` / :meth:`load` to
    persist it, and :meth:`predict_url` to score a single URL.
    """

    def __init__(self, estimator=None):
        self._estimator = estimator
        # Stored alongside the estimator so we can verify at load time that
        # the feature layout matches what the model was trained on.
        self.feature_names: List[str] = list(FEATURE_NAMES)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @classmethod
    def train(
        cls,
        urls: Sequence[str],
        labels: Sequence[int],
        *,
        test_size: float = 0.2,
        random_state: int = 42,
        n_estimators: int = 200,
    ) -> Tuple["PhishingModel", TrainReport]:
        """Fit a new model on ``(urls, labels)``.

        ``labels`` must be 0 (benign) or 1 (phishing).
        """
        # Imports are local to keep import cost down for the CLI's common
        # path (scoring with heuristics only).
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
        )
        from sklearn.model_selection import train_test_split

        if len(urls) != len(labels):
            raise ValueError("urls and labels must have the same length")
        if len(urls) < 4:
            raise ValueError(
                "Need at least 4 samples to train (got %d)" % len(urls)
            )

        X = np.array(
            [features_to_vector(extract_features(u)) for u in urls],
            dtype=np.float64,
        )
        y = np.asarray(labels, dtype=np.int64)

        # Stratify when both classes are present and large enough.
        stratify = y if len(np.unique(y)) > 1 and min(np.bincount(y)) >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )

        estimator = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
        )
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)

        report = TrainReport(
            accuracy=float(accuracy_score(y_test, y_pred)),
            precision=float(precision_score(y_test, y_pred, zero_division=0)),
            recall=float(recall_score(y_test, y_pred, zero_division=0)),
            f1=float(f1_score(y_test, y_pred, zero_division=0)),
            n_train=int(len(X_train)),
            n_test=int(len(X_test)),
            feature_importances=sorted(
                zip(FEATURE_NAMES, estimator.feature_importances_.tolist()),
                key=lambda x: -x[1],
            ),
        )

        model = cls(estimator)
        return model, report

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: os.PathLike) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "estimator": self._estimator,
            "feature_names": self.feature_names,
            "version": 1,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: os.PathLike) -> "PhishingModel":
        payload = joblib.load(path)
        if not isinstance(payload, dict) or "estimator" not in payload:
            raise ValueError(f"{path} is not a PhishGuard model file")
        if payload.get("feature_names") != FEATURE_NAMES:
            raise ValueError(
                "Model was trained with a different feature layout than the "
                "current version of phishguard. Re-train the model."
            )
        instance = cls(payload["estimator"])
        return instance

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_url(self, url: str) -> float:
        """Return P(phishing) in [0, 1] for a single URL."""
        if self._estimator is None:
            raise RuntimeError("Model is not trained or loaded")
        vec = np.array(
            [features_to_vector(extract_features(url))],
            dtype=np.float64,
        )
        # predict_proba returns [[p_benign, p_phish]] when both classes are
        # present. If the classifier was trained on a single class (degenerate
        # tiny dataset), fall back to predict().
        if hasattr(self._estimator, "predict_proba") and len(getattr(self._estimator, "classes_", [])) == 2:
            proba = self._estimator.predict_proba(vec)[0]
            classes = list(self._estimator.classes_)
            return float(proba[classes.index(1)])
        return float(self._estimator.predict(vec)[0])

    def predict_urls(self, urls: Iterable[str]) -> List[float]:
        return [self.predict_url(u) for u in urls]


def try_load_default_model() -> Optional[PhishingModel]:
    """Best-effort loader used by the CLI: returns ``None`` if there is no
    saved model rather than raising."""
    if DEFAULT_MODEL_PATH.exists():
        try:
            return PhishingModel.load(DEFAULT_MODEL_PATH)
        except Exception:
            return None
    return None


__all__ = [
    "PhishingModel",
    "TrainReport",
    "DEFAULT_MODEL_PATH",
    "try_load_default_model",
]
