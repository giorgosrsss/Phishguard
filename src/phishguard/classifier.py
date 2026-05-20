"""High-level ``classify()`` that blends heuristics with the ML model.

This is the function most library users will actually call. It hides the
distinction between "model available" and "no model trained yet": if a model
exists, it is used; if not, the heuristic score is returned alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .heuristics import score_heuristics
from .model import PhishingModel, try_load_default_model


# Verdict thresholds applied to the blended score.
VERDICT_THRESHOLDS = (
    (0.80, "PHISHING"),
    (0.55, "SUSPICIOUS"),
    (0.30, "LIKELY BENIGN"),
)


@dataclass
class Result:
    url: str
    score: float                 # blended score in [0, 1]
    verdict: str                 # "PHISHING" | "SUSPICIOUS" | "LIKELY BENIGN" | "BENIGN"
    heuristic_score: float
    model_score: Optional[float] = None
    signals: List[str] = field(default_factory=list)

    def pretty(self) -> str:
        """Return a multi-line human-readable summary."""
        def band(score: float) -> str:
            if score >= 0.80:
                return "HIGH"
            if score >= 0.55:
                return "MEDIUM"
            if score >= 0.30:
                return "LOW"
            return "VERY LOW"

        lines = [
            f"URL: {self.url}",
            f"Heuristic score : {self.heuristic_score:.2f}  ({band(self.heuristic_score)})",
        ]
        if self.model_score is None:
            lines.append("Model score     : (no trained model — run `phishguard train`)")
        else:
            lines.append(
                f"Model score     : {self.model_score:.2f}  ({band(self.model_score)})"
            )
        lines.append(f"Blended verdict : {self.score:.2f}  {self.verdict}")
        if self.signals:
            lines.append("")
            lines.append("Top signals:")
            for s in self.signals[:8]:
                lines.append(f"  + {s}")
        return "\n".join(lines)


def _verdict_for(score: float) -> str:
    for threshold, label in VERDICT_THRESHOLDS:
        if score >= threshold:
            return label
    return "BENIGN"


def classify(url: str, model: Optional[PhishingModel] = None) -> Result:
    """Score a URL using heuristics plus the ML model when available.

    Parameters
    ----------
    url:
        The URL to score. Schemes are inferred if missing.
    model:
        Optional pre-loaded :class:`PhishingModel`. If ``None``, tries to load
        the default model from ``~/.phishguard/model.joblib``; if no model is
        available, falls back to heuristics only.
    """
    heur = score_heuristics(url)

    if model is None:
        model = try_load_default_model()

    model_score: Optional[float] = None
    if model is not None:
        try:
            model_score = float(model.predict_url(url))
        except Exception:
            # Model failure should not break the heuristic path.
            model_score = None

    if model_score is None:
        blended = heur.score
    else:
        # Weighted average: model gets 60% weight when available because it
        # has seen labelled data; heuristics get 40% as a stable prior.
        blended = 0.4 * heur.score + 0.6 * model_score

    return Result(
        url=url,
        score=blended,
        verdict=_verdict_for(blended),
        heuristic_score=heur.score,
        model_score=model_score,
        signals=heur.signals,
    )


__all__ = ["Result", "classify", "VERDICT_THRESHOLDS"]
