"""PhishGuard — hybrid phishing URL classifier.

Public API:

    from phishguard import classify, Result, extract_features

    result = classify("http://example.com/login")
    print(result.score, result.verdict, result.signals)
"""

from .features import extract_features, FEATURE_NAMES
from .heuristics import score_heuristics, HeuristicResult
from .classifier import classify, Result

__all__ = [
    "classify",
    "Result",
    "extract_features",
    "FEATURE_NAMES",
    "score_heuristics",
    "HeuristicResult",
]

__version__ = "0.1.0"
