"""Rule-based heuristic phishing scorer.

Each rule fires independently and contributes a small weight to a running
total. The total is squashed into [0, 1] with a logistic so that the score
is interpretable as a "probability-like" number even though it is not a
true probability.

The point of the heuristics is twofold:

1. Provide a useful score even before any model is trained.
2. Generate human-readable explanations that the CLI can show alongside
   the model's verdict.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple

from .features import (
    IMPERSONATED_BRANDS,
    PHISHING_KEYWORDS,
    SUSPICIOUS_TLDS,
    URL_SHORTENERS,
    parse,
)


# Each rule is (weight, description). Higher weight = stronger phishing signal.
# Weights are tuned by intuition / common literature, not by training.
# Negative weights reduce suspicion.
_RULES_DOC = """\
Rule weights (sum is squashed via logistic into [0,1]):
  +2.0  raw IP address as host
  +1.5  '@' character in the authority (URL spoofing trick)
  +1.5  punycode / IDN host (xn--)
  +1.2  brand impersonation token outside registered domain
  +1.0  hostname is on a URL-shortener allowlist
  +1.0  suspicious / abused TLD
  +0.8  hostname entropy > 4.0 (very random-looking)
  +0.7  more than 3 subdomains
  +0.6  hex-encoded characters in the URL
  +0.6  uses non-standard port
  +0.5  3+ phishing keywords in URL
  +0.4  URL longer than 100 chars
  +0.4  more than 5 hyphens in URL
  +0.3  double-slash in path (after scheme)
  -0.6  uses HTTPS
"""


@dataclass
class HeuristicResult:
    score: float
    raw: float  # pre-logistic sum, useful for debugging
    signals: List[str] = field(default_factory=list)


def _logistic(x: float) -> float:
    """Standard logistic, clamped to avoid overflow."""
    if x > 50:
        return 1.0
    if x < -50:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def score_heuristics(url: str) -> HeuristicResult:
    """Score ``url`` using hand-written rules.

    Returns a :class:`HeuristicResult` containing a score in [0, 1], the raw
    pre-logistic sum, and a list of human-readable signal strings explaining
    which rules fired.
    """
    from .features import extract_features  # local import to avoid cycles

    feats = extract_features(url)
    p = parse(url)

    weighted: List[Tuple[float, str]] = []

    if feats["has_ip_host"]:
        weighted.append((2.0, "host is a raw IP address"))

    if feats["has_at_in_authority"]:
        weighted.append((1.5, "'@' in URL authority (classic spoofing trick)"))

    if feats["has_punycode"]:
        weighted.append((1.5, "punycode / IDN host (possible homograph attack)"))

    if feats["num_brand_impersonations"] >= 1:
        # Identify which brand(s) appear, for the explanation. Mirror the
        # zones used by extract_features so the message matches the count.
        sub = p.subdomain
        path = (p.parsed.path or "").lower()
        query = (p.parsed.query or "").lower()
        root = p.domain_root
        hits = []
        for b in IMPERSONATED_BRANDS:
            if b in sub or b in path or b in query or (root and b in root and root != b):
                hits.append(b)
            if len(hits) >= 3:
                break
        weighted.append((
            1.2,
            "brand impersonation token outside registered domain: "
            + ", ".join(hits),
        ))

    if feats["is_url_shortener"]:
        weighted.append((
            1.0,
            f"URL shortener domain ({p.registered_domain}); destination hidden",
        ))

    if feats["has_suspicious_tld"]:
        # Show the actual TLD so the user can see why.
        tld = p.suffix.split(".")[-1] if p.suffix else ""
        weighted.append((1.0, f"suspicious / abused TLD (.{tld})"))

    if feats["hostname_entropy"] > 4.0:
        weighted.append((
            0.8,
            f"high hostname entropy ({feats['hostname_entropy']:.2f}), "
            "looks random",
        ))

    if feats["num_subdomains"] > 3:
        weighted.append((
            0.7,
            f"unusually many subdomains ({int(feats['num_subdomains'])})",
        ))

    if feats["num_hex_encoded"] >= 1:
        weighted.append((
            0.6,
            f"hex-encoded characters in URL ({int(feats['num_hex_encoded'])})",
        ))

    if feats["has_non_standard_port"]:
        weighted.append((0.6, f"non-standard port: {p.port}"))

    if feats["num_phishing_keywords"] >= 3:
        # List up to three matched keywords for context.
        lowered = url.lower()
        hits = [kw for kw in PHISHING_KEYWORDS if kw in lowered][:3]
        weighted.append((
            0.5,
            "multiple phishing keywords in URL: " + ", ".join(hits),
        ))
    elif feats["num_phishing_keywords"] >= 1:
        lowered = url.lower()
        hits = [kw for kw in PHISHING_KEYWORDS if kw in lowered][:2]
        weighted.append((
            0.25,
            "phishing keyword in URL: " + ", ".join(hits),
        ))

    if feats["url_length"] > 100:
        weighted.append((
            0.4,
            f"unusually long URL ({int(feats['url_length'])} chars)",
        ))

    if feats["num_hyphens"] > 5:
        weighted.append((
            0.4,
            f"many hyphens in URL ({int(feats['num_hyphens'])})",
        ))

    if feats["has_double_slash_in_path"]:
        weighted.append((0.3, "double-slash in path (after scheme)"))

    if feats["uses_https"]:
        weighted.append((-0.6, "uses HTTPS"))
    else:
        weighted.append((0.2, "does not use HTTPS"))

    raw_sum = sum(w for w, _ in weighted)

    # Center the logistic around ~1.5 so a single mild signal does not
    # immediately push the score above 0.5. Empirically this gives better
    # behavior on benign inputs.
    score = _logistic(raw_sum - 1.5)

    # Drop the suppressing HTTPS / "no HTTPS" lines from the displayed signals
    # unless they were the only thing — they are noise on most reports.
    signals = [
        desc for w, desc in weighted
        if not desc.startswith("uses HTTPS") and not desc.startswith("does not use HTTPS")
    ]
    if not signals and weighted:
        # Keep at least one signal so the user sees something.
        signals = [weighted[0][1]]

    # Sort by weight descending for "most important first".
    weighted_sorted = sorted(
        (item for item in weighted if not item[1].startswith("uses HTTPS")
         and not item[1].startswith("does not use HTTPS")),
        key=lambda x: -x[0],
    )
    signals = [desc for _, desc in weighted_sorted]

    return HeuristicResult(score=score, raw=raw_sum, signals=signals)


__all__ = ["HeuristicResult", "score_heuristics"]
