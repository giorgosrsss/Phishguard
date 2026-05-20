"""URL feature extraction.

Given a URL string, ``extract_features`` returns a dictionary of numeric
features suitable both for the heuristic scorer and as input to the ML model.

The feature set is deliberately *lexical*: nothing here makes a network
request. That keeps the classifier fast, deterministic, and safe to run on
untrusted input.
"""

from __future__ import annotations

import ipaddress
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional
from urllib.parse import urlparse, ParseResult

import tldextract

# tldextract by default fetches the public suffix list from the network on
# first use. We disable that to keep the tool offline-safe; it falls back to
# the snapshot bundled with the library.
_TLD_EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=())


# ---------------------------------------------------------------------------
# Wordlists
# ---------------------------------------------------------------------------

# TLDs frequently abused in phishing campaigns. Not exhaustive, and inclusion
# does NOT mean a TLD is malicious — it means it has a high abuse rate
# historically and is therefore a useful signal.
SUSPICIOUS_TLDS = frozenset({
    "zip", "mov", "tk", "ml", "ga", "cf", "gq", "xyz", "top", "club",
    "click", "country", "stream", "download", "loan", "review", "men",
    "work", "date", "racing", "win", "bid", "trade", "party", "cricket",
    "science", "faith", "accountant", "kim", "biz",
})

# Words commonly seen in phishing URL paths/queries. Used both as a feature
# (count) and as an explanation in heuristics.
PHISHING_KEYWORDS = (
    "login", "log-in", "signin", "sign-in", "verify", "verification",
    "secure", "security", "account", "update", "confirm", "password",
    "credential", "wallet", "bank", "billing", "invoice", "payment",
    "support", "unlock", "suspend", "recovery", "auth", "session", "token",
    "webscr", "ebayisapi",
)

# Brands frequently impersonated in phishing. If one of these appears in the
# URL but the registered domain is NOT the real brand, that is a strong
# signal of impersonation.
IMPERSONATED_BRANDS = (
    "paypal", "apple", "icloud", "microsoft", "office365", "outlook",
    "google", "gmail", "youtube", "amazon", "netflix", "facebook",
    "instagram", "whatsapp", "linkedin", "twitter", "github", "dropbox",
    "wellsfargo", "chase", "bankofamerica", "citibank", "hsbc", "barclays",
    "santander", "coinbase", "binance", "metamask", "blockchain",
    "steam", "discord", "spotify", "adobe", "docusign",
)

# Well-known URL shorteners. Real shorteners hide the destination, which is a
# common phishing tactic.
URL_SHORTENERS = frozenset({
    "bit.ly", "t.co", "tinyurl.com", "goo.gl", "ow.ly", "is.gd", "buff.ly",
    "adf.ly", "bit.do", "cutt.ly", "shorte.st", "rebrand.ly", "rb.gy",
    "tiny.cc", "lnkd.in", "trib.al", "soo.gd", "clck.ru", "shorturl.at",
})


# Order matters — this list is the canonical feature vector layout the ML
# model is trained on. Keep it stable across versions.
FEATURE_NAMES: List[str] = [
    "url_length",
    "hostname_length",
    "path_length",
    "query_length",
    "num_dots",
    "num_hyphens",
    "num_slashes",
    "num_question_marks",
    "num_equals",
    "num_ampersands",
    "num_at",
    "num_digits",
    "num_special_chars",
    "digit_ratio",
    "letter_ratio",
    "special_char_ratio",
    "hostname_entropy",
    "url_entropy",
    "num_subdomains",
    "longest_token_length",
    "has_ip_host",
    "has_punycode",
    "has_non_standard_port",
    "has_suspicious_tld",
    "is_url_shortener",
    "uses_https",
    "has_at_in_authority",
    "has_double_slash_in_path",
    "num_phishing_keywords",
    "num_brand_impersonations",
    "num_hex_encoded",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HEX_PATTERN = re.compile(r"%[0-9a-fA-F]{2}")
_TOKEN_SPLIT = re.compile(r"[^A-Za-z0-9]+")


def _shannon_entropy(s: str) -> float:
    """Shannon entropy in bits of the characters in ``s``."""
    if not s:
        return 0.0
    counts = Counter(s)
    length = len(s)
    return -sum((c / length) * math.log2(c / length) for c in counts.values())


def _is_ip_address(host: str) -> bool:
    """Return True if ``host`` is a literal IPv4 or IPv6 address."""
    if not host:
        return False
    # IPv6 literals in URLs come wrapped in brackets, e.g. [::1].
    candidate = host.strip("[]")
    try:
        ipaddress.ip_address(candidate)
        return True
    except ValueError:
        return False


def _normalize_url(raw: str) -> str:
    """Ensure the URL has a scheme so ``urlparse`` populates ``netloc``.

    A bare hostname like ``paypal.com`` is treated as if the user typed it
    into a browser address bar — i.e. defaulted to ``https://``. This avoids
    biasing the ``uses_https`` feature against scheme-less inputs.
    """
    raw = raw.strip()
    if not raw:
        return raw
    if "://" not in raw:
        return "https://" + raw
    return raw


def _safe_parse(url: str) -> ParseResult:
    return urlparse(_normalize_url(url))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParsedUrl:
    """Lightweight bundle of parsed URL parts used by both the feature
    extractor and the heuristics module."""

    raw: str
    parsed: ParseResult
    hostname: str
    registered_domain: str
    domain_root: str  # the "label" before the public suffix, e.g. "google" in google.co.uk
    subdomain: str
    suffix: str
    port: Optional[int]


def parse(url: str) -> ParsedUrl:
    """Parse a URL into a structured form. Never raises; falls back to
    empty parts if the URL is malformed."""
    parsed = _safe_parse(url)
    hostname = (parsed.hostname or "").lower()

    extracted = _TLD_EXTRACTOR(hostname)
    registered = ".".join(p for p in (extracted.domain, extracted.suffix) if p)

    try:
        port = parsed.port
    except ValueError:
        # urlparse raises ValueError on garbage ports like "abc".
        port = None

    return ParsedUrl(
        raw=url,
        parsed=parsed,
        hostname=hostname,
        registered_domain=registered,
        domain_root=extracted.domain.lower(),
        subdomain=extracted.subdomain.lower(),
        suffix=extracted.suffix.lower(),
        port=port,
    )


def extract_features(url: str) -> Dict[str, float]:
    """Return the canonical numeric feature vector for ``url`` as a dict.

    Use ``FEATURE_NAMES`` to convert the dict to a stable ordered list:

        feats = extract_features(url)
        vector = [feats[name] for name in FEATURE_NAMES]
    """
    p = parse(url)
    parsed = p.parsed
    hostname = p.hostname
    path = parsed.path or ""
    query = parsed.query or ""

    # Counts on the raw URL string (after normalization).
    raw = _normalize_url(url)
    num_digits = sum(c.isdigit() for c in raw)
    num_letters = sum(c.isalpha() for c in raw)
    num_special = sum(not c.isalnum() for c in raw)
    total = max(len(raw), 1)

    # Tokens of the hostname (split on dots and hyphens) — useful to
    # detect long random-looking strings.
    tokens = [t for t in _TOKEN_SPLIT.split(hostname) if t]
    longest_token = max((len(t) for t in tokens), default=0)

    # Phishing keyword & brand impersonation counts. We look at the full URL
    # (lowercased) for keywords. For brand impersonation we want to detect
    # cases like "paypal.com.security-update.tk" or "account-locked-apple.support"
    # while NOT flagging the brand's actual site (e.g. paypal.com itself).
    lowered = raw.lower()
    num_phish_kw = sum(lowered.count(kw) for kw in PHISHING_KEYWORDS)

    # Search for brand tokens in subdomain, path, query — these zones never
    # contain the brand legitimately on the brand's own site. Also flag when
    # the brand is embedded in the domain root but the root is not exactly
    # the brand (e.g. "paypal-secure" contains "paypal" but isn't paypal.com).
    subdomain = p.subdomain
    path_lower = path.lower()
    query_lower = query.lower()
    domain_root = p.domain_root
    num_brand_imp = 0
    for brand in IMPERSONATED_BRANDS:
        in_zones = (
            brand in subdomain
            or brand in path_lower
            or brand in query_lower
        )
        in_root_but_not_root = (
            domain_root
            and brand in domain_root
            and domain_root != brand
        )
        if in_zones or in_root_but_not_root:
            num_brand_imp += 1

    has_non_standard_port = False
    if p.port is not None:
        scheme = (parsed.scheme or "http").lower()
        default = 443 if scheme == "https" else 80
        has_non_standard_port = p.port != default

    features: Dict[str, float] = {
        "url_length": float(len(raw)),
        "hostname_length": float(len(hostname)),
        "path_length": float(len(path)),
        "query_length": float(len(query)),
        "num_dots": float(raw.count(".")),
        "num_hyphens": float(raw.count("-")),
        "num_slashes": float(raw.count("/")),
        "num_question_marks": float(raw.count("?")),
        "num_equals": float(raw.count("=")),
        "num_ampersands": float(raw.count("&")),
        "num_at": float(raw.count("@")),
        "num_digits": float(num_digits),
        "num_special_chars": float(num_special),
        "digit_ratio": num_digits / total,
        "letter_ratio": num_letters / total,
        "special_char_ratio": num_special / total,
        "hostname_entropy": _shannon_entropy(hostname),
        "url_entropy": _shannon_entropy(raw),
        "num_subdomains": float(hostname.count(".")) if hostname else 0.0,
        "longest_token_length": float(longest_token),
        "has_ip_host": float(_is_ip_address(hostname)),
        "has_punycode": float("xn--" in hostname),
        "has_non_standard_port": float(has_non_standard_port),
        "has_suspicious_tld": float(p.suffix.split(".")[-1] in SUSPICIOUS_TLDS),
        "is_url_shortener": float(p.registered_domain in URL_SHORTENERS),
        "uses_https": float((parsed.scheme or "").lower() == "https"),
        "has_at_in_authority": float("@" in (parsed.netloc or "")),
        "has_double_slash_in_path": float("//" in path),
        "num_phishing_keywords": float(num_phish_kw),
        "num_brand_impersonations": float(num_brand_imp),
        "num_hex_encoded": float(len(_HEX_PATTERN.findall(raw))),
    }

    # Sanity check: every declared feature is populated.
    assert set(features.keys()) == set(FEATURE_NAMES), (
        "Feature dict does not match FEATURE_NAMES; update one or the other."
    )
    return features


def features_to_vector(features: Dict[str, float]) -> List[float]:
    """Convert a feature dict into the canonical ordered list."""
    return [float(features[name]) for name in FEATURE_NAMES]
