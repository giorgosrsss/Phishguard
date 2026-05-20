"""Tests for phishguard.heuristics and the high-level classify() blender."""

from __future__ import annotations

import pytest

from phishguard.classifier import classify
from phishguard.heuristics import score_heuristics


# ---------------------------------------------------------------------------
# Heuristic-only tests (no model in the loop)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "https://www.google.com/",
        "https://github.com/torvalds/linux",
        "https://docs.python.org/3/library/urllib.parse.html",
        "https://aws.amazon.com/console/",
    ],
)
def test_benign_urls_score_low(url):
    res = score_heuristics(url)
    assert res.score < 0.5, f"benign URL scored {res.score:.2f}: {url}"


@pytest.mark.parametrize(
    "url",
    [
        "http://192.168.1.4/login.php",
        "http://paypal.com.security-update-center.tk/login",
        "http://account-locked-apple.support/unlock",
        "http://verify-paypal-account.gq/cgi-bin/webscr?cmd=_login-run",
        "http://xn--pypal-4ve.com/signin",
        "http://evil.com@phishing.tk/",
    ],
)
def test_phishing_urls_score_high(url):
    res = score_heuristics(url)
    assert res.score > 0.5, f"phishing URL scored {res.score:.2f}: {url}"


def test_score_is_in_unit_interval():
    for url in [
        "",
        "not-a-url",
        "https://example.com/",
        "http://192.168.1.1/admin/login.php?next=/dashboard",
    ]:
        res = score_heuristics(url)
        assert 0.0 <= res.score <= 1.0


def test_signals_are_returned_for_phishing_url():
    res = score_heuristics("http://paypal.com.security-update.tk/login/verify")
    assert res.signals, "expected at least one human-readable signal"
    # Should mention either the suspicious TLD or brand impersonation.
    joined = " | ".join(res.signals).lower()
    assert "tld" in joined or "brand" in joined


# ---------------------------------------------------------------------------
# classify() blender (no model loaded -> heuristic only)
# ---------------------------------------------------------------------------


def test_classify_returns_result_without_model(tmp_path, monkeypatch):
    # Point the default model at a non-existent path so we are guaranteed to
    # exercise the heuristic-only branch.
    monkeypatch.setenv("PHISHGUARD_MODEL", str(tmp_path / "missing.joblib"))
    # Re-import to pick up the env var change (model module caches DEFAULT path).
    import importlib

    import phishguard.model as model_module
    importlib.reload(model_module)
    import phishguard.classifier as classifier_module
    importlib.reload(classifier_module)

    res = classifier_module.classify("https://www.google.com/")
    assert res.model_score is None
    assert 0.0 <= res.score <= 1.0
    assert res.verdict in {"PHISHING", "SUSPICIOUS", "LIKELY BENIGN", "BENIGN"}


def test_classify_pretty_output_contains_url():
    res = classify("https://www.google.com/")
    text = res.pretty()
    assert "https://www.google.com/" in text
    assert "Heuristic score" in text
    assert "Blended verdict" in text


def test_phishing_url_gets_non_benign_verdict():
    res = classify("http://paypal.com.security-update-center.tk/login/verify/account")
    assert res.verdict in {"PHISHING", "SUSPICIOUS"}
