"""Tests for phishguard.features."""

from __future__ import annotations

import math

import pytest

from phishguard.features import (
    FEATURE_NAMES,
    extract_features,
    features_to_vector,
    parse,
)


def test_feature_dict_keys_match_canonical_names():
    feats = extract_features("https://example.com/")
    assert set(feats.keys()) == set(FEATURE_NAMES)


def test_features_to_vector_is_ordered_and_float():
    feats = extract_features("https://example.com/")
    vec = features_to_vector(feats)
    assert len(vec) == len(FEATURE_NAMES)
    assert all(isinstance(v, float) for v in vec)


@pytest.mark.parametrize(
    "url,expected_https",
    [
        ("https://example.com/", 1.0),
        ("http://example.com/", 0.0),
        # Scheme-less input is normalized to https:// to match browser behavior
        # and avoid biasing the model against bare hostnames.
        ("example.com/path", 1.0),
        ("paypal.com", 1.0),
    ],
)
def test_uses_https_flag(url, expected_https):
    feats = extract_features(url)
    assert feats["uses_https"] == expected_https


def test_ip_host_detected():
    feats = extract_features("http://192.168.1.1/login")
    assert feats["has_ip_host"] == 1.0


def test_ipv6_host_detected():
    feats = extract_features("http://[2001:db8::1]/wp-login.php")
    assert feats["has_ip_host"] == 1.0


def test_dns_host_not_flagged_as_ip():
    feats = extract_features("https://www.example.com/")
    assert feats["has_ip_host"] == 0.0


def test_punycode_flag():
    feats = extract_features("http://xn--pypal-4ve.com/signin")
    assert feats["has_punycode"] == 1.0


def test_suspicious_tld_flag():
    feats = extract_features("http://account-verify.tk/login")
    assert feats["has_suspicious_tld"] == 1.0


def test_normal_tld_not_flagged_as_suspicious():
    feats = extract_features("https://www.google.com/")
    assert feats["has_suspicious_tld"] == 0.0


def test_url_shortener_flag():
    feats = extract_features("http://bit.ly/3xXxXxX")
    assert feats["is_url_shortener"] == 1.0


def test_at_in_authority_flag():
    feats = extract_features("http://evil.com@phishing.tk/")
    assert feats["has_at_in_authority"] == 1.0


def test_phishing_keywords_counted():
    feats = extract_features("http://example.com/login/verify/account")
    assert feats["num_phishing_keywords"] >= 3


def test_brand_impersonation_outside_registered_domain():
    # 'paypal' appears in the path / subdomain but not in the registered
    # domain (which is .tk) -> impersonation signal should fire.
    feats = extract_features("http://paypal-secure.update.tk/login")
    assert feats["num_brand_impersonations"] >= 1


def test_brand_in_real_domain_does_not_count():
    feats = extract_features("https://www.paypal.com/us/home")
    # paypal IS the registered domain here, so it is excluded from the count.
    assert feats["num_brand_impersonations"] == 0.0


def test_hex_encoded_characters_counted():
    feats = extract_features("http://example.com/%70assword%2Ereset")
    assert feats["num_hex_encoded"] == 2.0


def test_non_standard_port_flag():
    feats = extract_features("http://example.com:8443/path")
    assert feats["has_non_standard_port"] == 1.0


def test_default_port_not_flagged():
    feats = extract_features("https://example.com:443/")
    assert feats["has_non_standard_port"] == 0.0


def test_entropy_is_nonnegative_and_finite():
    feats = extract_features("https://example.com/")
    assert feats["url_entropy"] >= 0
    assert feats["hostname_entropy"] >= 0
    assert math.isfinite(feats["url_entropy"])
    assert math.isfinite(feats["hostname_entropy"])


def test_parse_handles_garbage_without_raising():
    # Should not raise on weird input.
    p = parse("not a url at all")
    assert p.hostname is not None  # may be empty but defined


def test_parse_handles_empty_string():
    p = parse("")
    assert p.hostname == ""


def test_extract_features_handles_empty_string():
    feats = extract_features("")
    assert feats["url_length"] == 0.0
