# PhishGuard — Phishing URL Classifier

A hybrid phishing-URL detector that combines hand-crafted heuristic rules with a
machine-learning classifier (scikit-learn `RandomForest`). Given any URL, it
returns a phishing-likelihood score between 0.0 (benign) and 1.0 (almost
certainly phishing), plus a human-readable explanation of the signals that
contributed to the decision.

## Why hybrid?

- **Heuristics** catch the obvious stuff instantly and require no training data.
  Things like raw-IP hosts, punycode, `@` in the authority, suspicious TLDs, and
  brand-keyword stuffing are easy wins.
- **ML model** learns the subtle statistical patterns heuristics miss (token
  distributions, entropy, character-class ratios, etc.) once you feed it
  labelled data.

The CLI returns the heuristic score, the model score, and a blended verdict.

## How it works

### Inference dataflow

When you call `classify(url)` (or `phishguard scan <url>`), the URL travels
through the pipeline below. The heuristic scorer and the ML model run on the
**same** feature vector and their scores are blended into a single verdict.

```mermaid
flowchart LR
    URL([URL string]) --> NORM[Normalize URL<br/>add scheme if missing]
    NORM --> PARSE[Parse with urllib + tldextract<br/>→ scheme, host, domain root,<br/>subdomain, suffix, path, query, port]
    PARSE --> FEAT[Extract 31 features<br/>lengths · counts · ratios · entropy<br/>IP / punycode / TLD / shortener flags<br/>phishing-keyword &amp; brand-impersonation counts]

    FEAT --> HEUR[Heuristic scorer<br/>15 weighted rules → logistic]
    FEAT --> ML[RandomForest model<br/>predict_proba]

    HEUR --> HSCORE[Heuristic score<br/>0.0 – 1.0]
    HEUR --> SIGNALS[Human-readable signals<br/>e.g. &quot;suspicious TLD .tk&quot;]
    ML --> MSCORE[Model score<br/>0.0 – 1.0]

    HSCORE --> BLEND[Blender<br/>0.4 × heuristic + 0.6 × model]
    MSCORE --> BLEND

    BLEND --> THRESH{Verdict thresholds}
    THRESH -->|≥ 0.80| V1[PHISHING]
    THRESH -->|≥ 0.55| V2[SUSPICIOUS]
    THRESH -->|≥ 0.30| V3[LIKELY BENIGN]
    THRESH -->|else| V4[BENIGN]

    V1 --> RESULT([Result<br/>score · verdict · signals])
    V2 --> RESULT
    V3 --> RESULT
    V4 --> RESULT
    SIGNALS --> RESULT

    classDef io fill:#e1f5ff,stroke:#0277bd,color:#01579b
    classDef heur fill:#f3e5f5,stroke:#7b1fa2,color:#4a148c
    classDef ml fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20
    classDef blend fill:#fff9c4,stroke:#f57f17,color:#e65100
    class URL,RESULT io
    class HEUR,HSCORE,SIGNALS heur
    class ML,MSCORE ml
    class BLEND,THRESH blend
```

If no trained model is available, the pipeline gracefully degrades to the
heuristic score alone — you still get a useful verdict, just without the
statistical layer.

### Training dataflow

`phishguard train --data <csv>` runs this pipeline once to produce a
serialized model that the inference path can load.

```mermaid
flowchart LR
    CSV([CSV<br/>url, label]) --> LOAD[Load &amp; validate<br/>require url + label columns]
    LOAD --> EXTRACT[Extract 31 features<br/>per URL]
    EXTRACT --> SPLIT[Stratified 80 / 20<br/>train / test split]
    SPLIT --> FIT[RandomForestClassifier<br/>n_estimators=200<br/>class_weight=balanced]
    FIT --> EVAL[Evaluate<br/>accuracy · precision · recall · F1<br/>+ feature importances]
    FIT --> SAVE([joblib.dump<br/>models/phishguard.joblib])

    classDef io fill:#e1f5ff,stroke:#0277bd,color:#01579b
    class CSV,SAVE io
```

### Module map

| Module | Responsibility |
| --- | --- |
| `phishguard.features` | URL parsing + 31-feature extraction (pure, no I/O) |
| `phishguard.heuristics` | Weighted rule scorer + signal explanations |
| `phishguard.model` | scikit-learn wrapper: train, save, load, predict |
| `phishguard.classifier` | High-level `classify()` that blends the two |
| `phishguard.cli` | `phishguard` command (`scan` / `train` / `features`) |

## Features extracted

Lexical / structural:

- URL, hostname, path, query lengths
- Counts of `.`, `-`, `/`, `?`, `=`, `&`, `@`, digits, special chars
- Digit ratio, letter ratio, special-char ratio
- Shannon entropy of hostname and full URL
- Number of subdomains, longest token length

Host signals:

- Hostname is a raw IPv4 / IPv6 address
- Punycode (`xn--`) present
- Uses non-standard port
- TLD is on a suspicious list (e.g. `.zip`, `.tk`, `.cf`, `.gq`, `.ml`)
- Known URL-shortener domain

Content signals:

- Phishing keywords (`login`, `verify`, `secure`, `account`, `update`, `bank`, …)
- Brand impersonation tokens (`paypal`, `apple`, `microsoft`, …) outside the
  registered domain
- Hex-encoded characters in path / query
- Double slashes after the scheme

## Quick start

```bash
# 1. Install (editable mode is handy while you're iterating)
pip install -e .

# 2. Train the model on the bundled seed dataset
phishguard train --data data/seed_dataset.csv --out models/phishguard.joblib

# 3. Score a URL
phishguard scan "http://paypa1-login.security-update.tk/verify?id=42"

# 4. After training
phishguard scan "http://paypa1-login.security-update.tk/verify?id=42" --model models/phishguard.joblib
```

Sample output:

```
URL: http://paypa1-login.security-update.tk/verify?id=42
Heuristic score : 0.82  (HIGH)
Model score     : 0.91  (HIGH)
Blended verdict : 0.87  PHISHING

Top signals:
  + suspicious TLD (.tk)
  + phishing keyword in path: 'verify'
  + brand impersonation token outside registered domain: 'paypal'
  + high hostname entropy (3.71)
  + uses HTTP, not HTTPS
```

## Library usage

```python
from phishguard import classify

result = classify("http://paypa1-login.security-update.tk/verify?id=42")
print(result.score, result.verdict, result.signals)
```

## Training on your own data

The trainer accepts a CSV with two columns: `url`, `label` (1 = phishing,
0 = benign). Public datasets that work out of the box:

- [PhishTank](https://www.phishtank.com/developer_info.php) (phishing samples)
- [Tranco list](https://tranco-list.eu/) (benign top sites)
- [Mendeley phishing dataset](https://data.mendeley.com/datasets/h3cgnj8hft)

```bash
phishguard train --data path/to/your.csv --out models/phishguard.joblib
```

## Project layout

```
phishing-url-classifier/
├── src/phishguard/
│   ├── __init__.py        # public API: classify(), Result
│   ├── features.py        # URL feature extraction
│   ├── heuristics.py      # rule-based scorer + signal explanations
│   ├── model.py           # scikit-learn wrapper (train / load / predict)
│   ├── cli.py             # `phishguard` command
│   └── data.py            # dataset loading helpers
├── data/seed_dataset.csv  # tiny labelled dataset to bootstrap training
├── tests/                 # pytest suite
├── pyproject.toml
└── requirements.txt
```

## Disclaimer

This tool is for **defensive research and education**. It will produce false
positives and false negatives. Do not use it as the sole gate for blocking
traffic in production. Pair it with reputation feeds (Google Safe Browsing,
PhishTank, etc.) and human review.
