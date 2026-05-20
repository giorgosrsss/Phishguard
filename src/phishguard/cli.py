"""Command-line interface for PhishGuard.

Subcommands:

* ``scan`` — score one or more URLs (positional args, ``--file``, or stdin).
* ``train`` — train a new ML model from a CSV of (url, label) pairs.
* ``features`` — dump the raw feature vector for a URL as JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from . import __version__
from .classifier import classify
from .model import DEFAULT_MODEL_PATH, PhishingModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _iter_urls(args: argparse.Namespace) -> Iterable[str]:
    """Yield URLs from positional args, ``--file``, or stdin in that order."""
    seen_any = False
    for url in args.urls or []:
        seen_any = True
        yield url
    if args.file:
        path = Path(args.file)
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#"):
                    seen_any = True
                    yield line
    if not seen_any and not sys.stdin.isatty():
        for line in sys.stdin:
            line = line.strip()
            if line:
                yield line


def _load_csv(path: Path) -> Tuple[List[str], List[int]]:
    """Load a CSV with at least ``url`` and ``label`` columns.

    Labels are coerced to int. Anything not parseable as 0/1 raises.
    """
    urls: List[str] = []
    labels: List[int] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or "url" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError(
                f"{path} must have a header with at least 'url' and 'label' columns"
            )
        for i, row in enumerate(reader, start=2):  # start=2 to account for header
            url = (row.get("url") or "").strip()
            label_raw = (row.get("label") or "").strip()
            if not url or not label_raw:
                continue
            try:
                label = int(label_raw)
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{i}: label must be 0 or 1, got {label_raw!r}"
                ) from exc
            if label not in (0, 1):
                raise ValueError(
                    f"{path}:{i}: label must be 0 or 1, got {label}"
                )
            urls.append(url)
            labels.append(label)
    if not urls:
        raise ValueError(f"{path} contained no usable rows")
    return urls, labels


def _resolve_model(path_arg: Optional[str]) -> Optional[PhishingModel]:
    """Load a model from ``path_arg`` if provided, else from the default
    location, else return None."""
    if path_arg:
        return PhishingModel.load(Path(path_arg))
    if DEFAULT_MODEL_PATH.exists():
        try:
            return PhishingModel.load(DEFAULT_MODEL_PATH)
        except Exception as exc:  # noqa: BLE001
            print(
                f"warning: failed to load default model at {DEFAULT_MODEL_PATH}: {exc}",
                file=sys.stderr,
            )
    return None


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def _cmd_scan(args: argparse.Namespace) -> int:
    model = _resolve_model(args.model)
    urls = list(_iter_urls(args))
    if not urls:
        print("error: no URLs provided (pass them as args, --file, or stdin)", file=sys.stderr)
        return 2

    if args.json:
        out = []
        for url in urls:
            r = classify(url, model=model)
            out.append({
                "url": r.url,
                "score": round(r.score, 4),
                "verdict": r.verdict,
                "heuristic_score": round(r.heuristic_score, 4),
                "model_score": None if r.model_score is None else round(r.model_score, 4),
                "signals": r.signals,
            })
        json.dump(out if len(out) > 1 else out[0], sys.stdout, indent=2)
        sys.stdout.write("\n")
        return _exit_code_for_results(out, args.fail_threshold)

    # Pretty mode.
    worst_score = 0.0
    for i, url in enumerate(urls):
        if i > 0:
            print()
            print("-" * 60)
            print()
        result = classify(url, model=model)
        worst_score = max(worst_score, result.score)
        print(result.pretty())

    if args.fail_threshold is not None and worst_score >= args.fail_threshold:
        return 1
    return 0


def _exit_code_for_results(results: list, threshold: Optional[float]) -> int:
    if threshold is None:
        return 0
    worst = max((r["score"] for r in results), default=0.0)
    return 1 if worst >= threshold else 0


def _cmd_train(args: argparse.Namespace) -> int:
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"error: {data_path} does not exist", file=sys.stderr)
        return 2

    print(f"Loading dataset: {data_path}")
    urls, labels = _load_csv(data_path)
    n_phish = sum(labels)
    print(f"  {len(urls)} samples ({n_phish} phishing, {len(urls) - n_phish} benign)")

    print("Training model...")
    model, report = PhishingModel.train(
        urls,
        labels,
        test_size=args.test_size,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
    )
    print(report.pretty())

    out_path = Path(args.out) if args.out else DEFAULT_MODEL_PATH
    model.save(out_path)
    print(f"\nSaved model -> {out_path}")
    return 0


def _cmd_features(args: argparse.Namespace) -> int:
    from .features import extract_features
    feats = extract_features(args.url)
    json.dump(feats, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


# ---------------------------------------------------------------------------
# Argparse wiring
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phishguard",
        description="Hybrid phishing URL classifier (heuristics + ML).",
    )
    parser.add_argument("--version", action="version", version=f"phishguard {__version__}")

    sub = parser.add_subparsers(dest="command", required=True)

    # scan
    scan = sub.add_parser("scan", help="Score one or more URLs")
    scan.add_argument("urls", nargs="*", help="URLs to score")
    scan.add_argument("-f", "--file", help="Read URLs from file (one per line)")
    scan.add_argument("--model", help=f"Path to model file (default: {DEFAULT_MODEL_PATH})")
    scan.add_argument("--json", action="store_true", help="Emit JSON instead of pretty text")
    scan.add_argument(
        "--fail-threshold",
        type=float,
        default=None,
        help="Exit with code 1 if any URL scores >= this threshold (useful in CI)",
    )
    scan.set_defaults(func=_cmd_scan)

    # train
    train = sub.add_parser("train", help="Train a new model from a CSV dataset")
    train.add_argument("--data", required=True, help="CSV with 'url' and 'label' columns")
    train.add_argument("--out", help=f"Output model path (default: {DEFAULT_MODEL_PATH})")
    train.add_argument("--test-size", type=float, default=0.2)
    train.add_argument("--random-state", type=int, default=42)
    train.add_argument("--n-estimators", type=int, default=200)
    train.set_defaults(func=_cmd_train)

    # features
    feats = sub.add_parser("features", help="Print the feature vector for a URL")
    feats.add_argument("url")
    feats.set_defaults(func=_cmd_features)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
