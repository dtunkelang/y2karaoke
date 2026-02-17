#!/usr/bin/env python3
"""Calibrate bootstrap suitability/confidence thresholds from report JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import quantiles
from typing import Any


def _iter_reports(root: Path, pattern: str) -> list[Path]:
    return sorted(p for p in root.glob(pattern) if p.is_file())


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    n = 100
    bins = quantiles(values, n=n, method="inclusive")
    idx = max(0, min(n - 2, int(round(q * (n - 1))) - 1))
    return float(bins[idx])


def _collect_metrics(report_docs: list[dict[str, Any]]) -> dict[str, list[float]]:
    detectability: list[float] = []
    word_level: list[float] = []
    line_conf: list[float] = []
    word_conf: list[float] = []

    for doc in report_docs:
        vs = doc.get("selected_visual_suitability") or {}
        d = vs.get("detectability_score")
        w = vs.get("word_level_score")
        if isinstance(d, (int, float)):
            detectability.append(float(d))
        if isinstance(w, (int, float)):
            word_level.append(float(w))

        out_path = doc.get("output_path")
        if isinstance(out_path, str):
            p = Path(out_path)
            if p.exists():
                try:
                    gold_doc = json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    continue
                for line in gold_doc.get("lines", []):
                    lc = line.get("confidence")
                    if isinstance(lc, (int, float)):
                        line_conf.append(float(lc))
                    for word in line.get("words", []):
                        wc = word.get("confidence")
                        if isinstance(wc, (int, float)):
                            word_conf.append(float(wc))

    return {
        "detectability": detectability,
        "word_level": word_level,
        "line_conf": line_conf,
        "word_conf": word_conf,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate bootstrap thresholds")
    parser.add_argument("--root", type=Path, default=Path("."), help="Repository root")
    parser.add_argument(
        "--glob",
        default="benchmarks/**/*.bootstrap-report.json",
        help="Glob pattern for report JSON files",
    )
    parser.add_argument(
        "--safety-percentile",
        type=float,
        default=0.2,
        help="Percentile used for conservative minimum thresholds",
    )
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    reports = _iter_reports(args.root.resolve(), args.glob)
    docs: list[dict[str, Any]] = []
    for rp in reports:
        try:
            docs.append(json.loads(rp.read_text(encoding="utf-8")))
        except Exception:
            continue

    m = _collect_metrics(docs)
    p = max(0.01, min(0.99, args.safety_percentile))

    recommendation = {
        "report_count": len(docs),
        "source_pattern": args.glob,
        "safety_percentile": p,
        "recommended": {
            "min_detectability": round(_pct(m["detectability"], p), 3),
            "min_word_level_score": round(_pct(m["word_level"], p), 3),
            "min_line_confidence_mean": round(_pct(m["line_conf"], p), 3),
            "min_word_confidence_mean": round(_pct(m["word_conf"], p), 3),
        },
        "observed": {
            "detectability_count": len(m["detectability"]),
            "word_level_count": len(m["word_level"]),
            "line_conf_count": len(m["line_conf"]),
            "word_conf_count": len(m["word_conf"]),
        },
    }

    if args.json:
        print(json.dumps(recommendation, indent=2))
    else:
        print("bootstrap_calibrate_thresholds")
        print(f"  reports: {recommendation['report_count']}")
        print(f"  safety_percentile: {p:.2f}")
        r = recommendation["recommended"]
        print("  suggested thresholds:")
        print(f"    --min-detectability {r['min_detectability']}")
        print(f"    --min-word-level-score {r['min_word_level_score']}")
        print(f"    --min-line-confidence-mean {r['min_line_confidence_mean']}")
        print(f"    --min-word-confidence-mean {r['min_word_confidence_mean']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
