#!/usr/bin/env python3
"""Recommend default benchmark strategy and bootstrap thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "benchmarks" / "results"


def _latest_matrix_report() -> Path | None:
    candidates = sorted(RESULTS_ROOT.glob("*-matrix/strategy_matrix_report.json"))
    return candidates[-1] if candidates else None


def _score_row(row: dict[str, Any]) -> float:
    """Heuristic score: higher is better."""

    def fnum(key: str) -> float | None:
        v = row.get(key)
        return float(v) if isinstance(v, (int, float)) else None

    score = 0.0
    p95 = fnum("agreement_start_p95_abs_sec_line_weighted_mean")
    mean = fnum("agreement_start_mean_abs_sec_line_weighted_mean")
    low = fnum("low_confidence_ratio_line_weighted_mean")
    dtw = fnum("dtw_line_coverage_line_weighted_mean")
    ok = fnum("songs_succeeded")
    total = fnum("songs_total")

    if p95 is not None:
        score += 2.0 * (1.0 / (1.0 + max(p95, 0.0)))
    if mean is not None:
        score += 1.5 * (1.0 / (1.0 + max(mean, 0.0)))
    if low is not None:
        score += 1.0 * max(0.0, 1.0 - min(low, 1.0))
    if dtw is not None:
        score += 1.0 * max(0.0, min(dtw, 1.0))
    if ok is not None and total is not None and total > 0:
        score += 1.0 * max(0.0, min(ok / total, 1.0))

    return round(score, 6)


def _best_strategy(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    scored = []
    for row in rows:
        if row.get("status") == "failed":
            continue
        scored.append({**row, "composite_score": _score_row(row)})
    if not scored:
        return None
    scored.sort(key=lambda r: r["composite_score"], reverse=True)
    return scored[0]


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix-report",
        type=Path,
        default=None,
        help="Path to strategy_matrix_report.json (default: latest under benchmarks/results)",
    )
    parser.add_argument(
        "--calibration-report",
        type=Path,
        default=None,
        help="Path to bootstrap calibration JSON output (optional)",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    matrix_path = args.matrix_report or _latest_matrix_report()
    if matrix_path is None:
        print("No matrix report found. Run make benchmark-matrix first.")
        return 1

    matrix_doc = _load_json(matrix_path)
    if not matrix_doc:
        print(f"Invalid matrix report: {matrix_path}")
        return 1

    rows = matrix_doc.get("rows", [])
    if not isinstance(rows, list):
        print(f"Invalid matrix rows in report: {matrix_path}")
        return 1

    best = _best_strategy([r for r in rows if isinstance(r, dict)])
    if best is None:
        print("No successful strategies found in matrix report.")
        return 1

    calibration = _load_json(args.calibration_report)
    recommended_thresholds = {}
    if calibration:
        rec = calibration.get("recommended", {})
        if isinstance(rec, dict):
            for k in (
                "min_detectability",
                "min_word_level_score",
                "min_line_confidence_mean",
                "min_word_confidence_mean",
            ):
                v = rec.get(k)
                if isinstance(v, (int, float)):
                    recommended_thresholds[k] = float(v)

    out = {
        "matrix_report": str(matrix_path),
        "recommended_strategy": best.get("strategy"),
        "strategy_score": best.get("composite_score"),
        "strategy_metrics": {
            "status": best.get("status"),
            "songs_total": best.get("songs_total"),
            "songs_succeeded": best.get("songs_succeeded"),
            "dtw_line_coverage_line_weighted_mean": best.get(
                "dtw_line_coverage_line_weighted_mean"
            ),
            "agreement_start_mean_abs_sec_line_weighted_mean": best.get(
                "agreement_start_mean_abs_sec_line_weighted_mean"
            ),
            "agreement_start_p95_abs_sec_line_weighted_mean": best.get(
                "agreement_start_p95_abs_sec_line_weighted_mean"
            ),
            "low_confidence_ratio_line_weighted_mean": best.get(
                "low_confidence_ratio_line_weighted_mean"
            ),
        },
        "recommended_bootstrap_thresholds": recommended_thresholds,
        "commands": {
            "benchmark": f"./venv/bin/python tools/run_benchmark_suite.py --strategy {best.get('strategy')}",
            "bootstrap": (
                "./venv/bin/python tools/bootstrap_gold_from_karaoke.py"
                + (
                    " "
                    + " ".join(
                        [
                            f"--min-detectability {recommended_thresholds.get('min_detectability')}",
                            f"--min-word-level-score {recommended_thresholds.get('min_word_level_score')}",
                        ]
                    )
                    if recommended_thresholds
                    else ""
                )
            ).strip(),
        },
    }

    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print("benchmark default recommendations")
        print(f"  matrix_report: {out['matrix_report']}")
        print(
            f"  strategy: {out['recommended_strategy']} (score={out['strategy_score']})"
        )
        if recommended_thresholds:
            print("  bootstrap thresholds:")
            for k, v in recommended_thresholds.items():
                print(f"    {k}: {v}")
        print("  commands:")
        print(f"    benchmark: {out['commands']['benchmark']}")
        print(f"    bootstrap: {out['commands']['bootstrap']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
