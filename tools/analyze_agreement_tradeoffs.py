#!/usr/bin/env python3
"""Analyze agreement tradeoffs across benchmark reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _resolve_report(path: Path) -> Path:
    return path / "benchmark_report.json" if path.is_dir() else path


def _load_report(path: Path) -> dict[str, Any]:
    return json.loads(_resolve_report(path).read_text(encoding="utf-8"))


def _aggregate_metrics(report: dict[str, Any]) -> dict[str, float]:
    agg = report.get("aggregate", {}) or {}
    if not isinstance(agg, dict):
        agg = {}
    return {
        "agreement_coverage_ratio_total": float(
            agg.get("agreement_coverage_ratio_total", 0.0) or 0.0
        ),
        "agreement_start_p95_abs_sec_mean": float(
            agg.get("agreement_start_p95_abs_sec_mean", 0.0) or 0.0
        ),
        "agreement_bad_ratio_total": float(
            agg.get("agreement_bad_ratio_total", 0.0) or 0.0
        ),
        "timing_quality_score_line_weighted_mean": float(
            agg.get("timing_quality_score_line_weighted_mean", 0.0) or 0.0
        ),
    }


def _parse_labeled_report(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"expected LABEL=PATH, got: {value!r}")
    label, path_raw = value.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"empty report label in {value!r}")
    return label, Path(path_raw.strip()).expanduser().resolve()


def _analyze(
    *,
    baseline_label: str,
    baseline_report: dict[str, Any],
    candidates: list[tuple[str, dict[str, Any]]],
    min_coverage_gain: float,
    max_bad_ratio_increase: float,
) -> dict[str, Any]:
    baseline = _aggregate_metrics(baseline_report)
    rows: list[dict[str, Any]] = []
    for label, report in candidates:
        metrics = _aggregate_metrics(report)
        coverage_gain = (
            metrics["agreement_coverage_ratio_total"]
            - baseline["agreement_coverage_ratio_total"]
        )
        bad_ratio_increase = (
            metrics["agreement_bad_ratio_total"] - baseline["agreement_bad_ratio_total"]
        )
        p95_delta = (
            metrics["agreement_start_p95_abs_sec_mean"]
            - baseline["agreement_start_p95_abs_sec_mean"]
        )
        timing_delta = (
            metrics["timing_quality_score_line_weighted_mean"]
            - baseline["timing_quality_score_line_weighted_mean"]
        )
        passes_guard = (
            coverage_gain >= min_coverage_gain
            and bad_ratio_increase <= max_bad_ratio_increase
        )
        score = (
            (coverage_gain * 3.0)
            - max(0.0, bad_ratio_increase) * 4.0
            - max(0.0, p95_delta) * 0.4
            + timing_delta * 0.5
        )
        rows.append(
            {
                "label": label,
                "coverage_gain": round(coverage_gain, 4),
                "bad_ratio_increase": round(bad_ratio_increase, 4),
                "p95_delta_sec": round(p95_delta, 4),
                "timing_quality_delta": round(timing_delta, 4),
                "passes_tradeoff_guard": bool(passes_guard),
                "tradeoff_score": round(score, 5),
            }
        )
    rows.sort(key=lambda row: float(row.get("tradeoff_score", 0.0)), reverse=True)
    best_score_candidate = rows[0]["label"] if rows else None
    best_guard_pass_candidate = None
    for row in rows:
        if bool(row.get("passes_tradeoff_guard")):
            best_guard_pass_candidate = row.get("label")
            break
    return {
        "baseline_label": baseline_label,
        "baseline_metrics": baseline,
        "best_score_candidate": best_score_candidate,
        "best_guard_pass_candidate": best_guard_pass_candidate,
        "rows": rows,
        "guard": {
            "min_coverage_gain": min_coverage_gain,
            "max_bad_ratio_increase": max_bad_ratio_increase,
        },
    }


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# Agreement Tradeoff Analysis", ""]
    lines.append(f"- Baseline: `{payload.get('baseline_label', '')}`")
    guard = payload.get("guard", {}) or {}
    lines.append(
        "- Guard: "
        f"`min_coverage_gain={float(guard.get('min_coverage_gain', 0.0) or 0.0):.4f}`, "
        f"`max_bad_ratio_increase={float(guard.get('max_bad_ratio_increase', 0.0) or 0.0):.4f}`"
    )
    lines.append(
        f"- Best score candidate: `{payload.get('best_score_candidate', None)}`"
    )
    lines.append(
        f"- Best guard-pass candidate: `{payload.get('best_guard_pass_candidate', None)}`"
    )
    lines.append("")
    lines.append(
        "| Label | Score | Coverage Δ | Bad Ratio Δ | P95 Δ (s) | Timing Δ | Guard Pass |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for row in payload.get("rows", []) or []:
        lines.append(
            f"| {row.get('label', '')} | {float(row.get('tradeoff_score', 0.0)):+.5f} | "
            f"{float(row.get('coverage_gain', 0.0)):+.4f} | "
            f"{float(row.get('bad_ratio_increase', 0.0)):+.4f} | "
            f"{float(row.get('p95_delta_sec', 0.0)):+.4f} | "
            f"{float(row.get('timing_quality_delta', 0.0)):+.4f} | "
            f"{'yes' if bool(row.get('passes_tradeoff_guard')) else 'no'} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        required=True,
        help="Baseline report in LABEL=PATH form",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="Candidate report in LABEL=PATH form (repeatable)",
    )
    parser.add_argument(
        "--min-coverage-gain",
        type=float,
        default=0.005,
        help="Coverage gain threshold for guard check",
    )
    parser.add_argument(
        "--max-bad-ratio-increase",
        type=float,
        default=0.002,
        help="Bad-ratio increase threshold for guard check",
    )
    parser.add_argument(
        "--output-json", type=Path, default=None, help="Output JSON path"
    )
    parser.add_argument(
        "--output-md", type=Path, default=None, help="Output markdown path"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    baseline_label, baseline_path = _parse_labeled_report(args.baseline)
    candidate_items = [_parse_labeled_report(value) for value in args.candidate]
    baseline_doc = _load_report(baseline_path)
    candidates = [(label, _load_report(path)) for label, path in candidate_items]

    payload = _analyze(
        baseline_label=baseline_label,
        baseline_report=baseline_doc,
        candidates=candidates,
        min_coverage_gain=float(args.min_coverage_gain),
        max_bad_ratio_increase=float(args.max_bad_ratio_increase),
    )
    out_root = baseline_path.parent
    out_json = args.output_json or (out_root / "agreement_tradeoff_analysis.json")
    out_md = args.output_md or (out_root / "agreement_tradeoff_analysis.md")
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_markdown(out_md, payload)
    print("agreement_tradeoff_analysis: OK")
    print(f"  candidates={len(payload.get('rows', []))}")
    print(f"  output_json={out_json}")
    print(f"  output_md={out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
