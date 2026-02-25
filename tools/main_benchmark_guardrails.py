#!/usr/bin/env python3
"""Run main benchmark suite and enforce committed aggregate guardrails."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Main benchmark + guardrails wrapper")
    p.add_argument(
        "--guardrails-json",
        type=Path,
        default=Path("benchmarks/main_benchmark_guardrails.json"),
        help="Committed main benchmark guardrail config JSON",
    )
    p.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Skip running tools/run_benchmark_suite.py and only enforce thresholds",
    )
    p.add_argument(
        "--report-json",
        type=Path,
        help="Override benchmark_report.json path (or latest.json pointer) to evaluate",
    )
    p.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter for child commands (default: current interpreter)",
    )
    return p.parse_args()


def _run(cmd: list[str]) -> int:
    print("+", " ".join(cmd))
    return subprocess.run(cmd).returncode


def _load_guardrails(path: Path) -> dict[str, Any]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise ValueError("Guardrails JSON root must be an object")
    thresholds = doc.get("thresholds", {})
    if not isinstance(thresholds, dict):
        raise ValueError("'thresholds' must be an object")
    manifest = doc.get("manifest", "benchmarks/main_benchmark_songs.yaml")
    latest_pointer = doc.get("latest_pointer", "benchmarks/results/latest.json")
    benchmark_args = doc.get("benchmark_args", [])
    if not isinstance(manifest, str):
        raise ValueError("'manifest' must be a string")
    if not isinstance(latest_pointer, str):
        raise ValueError("'latest_pointer' must be a string")
    if not isinstance(benchmark_args, list) or not all(
        isinstance(v, str) for v in benchmark_args
    ):
        raise ValueError("'benchmark_args' must be a string list")
    return {
        "manifest": Path(manifest),
        "latest_pointer": Path(latest_pointer),
        "benchmark_args": list(benchmark_args),
        "thresholds": thresholds,
    }


def _resolve_report_json(*, override: Path | None, latest_pointer: Path) -> Path:
    if override is not None:
        return override
    if latest_pointer.exists():
        raw = latest_pointer.read_text(encoding="utf-8").strip()
        if raw:
            return Path(raw)
    raise FileNotFoundError(
        f"Could not resolve benchmark report from latest pointer: {latest_pointer}"
    )


def _metric_value(report: dict[str, Any], key: str) -> float | None:
    aggregate = report.get("aggregate", {})
    if not isinstance(aggregate, dict):
        return None
    value = aggregate.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    if key in {
        "reference_divergence_suspected_ratio",
        "reference_divergence_suspected_count",
    }:
        # Backward compatibility for benchmark reports generated before the
        # reference-divergence signal was added.
        return 0.0
    return None


def _check_thresholds(report: dict[str, Any], thresholds: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    for key, raw_threshold in thresholds.items():
        if raw_threshold is None:
            continue
        if not isinstance(raw_threshold, (int, float)):
            issues.append(f"Invalid threshold {key!r}: must be numeric or null")
            continue
        threshold = float(raw_threshold)
        if key.startswith("min_"):
            metric_key = key[4:]
            value = _metric_value(report, metric_key)
            if value is None:
                issues.append(f"Missing aggregate metric for threshold: {metric_key}")
                continue
            if value < threshold:
                issues.append(
                    f"{metric_key} below threshold: {value:.4f} < {threshold:.4f}"
                )
            continue
        if key.startswith("max_"):
            metric_key = key[4:]
            value = _metric_value(report, metric_key)
            if value is None:
                issues.append(f"Missing aggregate metric for threshold: {metric_key}")
                continue
            if value > threshold:
                issues.append(
                    f"{metric_key} above threshold: {value:.4f} > {threshold:.4f}"
                )
            continue
        issues.append(
            f"Unsupported threshold key {key!r}; use min_<metric> or max_<metric>"
        )
    return issues


def main() -> int:
    args = _parse_args()
    cfg = _load_guardrails(args.guardrails_json)

    if not args.skip_benchmark:
        cmd = [
            args.python,
            "tools/run_benchmark_suite.py",
            "--manifest",
            str(cfg["manifest"]),
        ]
        cmd.extend(cfg["benchmark_args"])
        rc = _run(cmd)
        if rc != 0:
            return rc

    report_json_path = _resolve_report_json(
        override=args.report_json,
        latest_pointer=cfg["latest_pointer"],
    )
    report = json.loads(report_json_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        print("main_benchmark_guardrails: FAIL")
        print("- benchmark report root must be an object")
        return 1
    issues = _check_thresholds(report, cfg["thresholds"])
    if issues:
        print("main_benchmark_guardrails: FAIL")
        for issue in issues:
            print(f"- {issue}")
        return 1
    aggregate = report.get("aggregate", {}) if isinstance(report, dict) else {}
    suspect_count = (
        int(aggregate.get("reference_divergence_suspected_count", 0) or 0)
        if isinstance(aggregate, dict)
        else 0
    )
    print(
        "main_benchmark_guardrails: OK "
        f"({report_json_path}, reference_divergence_suspected_count={suspect_count})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
