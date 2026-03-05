#!/usr/bin/env python3
"""Run agreement-threshold benchmark sweeps and summarize tradeoffs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any


def _parse_float_list(raw: str) -> list[float]:
    values: list[float] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    if not values:
        raise ValueError("expected at least one float value")
    return values


def _candidate_label(text_sim: float, token_overlap: float) -> str:
    return f"ts{int(round(text_sim * 100)):02d}_to{int(round(token_overlap * 100)):02d}"


def _run_command(cmd: list[str], env: dict[str, str]) -> None:
    proc = subprocess.run(cmd, env=env, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline", type=Path, required=True, help="Baseline run dir/report"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("benchmarks/main_benchmark_songs.yaml"),
        help="Benchmark manifest to run for each threshold combo",
    )
    parser.add_argument(
        "--text-sim-values",
        type=str,
        required=True,
        help="Comma-separated text similarity thresholds, e.g. 0.60,0.58",
    )
    parser.add_argument(
        "--token-overlap-values",
        type=str,
        required=True,
        help="Comma-separated token overlap thresholds, e.g. 0.50,0.48",
    )
    parser.add_argument(
        "--run-id-prefix",
        type=str,
        default="agreement_sweep",
        help="Run-id prefix for generated benchmark runs",
    )
    parser.add_argument("--strategy", type=str, default="hybrid_whisper")
    parser.add_argument(
        "--offline", action="store_true", help="Run benchmark in offline mode"
    )
    parser.add_argument(
        "--min-coverage-gain",
        type=float,
        default=0.005,
        help="Tradeoff guard threshold",
    )
    parser.add_argument(
        "--max-bad-ratio-increase",
        type=float,
        default=0.002,
        help="Tradeoff guard threshold",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("benchmarks/results/agreement_tradeoff_sweep.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("benchmarks/results/agreement_tradeoff_sweep.md"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    text_values = _parse_float_list(args.text_sim_values)
    overlap_values = _parse_float_list(args.token_overlap_values)

    baseline_path = args.baseline.expanduser().resolve()
    baseline_label = "base"
    candidate_specs: list[tuple[str, Path]] = []
    for text_sim in text_values:
        for overlap in overlap_values:
            label = _candidate_label(text_sim, overlap)
            run_id = f"{args.run_id_prefix}_{label}"
            run_dir = Path("benchmarks/results") / run_id
            env = dict(os.environ)
            env["Y2KARAOKE_BENCH_AGREEMENT_MIN_TEXT_SIM"] = f"{text_sim:.2f}"
            env["Y2KARAOKE_BENCH_AGREEMENT_MIN_TOKEN_OVERLAP"] = f"{overlap:.2f}"
            cmd = [
                "./venv/bin/python",
                "tools/run_benchmark_suite.py",
                "--manifest",
                str(args.manifest),
                "--strategy",
                str(args.strategy),
                "--run-id",
                run_id,
            ]
            if args.offline:
                cmd.append("--offline")
            print(f"sweep_run: label={label} run_id={run_id}")
            _run_command(cmd, env=env)
            candidate_specs.append((label, run_dir.resolve()))

    analyze_cmd = [
        "./venv/bin/python",
        "tools/analyze_agreement_tradeoffs.py",
        "--baseline",
        f"{baseline_label}={baseline_path}",
        "--min-coverage-gain",
        f"{args.min_coverage_gain}",
        "--max-bad-ratio-increase",
        f"{args.max_bad_ratio_increase}",
        "--output-json",
        str(args.output_json),
        "--output-md",
        str(args.output_md),
    ]
    for label, path in candidate_specs:
        analyze_cmd.extend(["--candidate", f"{label}={path}"])
    _run_command(analyze_cmd, env=dict(os.environ))

    payload: dict[str, Any] = {
        "baseline": str(baseline_path),
        "candidates": [
            {"label": label, "run_dir": str(path)} for label, path in candidate_specs
        ],
        "analysis_json": str(args.output_json),
        "analysis_md": str(args.output_md),
    }
    print("agreement_threshold_sweep: OK")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
