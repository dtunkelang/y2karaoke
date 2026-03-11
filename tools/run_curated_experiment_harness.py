#!/usr/bin/env python3
"""Run baseline vs env-gated candidate experiments on the curated canary slice."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "benchmarks" / "results"
DEFAULT_GOLD_ROOT = REPO_ROOT / "benchmarks" / "gold_set_candidate" / "20260305T231015Z"
DEFAULT_MATCH = "Blinding Lights|Derniere danse|Mi Gente|DESPECHA"


@dataclass(frozen=True)
class ExperimentPreset:
    name: str
    env: dict[str, str]
    description: str


PRESETS: dict[str, ExperimentPreset] = {
    "baseline": ExperimentPreset(
        name="baseline",
        env={},
        description="Current default curated canary behavior",
    ),
    "repeat_duration": ExperimentPreset(
        name="repeat_duration",
        env={"Y2K_REPEAT_DURATION_NORMALIZE": "1"},
        description="Repeat-duration normalization experiment",
    ),
    "parallel_segment_assigner": ExperimentPreset(
        name="parallel_segment_assigner",
        env={"Y2K_WHISPER_SEGMENT_ASSIGN_PIPELINE": "parallel_experimental"},
        description="Parallel experimental segment assigner scaffold",
    ),
}


def _parse_env_overrides(values: list[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"invalid env override {value!r}; expected KEY=VALUE")
        key, raw = value.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"invalid env override {value!r}; empty key")
        env[key] = raw
    return env


def _run_benchmark(
    *,
    run_id: str,
    output_root: Path,
    gold_root: Path,
    match: str,
    max_songs: int,
    python_bin: str,
    env_overrides: dict[str, str],
) -> Path:
    cmd = [
        python_bin,
        str(REPO_ROOT / "tools" / "run_benchmark_suite.py"),
        "--run-id",
        run_id,
        "--output-root",
        str(output_root),
        "--offline",
        "--force",
        "--gold-root",
        str(gold_root),
        "--match",
        match,
        "--max-songs",
        str(max_songs),
    ]
    env = os.environ.copy()
    env.update(env_overrides)
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)
    return output_root / run_id / "benchmark_report.json"


def _run_compare(*, baseline: Path, corrected: Path, python_bin: str) -> None:
    cmd = [
        python_bin,
        str(REPO_ROOT / "tools" / "compare_benchmark_correction.py"),
        "--baseline",
        str(baseline.parent),
        "--corrected",
        str(corrected.parent),
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def _load_aggregate(report_path: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return report.get("aggregate", {})


def _print_summary(label: str, aggregate: dict[str, Any]) -> None:
    print(label)
    for key in (
        "curated_canary_avg_abs_word_start_delta_sec_word_weighted_mean",
        "curated_canary_gold_start_p95_abs_sec_mean",
        "curated_canary_gold_line_duration_mean_abs_sec_mean",
        "curated_canary_gold_downstream_regression_line_count_total",
        "curated_canary_gold_downstream_regression_mean_improvement_sec_mean",
    ):
        print(f"  {key}={aggregate.get(key)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        choices=sorted(PRESETS),
        required=True,
        help="Named env-gated experiment preset",
    )
    parser.add_argument(
        "--run-prefix",
        required=True,
        help="Prefix used to build baseline/candidate run ids",
    )
    parser.add_argument(
        "--match",
        default=DEFAULT_MATCH,
        help="Regex match forwarded to run_benchmark_suite.py",
    )
    parser.add_argument(
        "--max-songs",
        type=int,
        default=4,
        help="Number of songs to run in the curated slice",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Benchmark results root",
    )
    parser.add_argument(
        "--gold-root",
        type=Path,
        default=DEFAULT_GOLD_ROOT,
        help="Curated gold root",
    )
    parser.add_argument(
        "--python-bin",
        default=str(REPO_ROOT / "venv" / "bin" / "python"),
        help="Python interpreter to use",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        help="Additional KEY=VALUE env override for the candidate run",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    preset = PRESETS[args.experiment]
    extra_env = _parse_env_overrides(list(args.env))
    baseline_run_id = f"{args.run_prefix}_baseline"
    candidate_run_id = f"{args.run_prefix}_{preset.name}"

    baseline_report = _run_benchmark(
        run_id=baseline_run_id,
        output_root=args.output_root,
        gold_root=args.gold_root,
        match=args.match,
        max_songs=args.max_songs,
        python_bin=args.python_bin,
        env_overrides={},
    )
    candidate_env = dict(preset.env)
    candidate_env.update(extra_env)
    candidate_report = _run_benchmark(
        run_id=candidate_run_id,
        output_root=args.output_root,
        gold_root=args.gold_root,
        match=args.match,
        max_songs=args.max_songs,
        python_bin=args.python_bin,
        env_overrides=candidate_env,
    )

    _run_compare(
        baseline=baseline_report,
        corrected=candidate_report,
        python_bin=args.python_bin,
    )

    baseline_agg = _load_aggregate(baseline_report)
    candidate_agg = _load_aggregate(candidate_report)
    print(f"preset={preset.name} description={preset.description}")
    _print_summary("baseline", baseline_agg)
    _print_summary("candidate", candidate_agg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
