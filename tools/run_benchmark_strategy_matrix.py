#!/usr/bin/env python3
"""Run benchmark suite across multiple strategies and summarize results."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "benchmarks" / "results"


@dataclass
class StrategyRun:
    strategy: str
    run_id: str
    run_dir: Path
    exit_code: int
    report_path: Path


def _build_command(
    *,
    python_bin: str,
    strategy: str,
    run_id: str,
    output_root: Path,
    manifest: Path,
    gold_root: Path,
    cache_dir: Path | None,
    offline: bool,
    force: bool,
    max_songs: int,
    match: str,
    scenario: str,
) -> list[str]:
    cmd = [
        python_bin,
        str(REPO_ROOT / "tools" / "run_benchmark_suite.py"),
        "--strategy",
        strategy,
        "--run-id",
        run_id,
        "--output-root",
        str(output_root),
        "--manifest",
        str(manifest),
        "--gold-root",
        str(gold_root),
        "--scenario",
        scenario,
    ]
    if cache_dir is not None:
        cmd.extend(["--cache-dir", str(cache_dir)])
    if offline:
        cmd.append("--offline")
    if force:
        cmd.append("--force")
    if max_songs > 0:
        cmd.extend(["--max-songs", str(max_songs)])
    if match:
        cmd.extend(["--match", match])
    return cmd


def _extract_summary(report_json: dict[str, Any]) -> dict[str, Any]:
    aggregate = report_json.get("aggregate", {})

    def metric(*keys: str):
        for key in keys:
            if key in aggregate and aggregate.get(key) is not None:
                return aggregate.get(key)
        return None

    return {
        "status": report_json.get("status", "unknown"),
        "songs_total": aggregate.get("songs_total"),
        "songs_succeeded": aggregate.get("songs_succeeded"),
        "songs_failed": aggregate.get("songs_failed"),
        "dtw_line_coverage_line_weighted_mean": metric(
            "dtw_line_coverage_line_weighted_mean",
            "dtw_line_coverage_mean",
        ),
        "dtw_word_coverage_line_weighted_mean": metric(
            "dtw_word_coverage_line_weighted_mean",
            "dtw_word_coverage_mean",
        ),
        "agreement_start_mean_abs_sec_line_weighted_mean": metric(
            "agreement_start_mean_abs_sec_line_weighted_mean",
            "agreement_start_mean_abs_sec_mean",
        ),
        "agreement_start_p95_abs_sec_line_weighted_mean": metric(
            "agreement_start_p95_abs_sec_line_weighted_mean",
            "agreement_start_p95_abs_sec_mean",
        ),
        "whisper_anchor_start_mean_abs_sec_mean": metric(
            "whisper_anchor_start_mean_abs_sec_line_weighted_mean",
            "whisper_anchor_start_mean_abs_sec_mean",
        ),
        "whisper_anchor_start_p95_abs_sec_mean": metric(
            "whisper_anchor_start_p95_abs_sec_line_weighted_mean",
            "whisper_anchor_start_p95_abs_sec_mean",
        ),
        "low_confidence_ratio_line_weighted_mean": metric(
            "low_confidence_ratio_line_weighted_mean",
            "low_confidence_ratio_total",
        ),
        "gold_start_abs_word_weighted_mean": metric(
            "avg_abs_word_start_delta_sec_word_weighted_mean",
            "avg_abs_word_start_delta_sec_mean",
        ),
        "sum_song_elapsed_sec": report_json.get("sum_song_elapsed_sec"),
        "suite_wall_elapsed_sec": report_json.get("suite_wall_elapsed_sec"),
    }


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Benchmark Strategy Matrix",
        "",
        (
            "| strategy | status | songs ok/total | dtw line cov | dtw word cov | "
            "mean start abs (s) | p95 start abs (s) | low-conf ratio | sum song sec |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        ok = row.get("songs_succeeded")
        total = row.get("songs_total")
        lines.append(
            "| {strategy} | {status} | {ok}/{total} | {dl} | {dw} | {mean} | {p95} | {low} | {elapsed} |".format(
                strategy=row.get("strategy"),
                status=row.get("status"),
                ok=ok if ok is not None else "-",
                total=total if total is not None else "-",
                dl=row.get("dtw_line_coverage_line_weighted_mean", "-"),
                dw=row.get("dtw_word_coverage_line_weighted_mean", "-"),
                mean=row.get("agreement_start_mean_abs_sec_line_weighted_mean", "-"),
                p95=row.get("agreement_start_p95_abs_sec_line_weighted_mean", "-"),
                low=row.get("low_confidence_ratio_line_weighted_mean", "-"),
                elapsed=row.get("sum_song_elapsed_sec", "-"),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _best_row(
    rows: list[dict[str, Any]], key: str, *, higher_is_better: bool
) -> dict[str, Any] | None:
    scored = [r for r in rows if isinstance(r.get(key), (int, float))]
    if not scored:
        return None
    return (
        max(scored, key=lambda r: float(r[key]))
        if higher_is_better
        else min(scored, key=lambda r: float(r[key]))
    )


def _recommendations(rows: list[dict[str, Any]]) -> dict[str, Any]:
    best_p95 = _best_row(
        rows, "agreement_start_p95_abs_sec_line_weighted_mean", higher_is_better=False
    )
    best_mean = _best_row(
        rows, "agreement_start_mean_abs_sec_line_weighted_mean", higher_is_better=False
    )
    best_low_conf = _best_row(
        rows, "low_confidence_ratio_line_weighted_mean", higher_is_better=False
    )
    best_dtw = _best_row(
        rows, "dtw_line_coverage_line_weighted_mean", higher_is_better=True
    )
    fastest = _best_row(rows, "sum_song_elapsed_sec", higher_is_better=False)

    quality_runtime_candidates = [
        r
        for r in rows
        if isinstance(
            r.get("agreement_start_p95_abs_sec_line_weighted_mean"), (int, float)
        )
        and isinstance(r.get("sum_song_elapsed_sec"), (int, float))
    ]
    quality_runtime_best: dict[str, Any] | None = None
    if quality_runtime_candidates:

        def _composite(row: dict[str, Any]) -> float:
            p95 = float(row["agreement_start_p95_abs_sec_line_weighted_mean"])
            elapsed = float(row["sum_song_elapsed_sec"])
            # Keep quality dominant while still penalizing materially slower runs.
            return p95 + 0.001 * elapsed

        quality_runtime_best = min(quality_runtime_candidates, key=_composite)

    return {
        "best_p95_start_abs_sec": best_p95.get("strategy") if best_p95 else None,
        "best_mean_start_abs_sec": best_mean.get("strategy") if best_mean else None,
        "lowest_low_confidence_ratio": (
            best_low_conf.get("strategy") if best_low_conf else None
        ),
        "highest_dtw_line_coverage": best_dtw.get("strategy") if best_dtw else None,
        "fastest_runtime": fastest.get("strategy") if fastest else None,
        "best_quality_runtime_balance": (
            quality_runtime_best.get("strategy") if quality_runtime_best else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strategies",
        default="hybrid_dtw,hybrid_whisper,whisper_only,lrc_only",
        help="Comma-separated strategy list",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "benchmark_songs.yaml",
    )
    parser.add_argument(
        "--gold-root", type=Path, default=REPO_ROOT / "benchmarks" / "gold_set"
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-songs", type=int, default=0)
    parser.add_argument("--match", default="")
    parser.add_argument(
        "--scenario",
        choices=["default", "lyrics_no_timing"],
        default="default",
        help="Benchmark scenario forwarded to run_benchmark_suite.py",
    )
    parser.add_argument(
        "--matrix-id",
        default=datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
        help="Matrix run id prefix",
    )
    args = parser.parse_args()

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    if not strategies:
        print("No strategies specified")
        return 1

    rows: list[dict[str, Any]] = []
    failures = 0

    for strategy in strategies:
        run_id = f"{args.matrix_id}-{strategy}"
        run_dir = args.output_root / run_id
        report_path = run_dir / "benchmark_report.json"
        cmd = _build_command(
            python_bin=args.python_bin,
            strategy=strategy,
            run_id=run_id,
            output_root=args.output_root,
            manifest=args.manifest,
            gold_root=args.gold_root,
            cache_dir=args.cache_dir,
            offline=args.offline,
            force=args.force,
            max_songs=args.max_songs,
            match=args.match,
            scenario=args.scenario,
        )
        print(f"[{strategy}] running: {' '.join(cmd)}")
        proc = subprocess.run(cmd)
        run = StrategyRun(
            strategy=strategy,
            run_id=run_id,
            run_dir=run_dir,
            exit_code=proc.returncode,
            report_path=report_path,
        )

        if run.exit_code != 0 or not run.report_path.exists():
            failures += 1
            rows.append(
                {
                    "strategy": run.strategy,
                    "status": "failed",
                    "songs_total": None,
                    "songs_succeeded": None,
                    "songs_failed": None,
                }
            )
            continue

        report_doc = json.loads(run.report_path.read_text(encoding="utf-8"))
        rows.append({"strategy": strategy, **_extract_summary(report_doc)})

    matrix_dir = args.output_root / f"{args.matrix_id}-matrix"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    out_json = matrix_dir / "strategy_matrix_report.json"
    out_md = matrix_dir / "strategy_matrix_report.md"
    out_json.write_text(
        json.dumps({"rows": rows, "recommendations": _recommendations(rows)}, indent=2),
        encoding="utf-8",
    )
    _write_markdown(out_md, rows)
    print(f"Wrote matrix reports:\n  {out_json}\n  {out_md}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
