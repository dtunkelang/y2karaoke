#!/usr/bin/env python3
"""Compare benchmark reports to measure human-correction impact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "benchmarks" / "results"


METRICS_AGGREGATE: list[dict[str, str]] = [
    {
        "key": "timing_quality_score_line_weighted_mean",
        "label": "timing_quality",
        "direction": "higher",
    },
    {
        "key": "agreement_coverage_ratio_mean",
        "label": "agreement_coverage",
        "direction": "higher",
    },
    {
        "key": "agreement_start_p95_abs_sec_mean",
        "label": "agreement_start_p95_abs_sec",
        "direction": "lower",
    },
    {
        "key": "agreement_bad_ratio_mean",
        "label": "agreement_bad_ratio",
        "direction": "lower",
    },
    {
        "key": "dtw_word_coverage_line_weighted_mean",
        "label": "dtw_word_coverage",
        "direction": "higher",
    },
    {
        "key": "avg_abs_word_start_delta_sec_word_weighted_mean",
        "label": "gold_start_abs_word",
        "direction": "lower",
    },
]

METRICS_CURATED_CANARY: list[dict[str, str]] = [
    {
        "key": "curated_canary_song_count",
        "label": "curated_canary_song_count",
        "direction": "higher",
    },
    {
        "key": "curated_canary_gold_word_coverage_ratio_total",
        "label": "curated_canary_gold_word_coverage",
        "direction": "higher",
    },
    {
        "key": "curated_canary_avg_abs_word_start_delta_sec_word_weighted_mean",
        "label": "curated_canary_gold_start_abs_word",
        "direction": "lower",
    },
    {
        "key": "curated_canary_gold_start_p95_abs_sec_mean",
        "label": "curated_canary_gold_start_p95_abs_sec",
        "direction": "lower",
    },
    {
        "key": "curated_canary_reference_watchlist_count",
        "label": "curated_canary_watchlist_count",
        "direction": "lower",
    },
]

METRICS_SONG: list[dict[str, str]] = [
    {"key": "timing_quality_score", "label": "timing_quality", "direction": "higher"},
    {
        "key": "agreement_coverage_ratio",
        "label": "agreement_coverage",
        "direction": "higher",
    },
    {
        "key": "agreement_start_p95_abs_sec",
        "label": "agreement_start_p95_abs_sec",
        "direction": "lower",
    },
    {
        "key": "agreement_bad_ratio",
        "label": "agreement_bad_ratio",
        "direction": "lower",
    },
    {"key": "gold_start_mean_abs_sec", "label": "gold_start_abs", "direction": "lower"},
    {"key": "dtw_word_coverage", "label": "dtw_word_coverage", "direction": "higher"},
]


def _num(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _song_key(song: dict[str, Any]) -> str:
    artist = str(song.get("artist", "")).strip().lower()
    title = str(song.get("title", "")).strip().lower()
    return f"{artist}::{title}"


def _load_report(path: Path) -> dict[str, Any]:
    if path.is_dir():
        path = path / "benchmark_report.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_delta(
    base_value: float | None,
    corrected_value: float | None,
    direction: str,
) -> dict[str, Any]:
    if base_value is None or corrected_value is None:
        return {
            "baseline": base_value,
            "corrected": corrected_value,
            "delta": None,
            "improved": None,
        }

    delta = corrected_value - base_value
    if abs(delta) < 1e-12:
        improved: bool | None = None
    else:
        improved = delta > 0 if direction == "higher" else delta < 0
    return {
        "baseline": round(base_value, 6),
        "corrected": round(corrected_value, 6),
        "delta": round(delta, 6),
        "improved": improved,
    }


def _compare_aggregate(
    baseline_aggregate: dict[str, Any],
    corrected_aggregate: dict[str, Any],
    metrics: list[dict[str, str]],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for metric in metrics:
        key = metric["key"]
        out[key] = _metric_delta(
            _num(baseline_aggregate.get(key)),
            _num(corrected_aggregate.get(key)),
            metric["direction"],
        )
    return out


def _compare_songs(
    baseline_songs: list[dict[str, Any]],
    corrected_songs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    baseline_by_key = {_song_key(song): song for song in baseline_songs}
    corrected_by_key = {_song_key(song): song for song in corrected_songs}
    all_keys = sorted(set(baseline_by_key) & set(corrected_by_key))
    rows: list[dict[str, Any]] = []
    for key in all_keys:
        baseline_song = baseline_by_key[key]
        corrected_song = corrected_by_key[key]
        baseline_metrics = baseline_song.get("metrics", {}) or {}
        corrected_metrics = corrected_song.get("metrics", {}) or {}
        metric_deltas: dict[str, dict[str, Any]] = {}
        improved_count = 0
        regressed_count = 0
        for metric in METRICS_SONG:
            metric_key = metric["key"]
            delta_info = _metric_delta(
                _num(baseline_metrics.get(metric_key)),
                _num(corrected_metrics.get(metric_key)),
                metric["direction"],
            )
            metric_deltas[metric_key] = delta_info
            if delta_info["improved"] is True:
                improved_count += 1
            elif delta_info["improved"] is False:
                regressed_count += 1
        rows.append(
            {
                "song_key": key,
                "artist": baseline_song.get("artist"),
                "title": baseline_song.get("title"),
                "improved_metrics": improved_count,
                "regressed_metrics": regressed_count,
                "net_score": improved_count - regressed_count,
                "metrics": metric_deltas,
            }
        )
    return rows


def _summarize_song_comparison(song_rows: list[dict[str, Any]]) -> dict[str, Any]:
    improved = sum(1 for row in song_rows if row["net_score"] > 0)
    regressed = sum(1 for row in song_rows if row["net_score"] < 0)
    unchanged = sum(1 for row in song_rows if row["net_score"] == 0)
    return {
        "songs_compared": len(song_rows),
        "songs_net_improved": improved,
        "songs_net_regressed": regressed,
        "songs_net_unchanged": unchanged,
    }


def _write_markdown(path: Path, report: dict[str, Any], top_n: int) -> None:
    lines: list[str] = []
    lines.append("# Human Correction Delta Report")
    lines.append("")
    lines.append(f"- Baseline: `{report['baseline']}`")
    lines.append(f"- Corrected: `{report['corrected']}`")
    lines.append(f"- Songs compared: `{report['summary']['songs_compared']}`")
    lines.append(
        "- Net song outcomes: "
        f"improved=`{report['summary']['songs_net_improved']}`, "
        f"regressed=`{report['summary']['songs_net_regressed']}`, "
        f"unchanged=`{report['summary']['songs_net_unchanged']}`"
    )
    lines.append("")
    lines.append("## Aggregate Deltas")
    lines.append("")
    lines.append("| Metric | Baseline | Corrected | Delta | Improved |")
    lines.append("|---|---:|---:|---:|---|")
    for metric in METRICS_AGGREGATE:
        key = metric["key"]
        data = report["aggregate_deltas"][key]
        lines.append(
            "| {metric} | {base} | {corr} | {delta} | {improved} |".format(
                metric=metric["label"],
                base=_fmt_num(data["baseline"]),
                corr=_fmt_num(data["corrected"]),
                delta=_fmt_num(data["delta"], signed=True),
                improved=_fmt_improved(data["improved"]),
            )
        )
    curated = report.get("curated_canary_deltas", {})
    if curated:
        lines.append("")
        lines.append("## Curated Canary Deltas")
        lines.append("")
        lines.append("| Metric | Baseline | Corrected | Delta | Improved |")
        lines.append("|---|---:|---:|---:|---|")
        for metric in METRICS_CURATED_CANARY:
            key = metric["key"]
            data = curated.get(key)
            if not isinstance(data, dict):
                continue
            lines.append(
                "| {metric} | {base} | {corr} | {delta} | {improved} |".format(
                    metric=metric["label"],
                    base=_fmt_num(data["baseline"]),
                    corr=_fmt_num(data["corrected"]),
                    delta=_fmt_num(data["delta"], signed=True),
                    improved=_fmt_improved(data["improved"]),
                )
            )
        watchlist = report.get("curated_canary_watchlist", {})
        if isinstance(watchlist, dict):
            baseline_watch = watchlist.get("baseline")
            corrected_watch = watchlist.get("corrected")
            if isinstance(baseline_watch, list) or isinstance(corrected_watch, list):
                lines.append("")
                lines.append(
                    "- Curated canary watchlist baseline: "
                    + (
                        ", ".join(f"`{item}`" for item in baseline_watch)
                        if isinstance(baseline_watch, list) and baseline_watch
                        else "`-`"
                    )
                )
                lines.append(
                    "- Curated canary watchlist corrected: "
                    + (
                        ", ".join(f"`{item}`" for item in corrected_watch)
                        if isinstance(corrected_watch, list) and corrected_watch
                        else "`-`"
                    )
                )

    sorted_rows = sorted(
        report["song_deltas"],
        key=lambda row: (
            row["net_score"],
            row["improved_metrics"],
            -row["regressed_metrics"],
        ),
        reverse=True,
    )
    top_rows = sorted_rows[:top_n]
    bottom_rows = list(reversed(sorted_rows[-top_n:]))

    def write_song_table(title: str, rows: list[dict[str, Any]]) -> None:
        lines.append("")
        lines.append(f"## {title}")
        lines.append("")
        lines.append(
            "| Song | Net | Improved | Regressed | timing_quality_delta | agreement_cov_delta | p95_delta_sec |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for row in rows:
            m = row["metrics"]
            lines.append(
                "| {song} | {net} | {imp} | {reg} | {tq} | {cov} | {p95} |".format(
                    song=f"{row['artist']} - {row['title']}",
                    net=row["net_score"],
                    imp=row["improved_metrics"],
                    reg=row["regressed_metrics"],
                    tq=_fmt_num(m["timing_quality_score"]["delta"], signed=True),
                    cov=_fmt_num(m["agreement_coverage_ratio"]["delta"], signed=True),
                    p95=_fmt_num(
                        m["agreement_start_p95_abs_sec"]["delta"], signed=True
                    ),
                )
            )

    write_song_table("Top Net Improvements", top_rows)
    write_song_table("Top Net Regressions", bottom_rows)

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt_num(value: float | None, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    if signed:
        return f"{value:+.4f}"
    return f"{value:.4f}"


def _fmt_improved(value: bool | None) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "n/a"


def _build_comparison(
    baseline_report: dict[str, Any],
    corrected_report: dict[str, Any],
    baseline_label: str,
    corrected_label: str,
) -> dict[str, Any]:
    song_deltas = _compare_songs(
        baseline_report.get("songs", []) or [],
        corrected_report.get("songs", []) or [],
    )
    report = {
        "baseline": baseline_label,
        "corrected": corrected_label,
        "aggregate_deltas": _compare_aggregate(
            baseline_report.get("aggregate", {}) or {},
            corrected_report.get("aggregate", {}) or {},
            METRICS_AGGREGATE,
        ),
        "curated_canary_deltas": _compare_aggregate(
            baseline_report.get("aggregate", {}) or {},
            corrected_report.get("aggregate", {}) or {},
            METRICS_CURATED_CANARY,
        ),
        "curated_canary_watchlist": {
            "baseline": list(
                (baseline_report.get("aggregate", {}) or {}).get(
                    "curated_canary_reference_watchlist", []
                )
                or []
            ),
            "corrected": list(
                (corrected_report.get("aggregate", {}) or {}).get(
                    "curated_canary_reference_watchlist", []
                )
                or []
            ),
        },
        "song_deltas": song_deltas,
        "summary": _summarize_song_comparison(song_deltas),
    }
    return report


def _curated_canary_cli_summary(report: dict[str, Any]) -> list[str]:
    curated = report.get("curated_canary_deltas", {})
    if not isinstance(curated, dict) or not curated:
        return []
    start_abs = curated.get(
        "curated_canary_avg_abs_word_start_delta_sec_word_weighted_mean", {}
    )
    start_p95 = curated.get("curated_canary_gold_start_p95_abs_sec_mean", {})
    watchlist = curated.get("curated_canary_reference_watchlist_count", {})
    baseline_watch = (report.get("curated_canary_watchlist", {}) or {}).get(
        "baseline", []
    )
    corrected_watch = (report.get("curated_canary_watchlist", {}) or {}).get(
        "corrected", []
    )
    lines = ["  curated_canary:"]
    lines.append(
        "    gold_start_abs_word={base} -> {corr}".format(
            base=_fmt_num(start_abs.get("baseline")),
            corr=_fmt_num(start_abs.get("corrected")),
        )
    )
    lines.append(
        "    gold_start_p95_abs_sec={base} -> {corr}".format(
            base=_fmt_num(start_p95.get("baseline")),
            corr=_fmt_num(start_p95.get("corrected")),
        )
    )
    lines.append(
        "    watchlist_count={base} -> {corr}".format(
            base=_fmt_num(watchlist.get("baseline")),
            corr=_fmt_num(watchlist.get("corrected")),
        )
    )
    if isinstance(baseline_watch, list) or isinstance(corrected_watch, list):
        lines.append(
            "    watchlist_baseline="
            + (
                ", ".join(str(item) for item in baseline_watch)
                if isinstance(baseline_watch, list) and baseline_watch
                else "-"
            )
        )
        lines.append(
            "    watchlist_corrected="
            + (
                ", ".join(str(item) for item in corrected_watch)
                if isinstance(corrected_watch, list) and corrected_watch
                else "-"
            )
        )
    return lines


def _resolve_label(path: Path) -> str:
    if path.is_dir():
        return path.name
    return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare benchmark report deltas between baseline auto-alignment and "
            "human-corrected runs."
        )
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="Baseline benchmark run dir (or benchmark_report.json path)",
    )
    parser.add_argument(
        "--corrected",
        type=Path,
        required=True,
        help="Corrected benchmark run dir (or benchmark_report.json path)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional output JSON path (default: <corrected>/human_correction_delta.json)",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        help="Optional output markdown path (default: <corrected>/human_correction_delta.md)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of songs shown in top improvement/regression tables",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print JSON report to stdout",
    )
    parser.add_argument(
        "--assert-agreement-tradeoff",
        action="store_true",
        help=(
            "Fail with non-zero exit when agreement coverage gain is offset by "
            "material agreement-bad-ratio increase."
        ),
    )
    parser.add_argument(
        "--min-coverage-gain",
        type=float,
        default=0.005,
        help=(
            "Coverage-gain threshold that triggers tradeoff assertion "
            "(default: 0.005)"
        ),
    )
    parser.add_argument(
        "--max-bad-ratio-increase",
        type=float,
        default=0.002,
        help=(
            "Maximum allowed increase in agreement_bad_ratio_mean when "
            "coverage gain threshold is met (default: 0.002)"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    baseline_report = _load_report(args.baseline.expanduser().resolve())
    corrected_report = _load_report(args.corrected.expanduser().resolve())

    baseline_label = _resolve_label(args.baseline)
    corrected_label = _resolve_label(args.corrected)
    comparison = _build_comparison(
        baseline_report=baseline_report,
        corrected_report=corrected_report,
        baseline_label=baseline_label,
        corrected_label=corrected_label,
    )

    corrected_path = args.corrected.expanduser().resolve()
    default_output_root = (
        corrected_path if corrected_path.is_dir() else corrected_path.parent
    )
    output_json = (
        args.output_json or default_output_root / "human_correction_delta.json"
    )
    output_md = args.output_md or default_output_root / "human_correction_delta.md"

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    _write_markdown(output_md, comparison, top_n=max(1, int(args.top_n)))

    tradeoff_issue: str | None = None
    if args.assert_agreement_tradeoff:
        agree_cov = comparison["aggregate_deltas"].get(
            "agreement_coverage_ratio_mean", {}
        )
        agree_bad = comparison["aggregate_deltas"].get("agreement_bad_ratio_mean", {})
        cov_delta = _num(agree_cov.get("delta"))
        bad_delta = _num(agree_bad.get("delta"))
        if cov_delta is not None and bad_delta is not None:
            if cov_delta >= float(args.min_coverage_gain) and bad_delta > float(
                args.max_bad_ratio_increase
            ):
                tradeoff_issue = (
                    "agreement tradeoff assertion failed: "
                    f"coverage delta {cov_delta:+.4f} >= {float(args.min_coverage_gain):.4f} "
                    "while "
                    f"bad-ratio delta {bad_delta:+.4f} > {float(args.max_bad_ratio_increase):.4f}"
                )

    if args.print_json:
        print(json.dumps(comparison, indent=2))
        if tradeoff_issue:
            print(tradeoff_issue)
            return 1
    else:
        print("human_correction_delta: OK")
        print(f"  baseline={baseline_label}")
        print(f"  corrected={corrected_label}")
        print(f"  songs_compared={comparison['summary']['songs_compared']}")
        for line in _curated_canary_cli_summary(comparison):
            print(line)
        print(f"  output_json={output_json}")
        print(f"  output_md={output_md}")
        if tradeoff_issue:
            print(f"  assertion={tradeoff_issue}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
