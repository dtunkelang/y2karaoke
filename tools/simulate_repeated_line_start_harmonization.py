"""Simulate start-only harmonization across repeated lyric lines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools import analyze_repeated_line_drift as drift_tool


def _line_gold_map(gold: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {idx: line for idx, line in enumerate(gold.get("lines", []), start=1)}


def _estimate_error(value: float | None, gold: float | None) -> float | None:
    if value is None or gold is None:
        return None
    return abs(value - gold)


def _simulate(
    *,
    report: dict[str, Any],
    gold: dict[str, Any],
    min_start_error_span: float,
) -> dict[str, Any]:
    gold_map = _line_gold_map(gold)
    report_by_index = {int(line["index"]): line for line in report.get("lines", [])}
    groups = drift_tool._analyze(report=report, gold=gold)
    current_errors: list[float] = []
    simulated_errors: list[float] = []
    line_rows: list[dict[str, Any]] = []
    simulated_starts = {
        idx: float(line["start"]) for idx, line in report_by_index.items()
    }

    for group in groups:
        if float(group["start_error_span"]) < min_start_error_span:
            continue
        reference = min(
            group["occurrences"],
            key=lambda row: abs(float(row["start_error"])),
        )
        reference_shift = float(reference["start_error"])
        for row in group["occurrences"]:
            idx = int(row["index"])
            target_gold_line = gold_map[idx]
            simulated_starts[idx] = float(target_gold_line["start"]) + reference_shift

    for idx, report_line in sorted(report_by_index.items()):
        current_gold_line = gold_map.get(idx)
        if current_gold_line is None:
            continue
        gold_start = (
            float(current_gold_line["start"])
            if isinstance(current_gold_line.get("start"), (int, float))
            else None
        )
        current_start = float(report_line["start"])
        simulated_start = simulated_starts[idx]
        current_err = _estimate_error(current_start, gold_start)
        simulated_err = _estimate_error(simulated_start, gold_start)
        if current_err is not None:
            current_errors.append(current_err)
        if simulated_err is not None:
            simulated_errors.append(simulated_err)
        line_rows.append(
            {
                "index": idx,
                "text": str(report_line["text"]),
                "current_start": current_start,
                "simulated_start": simulated_start,
                "gold_start": gold_start,
                "current_start_error": current_err,
                "simulated_start_error": simulated_err,
            }
        )

    return {
        "current_start_mean": sum(current_errors) / max(1, len(current_errors)),
        "simulated_start_mean": sum(simulated_errors) / max(1, len(simulated_errors)),
        "lines": line_rows,
        "groups": groups,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("timing_report", help="Timing report JSON path")
    parser.add_argument("gold_json", help="Gold JSON path")
    parser.add_argument(
        "--min-start-error-span",
        type=float,
        default=0.3,
        help="Minimum repeated-line start error span to harmonize",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    args = parser.parse_args()

    report = drift_tool._load_json(Path(args.timing_report))
    gold = drift_tool._load_json(Path(args.gold_json))
    payload = _simulate(
        report=report,
        gold=gold,
        min_start_error_span=args.min_start_error_span,
    )

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(
        f"start mean: {payload['current_start_mean']:.3f} -> "
        f"{payload['simulated_start_mean']:.3f}"
    )
    for line in payload["lines"]:
        if line["current_start"] == line["simulated_start"]:
            continue
        print(
            f"line {line['index']}: {line['current_start_error']:.3f} -> "
            f"{line['simulated_start_error']:.3f} ({line['text']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
