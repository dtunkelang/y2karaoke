"""Simulate adjacent shared-boundary opportunity repairs offline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools import analyze_shared_boundary_opportunities as boundary_tool


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def _error(current: float, gold: float) -> float:
    return abs(current - gold)


def simulate(report: dict[str, Any], gold: dict[str, Any]) -> dict[str, Any]:
    payload = boundary_tool.analyze(report, gold)
    line_rows = report.get("lines", [])
    gold_rows = gold.get("lines", [])
    line_by_index = {int(line["index"]): line for line in line_rows}
    gold_by_index = {
        int(line["index"]): gold_row for line, gold_row in zip(line_rows, gold_rows)
    }
    current_starts: dict[int, float] = {
        int(line["index"]): float(line["start"]) for line in line_rows
    }
    current_ends: dict[int, float] = {
        int(line["index"]): float(line["end"]) for line in line_rows
    }
    simulated_starts = dict(current_starts)
    simulated_ends = dict(current_ends)
    opportunities: list[dict[str, Any]] = []

    for row in payload["rows"]:
        prev_index = int(row["prev_index"])
        next_index = int(row["next_index"])
        current_prev_end = simulated_ends[prev_index]
        current_next_start = simulated_starts[next_index]
        suggested_prev_end = row["suggested_prev_end"]
        suggested_next_start = float(row["suggested_next_start"])
        if suggested_prev_end is not None:
            simulated_ends[prev_index] = float(suggested_prev_end)
        simulated_starts[next_index] = suggested_next_start
        opportunities.append(
            {
                "family": str(row["family"]),
                "prev_index": prev_index,
                "next_index": next_index,
                "prev_end": {
                    "current": current_prev_end,
                    "simulated": simulated_ends[prev_index],
                    "gold": float(gold_by_index[prev_index]["end"]),
                },
                "next_start": {
                    "current": current_next_start,
                    "simulated": simulated_starts[next_index],
                    "gold": float(gold_by_index[next_index]["start"]),
                },
            }
        )

    current_start_errors = [
        _error(current_starts[index], float(gold_by_index[index]["start"]))
        for index in current_starts
    ]
    simulated_start_errors = [
        _error(simulated_starts[index], float(gold_by_index[index]["start"]))
        for index in simulated_starts
    ]
    current_end_errors = [
        _error(current_ends[index], float(gold_by_index[index]["end"]))
        for index in current_ends
    ]
    simulated_end_errors = [
        _error(simulated_ends[index], float(gold_by_index[index]["end"]))
        for index in simulated_ends
    ]

    lines: list[dict[str, Any]] = []
    for index in sorted(line_by_index):
        line = line_by_index[index]
        gold_line = gold_by_index[index]
        lines.append(
            {
                "index": index,
                "text": str(line["text"]),
                "current_start_error": _error(
                    current_starts[index], float(gold_line["start"])
                ),
                "simulated_start_error": _error(
                    simulated_starts[index], float(gold_line["start"])
                ),
                "current_end_error": _error(
                    current_ends[index], float(gold_line["end"])
                ),
                "simulated_end_error": _error(
                    simulated_ends[index], float(gold_line["end"])
                ),
            }
        )

    return {
        "artist": report.get("artist"),
        "title": report.get("title"),
        "current_start_mean": _mean(current_start_errors),
        "simulated_start_mean": _mean(simulated_start_errors),
        "current_end_mean": _mean(current_end_errors),
        "simulated_end_mean": _mean(simulated_end_errors),
        "opportunities": opportunities,
        "lines": lines,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("timing_report", help="Timing report JSON path")
    parser.add_argument("gold_json", help="Gold timing JSON path")
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    args = parser.parse_args()

    payload = simulate(
        _load_json(Path(args.timing_report)),
        _load_json(Path(args.gold_json)),
    )
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    label = " - ".join(
        part for part in [payload.get("artist"), payload.get("title")] if part
    )
    print(label)
    print(
        f"start mean: {payload['current_start_mean']:.3f} -> "
        f"{payload['simulated_start_mean']:.3f}"
    )
    print(
        f"end mean: {payload['current_end_mean']:.3f} -> "
        f"{payload['simulated_end_mean']:.3f}"
    )
    for row in payload["opportunities"]:
        print(
            f"lines {row['prev_index']}/{row['next_index']} {row['family']}: "
            "prev end "
            f"{row['prev_end']['current']:.3f}->{row['prev_end']['simulated']:.3f}, "
            "next start "
            f"{row['next_start']['current']:.3f}->{row['next_start']['simulated']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
