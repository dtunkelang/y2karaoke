#!/usr/bin/env python3
"""Summarize which alignment trace stages materially move line timings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _snapshot_count_key(snapshot: dict[str, Any]) -> str:
    return "count" if "count" in snapshot else "line_count"


def _line_map(snapshot: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(line["line_index"]): line for line in snapshot.get("lines", [])}


def analyze_stage_effects(
    payload: dict[str, Any],
    *,
    min_delta_sec: float = 0.05,
) -> dict[str, Any]:
    snapshots = payload.get("snapshots", [])
    if not snapshots:
        return {"snapshots": 0, "lines": []}

    first_snapshot = snapshots[0]
    first_lines = _line_map(first_snapshot)
    latest_by_line: dict[int, dict[str, Any]] = {
        idx: {
            "line_index": idx,
            "text": str(line.get("text", "")),
            "initial_stage": str(first_snapshot["stage"]),
            "initial_start": float(line["start"]),
            "initial_end": float(line["end"]),
            "final_stage": str(first_snapshot["stage"]),
            "final_start": float(line["start"]),
            "final_end": float(line["end"]),
            "first_changed_stage": None,
            "max_start_delta_stage": None,
            "max_start_delta": 0.0,
            "max_end_delta_stage": None,
            "max_end_delta": 0.0,
            "stage_changes": [],
        }
        for idx, line in first_lines.items()
    }

    previous_lines = first_lines
    for snapshot in snapshots[1:]:
        current_lines = _line_map(snapshot)
        stage_name = str(snapshot["stage"])
        for idx, current in current_lines.items():
            if idx not in latest_by_line or idx not in previous_lines:
                continue
            previous = previous_lines[idx]
            start_delta = float(current["start"]) - float(previous["start"])
            end_delta = float(current["end"]) - float(previous["end"])
            if abs(start_delta) < min_delta_sec and abs(end_delta) < min_delta_sec:
                continue
            line_state = latest_by_line[idx]
            if line_state["first_changed_stage"] is None:
                line_state["first_changed_stage"] = stage_name
            if abs(start_delta) > abs(line_state["max_start_delta"]):
                line_state["max_start_delta"] = start_delta
                line_state["max_start_delta_stage"] = stage_name
            if abs(end_delta) > abs(line_state["max_end_delta"]):
                line_state["max_end_delta"] = end_delta
                line_state["max_end_delta_stage"] = stage_name
            line_state["final_stage"] = stage_name
            line_state["final_start"] = float(current["start"])
            line_state["final_end"] = float(current["end"])
            line_state["stage_changes"].append(
                {
                    "stage": stage_name,
                    "start_delta": round(start_delta, 3),
                    "end_delta": round(end_delta, 3),
                    "start": round(float(current["start"]), 3),
                    "end": round(float(current["end"]), 3),
                }
            )
        previous_lines = current_lines

    rows = []
    for row in latest_by_line.values():
        total_start_shift = row["final_start"] - row["initial_start"]
        total_end_shift = row["final_end"] - row["initial_end"]
        row["total_start_shift"] = round(total_start_shift, 3)
        row["total_end_shift"] = round(total_end_shift, 3)
        row["abs_total_shift"] = round(
            max(abs(total_start_shift), abs(total_end_shift)),
            3,
        )
        rows.append(row)

    rows.sort(
        key=lambda row: (
            -row["abs_total_shift"],
            -abs(row["max_start_delta"]),
            -abs(row["max_end_delta"]),
            row["line_index"],
        )
    )
    return {
        "metadata": payload.get("metadata", {}),
        "snapshots": len(snapshots),
        "initial_stage": first_snapshot["stage"],
        "final_stage": snapshots[-1]["stage"],
        "line_count": first_snapshot.get(_snapshot_count_key(first_snapshot), 0),
        "lines": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_json", help="Trace JSON path")
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    parser.add_argument(
        "--min-delta-sec",
        type=float,
        default=0.05,
        help="Ignore smaller per-stage moves",
    )
    args = parser.parse_args()

    payload = analyze_stage_effects(
        _load_json(Path(args.trace_json)),
        min_delta_sec=args.min_delta_sec,
    )
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(
        f"{Path(args.trace_json).name}: {payload['snapshots']} snapshots, "
        f"{payload['initial_stage']} -> {payload['final_stage']}"
    )
    for row in payload["lines"]:
        if not row["stage_changes"]:
            continue
        print(
            f"line {row['line_index']} {row['text']}: "
            f"start {row['initial_start']:.3f}->{row['final_start']:.3f} "
            f"({row['total_start_shift']:+.3f}), "
            f"end {row['initial_end']:.3f}->{row['final_end']:.3f} "
            f"({row['total_end_shift']:+.3f})"
        )
        print(
            f"  first change: {row['first_changed_stage']}; "
            f"largest start delta: {row['max_start_delta_stage']} "
            f"({row['max_start_delta']:+.3f}); "
            f"largest end delta: {row['max_end_delta_stage']} "
            f"({row['max_end_delta']:+.3f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
