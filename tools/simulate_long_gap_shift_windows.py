#!/usr/bin/env python3
"""Simulate paired start/end windows for long-gap shift candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_gold_line(gold: dict[str, Any], line_index: int) -> dict[str, Any] | None:
    for line in gold.get("lines", []):
        if int(line.get("line_index", 0) or 0) == line_index:
            return line
    return None


def _build_candidate_windows(
    *,
    candidate_onsets: list[float],
    current_duration: float,
    baseline_duration: float | None,
    next_gold_start: float | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for onset in candidate_onsets:
        current_end = onset + current_duration
        row = {
            "candidate_onset": round(onset, 3),
            "current_duration_end": round(current_end, 3),
        }
        if baseline_duration is not None:
            baseline_end = onset + baseline_duration
            row["baseline_duration_end"] = round(baseline_end, 3)
        if next_gold_start is not None:
            row["overlaps_next_gold_with_current_duration"] = (
                current_end > next_gold_start
            )
            if baseline_duration is not None:
                row["overlaps_next_gold_with_baseline_duration"] = (
                    baseline_end > next_gold_start
                )
        rows.append(row)
    return rows


def analyze(
    *,
    trace_path: Path,
    gold_path: Path,
    baseline_timing_path: Path | None = None,
) -> dict[str, Any]:
    trace = _load_json(trace_path)
    gold = _load_json(gold_path)
    baseline_timing = _load_json(baseline_timing_path) if baseline_timing_path else None
    baseline_lines = baseline_timing.get("lines", []) if baseline_timing else []
    analyzed_rows: list[dict[str, Any]] = []

    for row in trace.get("rows", []):
        if row.get("decision") != "shift":
            continue
        line_index = int(row.get("line_index") or 0)
        gold_line = _find_gold_line(gold, line_index)
        if gold_line is None:
            continue
        next_gold_line = _find_gold_line(gold, line_index + 1)
        baseline_duration = None
        if 0 < line_index <= len(baseline_lines):
            baseline_duration = float(baseline_lines[line_index - 1]["end"]) - float(
                baseline_lines[line_index - 1]["start"]
            )
        candidate_onsets = [float(onset) for onset in row.get("candidate_onsets", [])]
        if not candidate_onsets:
            continue
        current_duration = float(row["end"]) - float(row["start"])
        analyzed_rows.append(
            {
                "call_index": int(row.get("call_index") or 0),
                "line_index": line_index,
                "text": row.get("text"),
                "gold_start": round(float(gold_line["start"]), 3),
                "gold_end": round(float(gold_line["end"]), 3),
                "current_start": round(float(row["start"]), 3),
                "current_end": round(float(row["end"]), 3),
                "chosen_onset": round(float(row["chosen_onset"]), 3),
                "current_duration": round(current_duration, 3),
                "baseline_duration": (
                    round(baseline_duration, 3)
                    if baseline_duration is not None
                    else None
                ),
                "candidate_windows": _build_candidate_windows(
                    candidate_onsets=candidate_onsets,
                    current_duration=current_duration,
                    baseline_duration=baseline_duration,
                    next_gold_start=(
                        float(next_gold_line["start"])
                        if next_gold_line is not None
                        else None
                    ),
                ),
            }
        )

    return {"rows": analyzed_rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--baseline-timing", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            analyze(
                trace_path=args.trace,
                gold_path=args.gold,
                baseline_timing_path=args.baseline_timing,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
