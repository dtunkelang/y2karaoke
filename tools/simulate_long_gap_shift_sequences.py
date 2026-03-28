#!/usr/bin/env python3
"""Simulate sequence-level candidate selection for long-gap shift rows."""

from __future__ import annotations

import argparse
import itertools
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


def _baseline_duration(
    baseline_timing: dict[str, Any] | None, line_index: int
) -> float | None:
    if baseline_timing is None:
        return None
    lines = baseline_timing.get("lines", [])
    if 0 < line_index <= len(lines):
        return float(lines[line_index - 1]["end"]) - float(
            lines[line_index - 1]["start"]
        )
    return None


def _collect_shift_rows(trace: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in trace.get("rows", []) if row.get("decision") == "shift"]


def analyze(
    *,
    trace_path: Path,
    gold_path: Path,
    baseline_timing_path: Path | None = None,
    min_gap: float = 0.05,
    max_results: int = 10,
) -> dict[str, Any]:
    trace = _load_json(trace_path)
    gold = _load_json(gold_path)
    baseline_timing = _load_json(baseline_timing_path) if baseline_timing_path else None
    shift_rows = _collect_shift_rows(trace)
    if len(shift_rows) < 2:
        return {"rows": []}

    analyzed_rows: list[dict[str, Any]] = []
    for first, second in zip(shift_rows, shift_rows[1:]):
        first_index = int(first.get("line_index") or 0)
        second_index = int(second.get("line_index") or 0)
        if second_index != first_index + 1:
            continue
        first_gold = _find_gold_line(gold, first_index)
        second_gold = _find_gold_line(gold, second_index)
        third_gold = _find_gold_line(gold, second_index + 1)
        if first_gold is None or second_gold is None:
            continue

        first_current_duration = float(first["end"]) - float(first["start"])
        second_current_duration = float(second["end"]) - float(second["start"])
        first_baseline_duration = _baseline_duration(baseline_timing, first_index)
        second_baseline_duration = _baseline_duration(baseline_timing, second_index)
        first_candidates = [float(onset) for onset in first.get("candidate_onsets", [])]
        second_candidates = [
            float(onset) for onset in second.get("candidate_onsets", [])
        ]

        pair_rows: list[dict[str, Any]] = []
        for first_onset, second_onset in itertools.product(
            first_candidates, second_candidates
        ):
            first_end = first_onset + first_current_duration
            second_end = second_onset + second_current_duration
            valid_with_current = first_end <= second_onset - min_gap
            valid_against_next = (
                third_gold is None or second_end <= float(third_gold["start"]) - min_gap
            )
            row: dict[str, Any] = {
                "first_onset": round(first_onset, 3),
                "first_end_current": round(first_end, 3),
                "second_onset": round(second_onset, 3),
                "second_end_current": round(second_end, 3),
                "valid_current_pair": bool(valid_with_current and valid_against_next),
                "current_pair_score": round(
                    abs(first_onset - float(first_gold["start"]))
                    + abs(first_end - float(first_gold["end"]))
                    + abs(second_onset - float(second_gold["start"]))
                    + abs(second_end - float(second_gold["end"])),
                    3,
                ),
            }
            if first_baseline_duration is not None:
                first_end_baseline = first_onset + first_baseline_duration
                row["first_end_baseline"] = round(first_end_baseline, 3)
            if second_baseline_duration is not None:
                second_end_baseline = second_onset + second_baseline_duration
                row["second_end_baseline"] = round(second_end_baseline, 3)
            pair_rows.append(row)

        pair_rows.sort(
            key=lambda row: (
                not row["valid_current_pair"],
                float(row["current_pair_score"]),
                abs(float(row["first_onset"]) - float(first_gold["start"])),
                abs(float(row["second_onset"]) - float(second_gold["start"])),
            )
        )
        analyzed_rows.append(
            {
                "first_line_index": first_index,
                "first_text": first.get("text"),
                "second_line_index": second_index,
                "second_text": second.get("text"),
                "first_gold_start": round(float(first_gold["start"]), 3),
                "first_gold_end": round(float(first_gold["end"]), 3),
                "second_gold_start": round(float(second_gold["start"]), 3),
                "second_gold_end": round(float(second_gold["end"]), 3),
                "first_current_duration": round(first_current_duration, 3),
                "second_current_duration": round(second_current_duration, 3),
                "best_pairs": pair_rows[:max_results],
            }
        )

    return {"rows": analyzed_rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--baseline-timing", type=Path)
    parser.add_argument("--min-gap", type=float, default=0.05)
    parser.add_argument("--max-results", type=int, default=10)
    args = parser.parse_args()
    print(
        json.dumps(
            analyze(
                trace_path=args.trace,
                gold_path=args.gold,
                baseline_timing_path=args.baseline_timing,
                min_gap=args.min_gap,
                max_results=args.max_results,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
