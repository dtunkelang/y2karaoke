#!/usr/bin/env python3
"""Simulate segment-pull retiming from local subphrase windows in merged segments."""

from __future__ import annotations

import argparse
import json
import importlib.util
from pathlib import Path
from typing import Any


def _load_subphrase_analyzer():
    module_path = (
        Path(__file__).resolve().parent / "analyze_segment_subphrase_windows.py"
    )
    spec = importlib.util.spec_from_file_location(
        "analyze_segment_subphrase_windows_module", module_path
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def analyze(
    *, timing_path: Path, segments_path: Path, min_gap: float = 0.05
) -> dict[str, Any]:
    timing = _load_json(timing_path)
    subphrase = _load_subphrase_analyzer().analyze(
        timing_path=timing_path,
        segments_path=segments_path,
    )
    baseline_rows = {int(row["line_index"]): row for row in timing.get("lines", [])}
    phrase_rows = {int(row["line_index"]): row for row in subphrase.get("rows", [])}

    simulated_rows: list[dict[str, Any]] = []
    previous_end: float | None = None
    next_starts = {
        int(row.get("line_index", 0) or 0): float(row["start"])
        for row in timing.get("lines", [])
    }

    for line_index in sorted(phrase_rows):
        line = baseline_rows[line_index]
        phrase = phrase_rows[line_index].get("phrase_window")
        if phrase is None:
            continue
        current_start = float(line["start"])
        current_end = float(line["end"])
        duration = current_end - current_start
        target_start = float(phrase["window_start"])
        if previous_end is not None:
            target_start = max(target_start, previous_end + min_gap)
        target_end = target_start + duration
        next_start = next_starts.get(line_index + 1)
        if next_start is not None:
            target_end = min(target_end, next_start - min_gap)
        simulated_rows.append(
            {
                "line_index": line_index,
                "text": line["text"],
                "current_start": round(current_start, 3),
                "current_end": round(current_end, 3),
                "phrase_window_start": round(float(phrase["window_start"]), 3),
                "phrase_window_end": round(float(phrase["window_end"]), 3),
                "simulated_start": round(target_start, 3),
                "simulated_end": round(target_end, 3),
            }
        )
        previous_end = target_end

    return {"rows": simulated_rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing", type=Path, required=True)
    parser.add_argument("--segments", type=Path, required=True)
    parser.add_argument("--min-gap", type=float, default=0.05)
    args = parser.parse_args()
    print(
        json.dumps(
            analyze(
                timing_path=args.timing,
                segments_path=args.segments,
                min_gap=args.min_gap,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
