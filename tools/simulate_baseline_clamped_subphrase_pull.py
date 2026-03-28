#!/usr/bin/env python3
"""Simulate selective subphrase pull with a baseline-clamped start."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_selective_pull():
    module_path = Path(__file__).resolve().parent / "simulate_segment_subphrase_pull.py"
    spec = importlib.util.spec_from_file_location(
        "simulate_segment_subphrase_pull_module", module_path
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _normalize_tokens(line: dict[str, Any]) -> list[str]:
    return [
        "".join(ch for ch in word.get("text", "").lower() if ch.isalpha())
        for word in line.get("words", [])
        if "".join(ch for ch in word.get("text", "").lower() if ch.isalpha())
    ]


def _is_alternating_pair(
    prev_tokens: list[str], current_tokens: list[str], next_tokens: list[str]
) -> bool:
    return (
        len(prev_tokens) == len(current_tokens) == 3
        and prev_tokens[0] == current_tokens[0]
        and prev_tokens[1] == current_tokens[2]
        and prev_tokens[2] == current_tokens[1]
        and prev_tokens[1] != prev_tokens[2]
        and not any(token in prev_tokens for token in next_tokens)
    )


def analyze(
    *,
    timing_path: Path,
    segments_path: Path,
    baseline_timing_path: Path,
    max_early_pull_sec: float = 0.4,
    min_gap: float = 0.05,
) -> dict[str, Any]:
    selective = _load_selective_pull().analyze(
        timing_path=timing_path,
        segments_path=segments_path,
        min_gap=min_gap,
    )
    baseline = _load_json(baseline_timing_path)
    baseline_rows = {
        int(row.get("line_index", 0) or 0): row for row in baseline.get("lines", [])
    }
    rows: list[dict[str, Any]] = []

    for row in selective.get("rows", []):
        line_index = int(row["line_index"])
        baseline_row = baseline_rows.get(line_index)
        if baseline_row is None:
            continue
        prev_row = baseline_rows.get(line_index - 1)
        next_row = baseline_rows.get(line_index + 1)
        if prev_row is None or next_row is None:
            continue
        if not _is_alternating_pair(
            _normalize_tokens(prev_row),
            _normalize_tokens(baseline_row),
            _normalize_tokens(next_row),
        ):
            continue
        baseline_start = float(baseline_row["start"])
        clamped_start = max(
            float(row["simulated_start"]), baseline_start - max_early_pull_sec
        )
        phrase_window_end = float(row["phrase_window_end"])
        duration = float(row["current_end"]) - float(row["current_start"])
        target_end = max(clamped_start + duration, phrase_window_end)
        rows.append(
            {
                **row,
                "baseline_start": round(baseline_start, 3),
                "clamped_start": round(clamped_start, 3),
                "clamped_end": round(target_end, 3),
            }
        )

    return {"rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing", type=Path, required=True)
    parser.add_argument("--segments", type=Path, required=True)
    parser.add_argument("--baseline-timing", type=Path, required=True)
    parser.add_argument("--max-early-pull-sec", type=float, default=0.4)
    parser.add_argument("--min-gap", type=float, default=0.05)
    args = parser.parse_args()
    print(
        json.dumps(
            analyze(
                timing_path=args.timing,
                segments_path=args.segments,
                baseline_timing_path=args.baseline_timing,
                max_early_pull_sec=args.max_early_pull_sec,
                min_gap=args.min_gap,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
