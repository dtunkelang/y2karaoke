#!/usr/bin/env python3
"""Analyze long-gap shift onset candidates against gold timing."""

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


def analyze(*, trace_path: Path, gold_path: Path) -> dict[str, Any]:
    trace = _load_json(trace_path)
    gold = _load_json(gold_path)
    rows = trace.get("rows", [])
    analyzed_rows: list[dict[str, Any]] = []
    for row in rows:
        if row.get("decision") != "shift":
            continue
        line_index = int(row.get("line_index") or 0)
        gold_line = _find_gold_line(gold, line_index)
        if gold_line is None:
            continue
        gold_start = float(gold_line["start"])
        current_start = float(row["start"])
        chosen_onset = float(row["chosen_onset"])
        candidate_onsets = [float(onset) for onset in row.get("candidate_onsets", [])]
        if not candidate_onsets:
            continue
        best_onset = min(candidate_onsets, key=lambda onset: abs(onset - gold_start))
        analyzed_rows.append(
            {
                "call_index": int(row.get("call_index") or 0),
                "line_index": line_index,
                "text": row.get("text"),
                "gold_start": round(gold_start, 3),
                "current_start": round(current_start, 3),
                "chosen_onset": round(chosen_onset, 3),
                "best_onset": round(best_onset, 3),
                "chosen_abs_error": round(abs(chosen_onset - gold_start), 3),
                "best_abs_error": round(abs(best_onset - gold_start), 3),
                "candidate_onsets": [round(onset, 3) for onset in candidate_onsets],
            }
        )
    return {"rows": analyzed_rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(trace_path=args.trace, gold_path=args.gold), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
