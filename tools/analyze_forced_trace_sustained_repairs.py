#!/usr/bin/env python3
"""Analyze sustained-duration expansions from a forced fallback trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _lines_by_index(snapshot: dict[str, Any]) -> dict[int, dict[str, Any]]:
    lines = snapshot.get("lines", [])
    return {
        int(line.get("line_index", idx + 1) or idx + 1): line
        for idx, line in enumerate(lines)
    }


def _duration(line: dict[str, Any]) -> float:
    return float(line.get("duration", 0.0) or 0.0)


def analyze_trace(
    *,
    trace_path: Path,
    min_duration_gain_sec: float = 2.0,
    min_duration_gain_ratio: float = 2.0,
) -> dict[str, Any]:
    trace = _load_json(trace_path)
    snapshots = {snap["stage"]: snap for snap in trace.get("snapshots", [])}
    loaded = snapshots.get("loaded_forced_alignment")
    sustained = snapshots.get("after_sustained_line_repair")
    finalized = snapshots.get("after_finalize_forced_line_timing")
    rows: list[dict[str, Any]] = []

    if loaded is None or sustained is None:
        return {
            "trace_path": str(trace_path),
            "candidate_count": 0,
            "rows": rows,
        }

    loaded_by_index = _lines_by_index(loaded)
    sustained_by_index = _lines_by_index(sustained)
    finalized_by_index = _lines_by_index(finalized or {"lines": []})

    for line_index, loaded_line in loaded_by_index.items():
        sustained_line = sustained_by_index.get(line_index)
        if sustained_line is None:
            continue
        loaded_duration = _duration(loaded_line)
        sustained_duration = _duration(sustained_line)
        if loaded_duration <= 0.0:
            continue
        duration_gain_sec = sustained_duration - loaded_duration
        duration_gain_ratio = sustained_duration / loaded_duration
        if duration_gain_sec < min_duration_gain_sec:
            continue
        if duration_gain_ratio < min_duration_gain_ratio:
            continue
        finalized_line = finalized_by_index.get(line_index, sustained_line)
        rows.append(
            {
                "line_index": line_index,
                "text": loaded_line.get("text", ""),
                "loaded_start": round(float(loaded_line.get("start", 0.0)), 3),
                "loaded_end": round(float(loaded_line.get("end", 0.0)), 3),
                "loaded_duration": round(loaded_duration, 3),
                "sustained_start": round(float(sustained_line.get("start", 0.0)), 3),
                "sustained_end": round(float(sustained_line.get("end", 0.0)), 3),
                "sustained_duration": round(sustained_duration, 3),
                "duration_gain_sec": round(duration_gain_sec, 3),
                "duration_gain_ratio": round(duration_gain_ratio, 3),
                "final_start": round(float(finalized_line.get("start", 0.0)), 3),
                "final_end": round(float(finalized_line.get("end", 0.0)), 3),
                "final_duration": round(_duration(finalized_line), 3),
            }
        )

    return {
        "trace_path": str(trace_path),
        "candidate_count": len(rows),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze_trace(trace_path=args.trace), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
