#!/usr/bin/env python3
"""Summarize segment-cursor stall runs from segment selection traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("rows", [])
    return [row for row in rows if isinstance(row, dict)]


def _stall_runs(
    rows: list[dict[str, Any]],
    *,
    min_run_len: int,
    max_score: float,
) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for row in rows:
        seg = row.get("final_segment")
        score = float(row.get("final_score", 0.0))
        cursor_before = row.get("seg_cursor_before")
        cursor_after = row.get("seg_cursor_after")
        line_index = row.get("line_index")
        if (
            isinstance(seg, int)
            and isinstance(cursor_before, int)
            and isinstance(cursor_after, int)
            and isinstance(line_index, int)
            and seg == cursor_before == cursor_after
            and score <= max_score
        ):
            if current is None or current["segment_index"] != seg:
                current = {
                    "segment_index": seg,
                    "start_line": line_index,
                    "end_line": line_index,
                    "line_count": 1,
                    "scores": [score],
                    "lines": [line_index],
                }
                runs.append(current)
            else:
                current["end_line"] = line_index
                current["line_count"] += 1
                current["scores"].append(score)
                current["lines"].append(line_index)
            continue
        current = None
    return [run for run in runs if run["line_count"] >= min_run_len]


def _format_markdown(
    *,
    trace_path: Path,
    runs: list[dict[str, Any]],
    max_score: float,
    min_run_len: int,
) -> str:
    lines = [
        "# Segment Cursor Stall Analysis",
        "",
        f"- Trace: `{trace_path}`",
        f"- Max score threshold: `{max_score}`",
        f"- Minimum run length: `{min_run_len}`",
        "",
    ]
    if not runs:
        lines.append("No qualifying stall runs found.")
        return "\n".join(lines) + "\n"
    for run in runs:
        mean_score = sum(run["scores"]) / len(run["scores"])
        lines.extend(
            [
                f"## Segment {run['segment_index']}",
                "",
                f"- Lines: `{run['start_line']}-{run['end_line']}`",
                f"- Count: `{run['line_count']}`",
                f"- Mean score: `{mean_score:.4f}`",
                f"- Scores: `{', '.join(f'{score:.4f}' for score in run['scores'])}`",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--min-run-len", type=int, default=3)
    parser.add_argument("--max-score", type=float, default=0.2)
    args = parser.parse_args()

    trace_path = Path(args.trace_json)
    rows = _load_rows(trace_path)
    runs = _stall_runs(rows, min_run_len=args.min_run_len, max_score=args.max_score)
    output = _format_markdown(
        trace_path=trace_path,
        runs=runs,
        max_score=args.max_score,
        min_run_len=args.min_run_len,
    )
    Path(args.output_md).write_text(output, encoding="utf-8")
    print(
        "segment_cursor_stall_analysis: OK\n"
        f"  trace={trace_path}\n"
        f"  output_md={args.output_md}\n"
        f"  runs={len(runs)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
