#!/usr/bin/env python3
"""Summarize stalled-segment rescue opportunities from segment selection traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("rows", [])
    return [row for row in rows if isinstance(row, dict)]


def _candidate_gain(
    row: dict[str, Any], key: str
) -> tuple[float | None, dict[str, Any] | None]:
    current = float(row.get("final_score", 0.0) or 0.0)
    best_gain: float | None = None
    best_item: dict[str, Any] | None = None
    for item in row.get(key, []) or []:
        if not isinstance(item, dict):
            continue
        seg_score = item.get("score")
        merged_score = item.get("merged_score")
        candidates = []
        if isinstance(seg_score, (int, float)):
            candidates.append((float(seg_score), "score"))
        if isinstance(merged_score, (int, float)):
            candidates.append((float(merged_score), "merged_score"))
        for score, source in candidates:
            gain = score - current
            if best_gain is None or gain > best_gain:
                best_gain = gain
                best_item = {
                    "segment_index": item.get("segment_index"),
                    "source": source,
                    "candidate_score": round(score, 4),
                    "gain": round(gain, 4),
                    "bag_preview": item.get("bag_preview", []),
                }
    return best_gain, best_item


def _collect_opportunities(
    rows: list[dict[str, Any]],
    *,
    min_gain: float,
    trace_key: str,
) -> list[dict[str, Any]]:
    opportunities: list[dict[str, Any]] = []
    for row in rows:
        gain, best_item = _candidate_gain(row, trace_key)
        if gain is None or best_item is None or gain < min_gain:
            continue
        opportunities.append(
            {
                "line_index": row.get("line_index"),
                "words": row.get("words", []),
                "final_segment": row.get("final_segment"),
                "final_score": round(float(row.get("final_score", 0.0) or 0.0), 4),
                "seg_cursor_before": row.get("seg_cursor_before"),
                "best_candidate": best_item,
            }
        )
    return opportunities


def _format_markdown(
    *,
    trace_path: Path,
    opportunities: list[dict[str, Any]],
    min_gain: float,
    trace_key: str,
) -> str:
    lines = [
        "# Stalled Segment Rescue Opportunities",
        "",
        f"- Trace: `{trace_path}`",
        f"- Trace key: `{trace_key}`",
        f"- Minimum gain: `{min_gain}`",
        "",
    ]
    if not opportunities:
        lines.append("No qualifying stalled-segment rescue opportunities found.")
        return "\n".join(lines) + "\n"
    for row in opportunities:
        candidate = row["best_candidate"]
        lines.extend(
            [
                f"## Line {row['line_index']}",
                "",
                f"- Words: `{' '.join(row.get('words', []))}`",
                f"- Cursor/final segment: `{row.get('seg_cursor_before')} -> {row.get('final_segment')}`",
                f"- Final score: `{row.get('final_score')}`",
                f"- Best candidate segment: `{candidate.get('segment_index')}` via `{candidate.get('source')}`",
                f"- Candidate score: `{candidate.get('candidate_score')}`",
                f"- Gain: `{candidate.get('gain')}`",
                f"- Bag preview: `{', '.join(candidate.get('bag_preview', []))}`",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--min-gain", type=float, default=0.1)
    parser.add_argument(
        "--trace-key",
        default="experimental_low_score_stalled_segment_scores",
    )
    args = parser.parse_args()

    trace_path = Path(args.trace_json)
    rows = _load_rows(trace_path)
    opportunities = _collect_opportunities(
        rows,
        min_gain=args.min_gain,
        trace_key=args.trace_key,
    )
    output = _format_markdown(
        trace_path=trace_path,
        opportunities=opportunities,
        min_gain=args.min_gain,
        trace_key=args.trace_key,
    )
    Path(args.output_md).write_text(output, encoding="utf-8")
    print(
        "stalled_segment_opportunities: OK\n"
        f"  trace={trace_path}\n"
        f"  output_md={args.output_md}\n"
        f"  opportunities={len(opportunities)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
