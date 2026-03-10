#!/usr/bin/env python3
"""Summarize low-confidence mapper assignment lines from mapper trace output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_lines(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [row for row in data.get("lines", []) if isinstance(row, dict)]


def _summarize(
    lines: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    low_conf = []
    total = 0
    for row in lines:
        profile = row.get("assignment_confidence") or {}
        if not isinstance(profile, dict):
            continue
        total += 1
        if profile.get("low_confidence"):
            low_conf.append(
                {
                    "line_index": row.get("line_index"),
                    "text": row.get("text"),
                    "mapped_start": row.get("mapped_start"),
                    "line_segment": row.get("line_segment"),
                    "assigned_segment_votes": row.get("assigned_segment_votes"),
                    "profile": profile,
                }
            )
    summary = {
        "line_count": total,
        "low_confidence_line_count": len(low_conf),
    }
    return low_conf, summary


def _format_markdown(
    *,
    trace_path: Path,
    summary: dict[str, Any],
    low_conf: list[dict[str, Any]],
) -> str:
    lines = [
        "# Mapper Assignment Confidence",
        "",
        f"- Trace: `{trace_path}`",
        f"- Total traced lines: `{summary['line_count']}`",
        f"- Low-confidence lines: `{summary['low_confidence_line_count']}`",
        "",
    ]
    if not low_conf:
        lines.append("No low-confidence lines found.")
        return "\n".join(lines) + "\n"
    for row in low_conf:
        profile = row["profile"]
        lines.extend(
            [
                f"## Line {row['line_index']}",
                "",
                f"- Text: `{row['text']}`",
                f"- Mapped start: `{row['mapped_start']}`",
                f"- Line segment: `{row['line_segment']}`",
                f"- Assigned segment votes: `{row['assigned_segment_votes']}`",
                f"- Total assigned: `{profile.get('total_assigned')}`",
                f"- Lexical overlap ratio: `{profile.get('lexical_overlap_ratio')}`",
                f"- Placeholder ratio: `{profile.get('placeholder_ratio')}`",
                f"- Median start drift: `{profile.get('median_start_drift_sec')}`",
                f"- Max start drift: `{profile.get('max_start_drift_sec')}`",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    trace_path = Path(args.trace_json)
    low_conf, summary = _summarize(_load_lines(trace_path))
    output = _format_markdown(
        trace_path=trace_path,
        summary=summary,
        low_conf=low_conf,
    )
    Path(args.output_md).write_text(output, encoding="utf-8")
    print(
        "mapper_assignment_confidence: OK\n"
        f"  trace={trace_path}\n"
        f"  output_md={args.output_md}\n"
        f"  low_confidence_lines={summary['low_confidence_line_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
