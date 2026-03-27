"""Tracing helpers for forced-alignment stages."""

from __future__ import annotations

import json
import os
from typing import Any

from ... import models


def parse_forced_trace_line_range() -> tuple[int, int] | None:
    raw = os.environ.get("Y2K_TRACE_WHISPER_LINE_RANGE", "").strip()
    if not raw:
        return None
    try:
        start_s, end_s = raw.split("-", 1)
        start = int(start_s)
        end = int(end_s)
    except (TypeError, ValueError):
        return None
    if start <= 0 or end < start:
        return None
    return start, end


def serialize_forced_trace_lines(
    lines: list[models.Line],
    *,
    line_range: tuple[int, int] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, line in enumerate(lines, start=1):
        if line_range is not None and not (line_range[0] <= idx <= line_range[1]):
            continue
        if not line.words:
            continue
        rows.append(
            {
                "line_index": idx,
                "text": line.text,
                "start": round(line.start_time, 3),
                "end": round(line.end_time, 3),
                "duration": round(line.end_time - line.start_time, 3),
                "words": [
                    {
                        "text": word.text,
                        "start": round(word.start_time, 3),
                        "end": round(word.end_time, 3),
                    }
                    for word in line.words
                ],
            }
        )
    return rows


def capture_forced_trace_snapshot(
    snapshots: list[dict[str, Any]],
    *,
    stage: str,
    lines: list[models.Line],
    line_range: tuple[int, int] | None,
) -> None:
    snapshots.append(
        {
            "stage": stage,
            "line_count": len(lines),
            "lines": serialize_forced_trace_lines(lines, line_range=line_range),
        }
    )


def maybe_write_forced_trace_snapshot_file(
    *,
    trace_path: str,
    snapshots: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    if not trace_path:
        return
    with open(trace_path, "w", encoding="utf-8") as fh:
        json.dump({"metadata": metadata, "snapshots": snapshots}, fh, indent=2)
