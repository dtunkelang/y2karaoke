"""Trace helpers for Whisper integration alignment diagnostics."""

from __future__ import annotations

import json
import os
from typing import Any

from ... import models


def parse_trace_line_range() -> tuple[int, int] | None:
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


def capture_trace_snapshot(
    snapshots: list[dict[str, Any]],
    *,
    stage: str,
    lines: list[models.Line],
    line_range: tuple[int, int] | None,
) -> None:
    if line_range is None:
        return
    start_idx, end_idx = line_range
    rows: list[dict[str, Any]] = []
    for idx, line in enumerate(lines, start=1):
        if idx < start_idx or idx > end_idx or not line.words:
            continue
        rows.append(
            {
                "line_index": idx,
                "text": line.text,
                "start": round(line.start_time, 3),
                "end": round(line.end_time, 3),
                "duration": round(line.end_time - line.start_time, 3),
            }
        )
    snapshots.append({"stage": stage, "count": len(lines), "lines": rows})


def maybe_write_trace_snapshot_file(
    *,
    snapshots: list[dict[str, Any]],
    trace_path: str,
) -> None:
    if not trace_path:
        return
    payload = {"snapshots": snapshots}
    with open(trace_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
