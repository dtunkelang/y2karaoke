#!/usr/bin/env python3
"""Analyze exact adjacent boundary compression from a forced fallback trace."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_tokens(text: str) -> list[str]:
    return [
        token for token in re.sub(r"[^a-z0-9]+", " ", text.lower()).split() if token
    ]


def _matches(left: str, right: str) -> bool:
    return _normalize_tokens(left) == _normalize_tokens(right)


def _rows_for_stage(
    *,
    stage: str,
    lines: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    max_segment_gap_sec: float,
    min_tail_shortfall_sec: float,
    min_next_early_start_sec: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pair_count = min(len(lines), len(segments)) - 1
    for idx in range(pair_count):
        left = lines[idx]
        right = lines[idx + 1]
        left_segment = segments[idx]
        right_segment = segments[idx + 1]
        if not _matches(str(left.get("text", "")), str(left_segment.get("text", ""))):
            continue
        if not _matches(
            str(right.get("text", "")),
            str(right_segment.get("text", "")),
        ):
            continue
        segment_gap = float(right_segment.get("start", 0.0)) - float(
            left_segment.get("end", 0.0)
        )
        if abs(segment_gap) > max_segment_gap_sec:
            continue
        tail_shortfall = float(left_segment.get("end", 0.0)) - float(
            left.get("end", 0.0)
        )
        next_early_start = float(right_segment.get("start", 0.0)) - float(
            right.get("start", 0.0)
        )
        if tail_shortfall < min_tail_shortfall_sec:
            continue
        if next_early_start < min_next_early_start_sec:
            continue
        rows.append(
            {
                "stage": stage,
                "left_index": int(left.get("line_index", idx + 1) or idx + 1),
                "right_index": int(right.get("line_index", idx + 2) or idx + 2),
                "left_text": left.get("text", ""),
                "right_text": right.get("text", ""),
                "line_end": round(float(left.get("end", 0.0)), 3),
                "segment_end": round(float(left_segment.get("end", 0.0)), 3),
                "tail_shortfall_sec": round(tail_shortfall, 3),
                "line_start_next": round(float(right.get("start", 0.0)), 3),
                "segment_start_next": round(float(right_segment.get("start", 0.0)), 3),
                "next_early_start_sec": round(next_early_start, 3),
            }
        )
    return rows


def analyze_trace(
    *,
    trace_path: Path,
    max_segment_gap_sec: float = 0.1,
    min_tail_shortfall_sec: float = 0.3,
    min_next_early_start_sec: float = 0.3,
) -> dict[str, Any]:
    trace = _load_json(trace_path)
    metadata = trace.get("metadata", {})
    segments = metadata.get("transcription_segment_preview", [])
    snapshots = {snap["stage"]: snap for snap in trace.get("snapshots", [])}
    rows: list[dict[str, Any]] = []

    for stage in (
        "loaded_forced_alignment",
        "after_post_finalize_refrain_repairs",
        "after_restore_exact_segment_boundaries",
        "final_forced_lines",
    ):
        snapshot = snapshots.get(stage)
        if snapshot is None or not segments:
            continue
        rows.extend(
            _rows_for_stage(
                stage=stage,
                lines=snapshot.get("lines", []),
                segments=segments,
                max_segment_gap_sec=max_segment_gap_sec,
                min_tail_shortfall_sec=min_tail_shortfall_sec,
                min_next_early_start_sec=min_next_early_start_sec,
            )
        )

    return {
        "trace_path": str(trace_path),
        "transcription_segment_count": metadata.get("transcription_segment_count", 0),
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
