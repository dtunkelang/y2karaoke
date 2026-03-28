#!/usr/bin/env python3
"""Analyze forced-trace lines that lack lexical support in accepted transcription."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _overlap_ratio(left: list[str], right: list[str]) -> float:
    if not left or not right:
        return 0.0
    left_set = set(left)
    right_set = set(right)
    return len(left_set & right_set) / max(len(left_set), len(right_set))


def _best_transcription_overlap(
    line_text: str,
    transcription_segments: list[dict[str, Any]],
) -> tuple[float, dict[str, Any] | None]:
    line_tokens = _normalize_tokens(line_text)
    best_ratio = 0.0
    best_segment: dict[str, Any] | None = None
    for segment in transcription_segments:
        ratio = _overlap_ratio(
            line_tokens, _normalize_tokens(str(segment.get("text", "")))
        )
        if ratio > best_ratio:
            best_ratio = ratio
            best_segment = segment
    return best_ratio, best_segment


def analyze_trace(
    *,
    trace_path: Path,
    max_transcription_overlap_ratio: float = 0.25,
    min_line_duration_sec: float = 0.5,
    min_sustained_gain_sec: float = 0.5,
) -> dict[str, Any]:
    trace = _load_json(trace_path)
    metadata = trace.get("metadata", {})
    transcription_segments = metadata.get("transcription_segment_preview", [])
    snapshots = {snap["stage"]: snap for snap in trace.get("snapshots", [])}
    loaded = snapshots.get("loaded_forced_alignment", {"lines": []})
    sustained = snapshots.get("after_sustained_line_repair", {"lines": []})
    sustained_by_index = {
        int(line.get("line_index", idx + 1) or idx + 1): line
        for idx, line in enumerate(sustained.get("lines", []))
    }
    rows: list[dict[str, Any]] = []

    for idx, line in enumerate(loaded.get("lines", []), start=1):
        line_index = int(line.get("line_index", idx) or idx)
        duration = float(line.get("duration", 0.0) or 0.0)
        if duration < min_line_duration_sec:
            continue
        overlap_ratio, best_segment = _best_transcription_overlap(
            str(line.get("text", "")),
            transcription_segments,
        )
        if overlap_ratio > max_transcription_overlap_ratio:
            continue
        sustained_line = sustained_by_index.get(line_index, line)
        sustained_end = float(sustained_line.get("end", line.get("end", 0.0)))
        sustained_gain_sec = sustained_end - float(line.get("end", 0.0))
        if sustained_gain_sec < min_sustained_gain_sec:
            continue
        rows.append(
            {
                "line_index": line_index,
                "text": line.get("text", ""),
                "loaded_start": round(float(line.get("start", 0.0)), 3),
                "loaded_end": round(float(line.get("end", 0.0)), 3),
                "loaded_duration": round(duration, 3),
                "best_transcription_overlap_ratio": round(overlap_ratio, 3),
                "best_transcription_segment": (
                    {
                        "start": round(float(best_segment.get("start", 0.0)), 3),
                        "end": round(float(best_segment.get("end", 0.0)), 3),
                        "text": best_segment.get("text", ""),
                    }
                    if best_segment is not None
                    else None
                ),
                "sustained_start": round(float(sustained_line.get("start", 0.0)), 3),
                "sustained_end": round(sustained_end, 3),
                "sustained_gain_sec": round(sustained_gain_sec, 3),
            }
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
