#!/usr/bin/env python3
"""Analyze restored lines that still have a later exact phrase window available."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z]+", "", text.lower())


def _find_exact_phrase_window(
    line: dict[str, Any], segments: list[dict[str, Any]]
) -> tuple[float, float] | None:
    tokens = [_normalize(word["text"]) for word in line.get("words", [])]
    tokens = [token for token in tokens if token]
    if not tokens:
        return None
    for segment in segments:
        words = segment.get("words", [])
        seg_tokens = [_normalize(word["text"]) for word in words]
        for idx in range(0, len(seg_tokens) - len(tokens) + 1):
            if seg_tokens[idx : idx + len(tokens)] != tokens:
                continue
            return float(words[idx]["start"]), float(
                words[idx + len(tokens) - 1]["end"]
            )
    return None


def analyze(
    *,
    timing_path: Path,
    baseline_timing_path: Path,
    segments_path: Path,
    min_start_gain_sec: float = 0.25,
    baseline_anchor_tolerance: float = 0.08,
) -> dict[str, Any]:
    timing = _load_json(timing_path)
    baseline_lines = {
        int(row.get("line_index", 0) or 0): row
        for row in _load_json(baseline_timing_path).get("lines", [])
    }
    segments = _load_json(segments_path).get("segments", [])
    rows: list[dict[str, Any]] = []

    for line in timing.get("lines", []):
        line_index = int(line.get("index", 0) or 0)
        baseline = baseline_lines.get(line_index)
        if baseline is None:
            continue
        phrase_window = _find_exact_phrase_window(line, segments)
        if phrase_window is None:
            continue
        phrase_start, phrase_end = phrase_window
        current_start = float(line["start"])
        current_end = float(line["end"])
        baseline_start = float(baseline["start"])
        if phrase_start <= current_start + min_start_gain_sec:
            continue
        rows.append(
            {
                "line_index": line_index,
                "text": line["text"],
                "current_start": round(current_start, 3),
                "current_end": round(current_end, 3),
                "baseline_start": round(baseline_start, 3),
                "phrase_start": round(phrase_start, 3),
                "phrase_end": round(phrase_end, 3),
                "start_gain_sec": round(phrase_start - current_start, 3),
                "baseline_anchor_delta_sec": round(
                    abs(current_start - baseline_start), 3
                ),
                "blocked_by_baseline_anchor_tolerance": abs(
                    current_start - baseline_start
                )
                > baseline_anchor_tolerance,
            }
        )

    return {"rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing", type=Path, required=True)
    parser.add_argument("--baseline-timing", type=Path, required=True)
    parser.add_argument("--segments", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            analyze(
                timing_path=args.timing,
                baseline_timing_path=args.baseline_timing,
                segments_path=args.segments,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
