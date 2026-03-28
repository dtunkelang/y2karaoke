#!/usr/bin/env python3
"""Simulate extending only the last word of final lines toward local tail evidence."""

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


def _find_phrase_window(
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
    segments_path: Path,
    baseline_timing_path: Path,
    min_word_count: int = 5,
    min_end_gain_sec: float = 0.35,
    max_start_delta_sec: float = 0.45,
) -> dict[str, Any]:
    timing = _load_json(timing_path)
    segments = _load_json(segments_path).get("segments", [])
    baseline_lines = {
        int(row.get("line_index", 0) or 0): row
        for row in _load_json(baseline_timing_path).get("lines", [])
    }
    rows: list[dict[str, Any]] = []

    timing_lines = timing.get("lines", [])
    for idx, line in enumerate(timing_lines):
        line_index = int(line.get("line_index", 0) or 0)
        baseline = baseline_lines.get(line_index)
        if baseline is None or idx != len(timing_lines) - 1:
            continue
        words = line.get("words", [])
        if len(words) < min_word_count or len(words) != len(baseline.get("words", [])):
            continue
        if abs(float(line["start"]) - float(baseline["start"])) > max_start_delta_sec:
            continue
        phrase_window = _find_phrase_window(line, segments)
        if phrase_window is None:
            continue
        _window_start, window_end = phrase_window
        if window_end <= float(line["end"]) + min_end_gain_sec:
            continue
        rows.append(
            {
                "line_index": line_index,
                "text": line["text"],
                "current_start": round(float(line["start"]), 3),
                "current_end": round(float(line["end"]), 3),
                "baseline_end": round(float(baseline["end"]), 3),
                "phrase_window_end": round(window_end, 3),
                "simulated_start": round(float(line["start"]), 3),
                "simulated_end": round(window_end, 3),
            }
        )

    return {"rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing", type=Path, required=True)
    parser.add_argument("--segments", type=Path, required=True)
    parser.add_argument("--baseline-timing", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            analyze(
                timing_path=args.timing,
                segments_path=args.segments,
                baseline_timing_path=args.baseline_timing,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
