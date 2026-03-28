#!/usr/bin/env python3
"""Analyze subphrase timing windows for lines inside merged Whisper segments."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_token(text: str) -> str:
    return re.sub(r"[^a-z]+", "", text.lower())


def _line_phrase_window(
    line: dict[str, Any], segment: dict[str, Any]
) -> dict[str, Any] | None:
    line_tokens = [_normalize_token(word["text"]) for word in line.get("words", [])]
    line_tokens = [token for token in line_tokens if token]
    seg_words = segment.get("words", [])
    seg_tokens = [_normalize_token(word["text"]) for word in seg_words]
    seg_tokens = [token for token in seg_tokens]
    if not line_tokens or not seg_words:
        return None

    best_start = None
    best_len = 0
    token_count = len(line_tokens)
    for start in range(0, len(seg_words) - token_count + 1):
        window_tokens = [
            _normalize_token(seg_words[start + offset]["text"])
            for offset in range(token_count)
        ]
        if window_tokens == line_tokens:
            best_start = start
            best_len = token_count
            break

    if best_start is None:
        return None

    first_word = seg_words[best_start]
    last_word = seg_words[best_start + best_len - 1]
    return {
        "matched_tokens": line_tokens,
        "window_start": round(float(first_word["start"]), 3),
        "window_end": round(float(last_word["end"]), 3),
        "segment_start": round(float(segment["start"]), 3),
        "segment_end": round(float(segment["end"]), 3),
    }


def analyze(*, timing_path: Path, segments_path: Path) -> dict[str, Any]:
    timing = _load_json(timing_path)
    segments_payload = _load_json(segments_path)
    segments = segments_payload.get("segments", [])
    rows: list[dict[str, Any]] = []

    for line in timing.get("lines", []):
        line_start = float(line["start"])
        line_end = float(line["end"])
        line_index = int(line.get("line_index", 0) or 0)
        containing_segment = None
        for segment in segments:
            if (
                float(segment["start"]) - 0.25
                <= line_start
                <= float(segment["end"]) + 0.25
            ):
                containing_segment = segment
                break
        if containing_segment is None:
            continue
        phrase_window = _line_phrase_window(line, containing_segment)
        rows.append(
            {
                "line_index": line_index,
                "text": line.get("text", ""),
                "line_start": round(line_start, 3),
                "line_end": round(line_end, 3),
                "segment_start": round(float(containing_segment["start"]), 3),
                "segment_end": round(float(containing_segment["end"]), 3),
                "phrase_window": phrase_window,
            }
        )

    return {"rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing", type=Path, required=True)
    parser.add_argument("--segments", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            analyze(timing_path=args.timing, segments_path=args.segments),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
