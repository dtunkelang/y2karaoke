#!/usr/bin/env python3
"""Analyze low-confidence leading overhangs in raw WhisperX forced traces."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

_LIGHT_LEADING_TOKENS = {"the", "a", "an", "if"}
_TOKEN_RE = re.compile(r"[a-z0-9']+")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _norm_token(text: str) -> str:
    return "".join(_TOKEN_RE.findall(text.lower()))


def analyze_trace(
    *,
    trace_path: Path,
    max_first_score: float = 0.4,
    min_overhang_sec: float = 0.35,
    min_word_count: int = 4,
) -> dict[str, Any]:
    trace = _load_json(trace_path)
    rows: list[dict[str, Any]] = []

    for line_mapping in trace.get("line_mappings", []):
        segment_words = line_mapping.get("segment_words", [])
        aligned_segments = trace.get("aligned_segments", [])
        line_index = int(line_mapping.get("line_index", 0) or 0)
        if line_index < 0 or line_index >= len(aligned_segments):
            continue
        aligned_words = aligned_segments[line_index].get("words", [])
        if not segment_words or not aligned_words:
            continue
        if len(segment_words) < min_word_count:
            continue
        first_word = aligned_words[0]
        first_score = first_word.get("score")
        if not isinstance(first_score, (int, float)) or first_score > max_first_score:
            continue
        first_text = str(first_word.get("word") or first_word.get("text") or "")
        if _norm_token(first_text) not in _LIGHT_LEADING_TOKENS:
            continue
        segment_start = float(
            line_mapping.get("segment_start", segment_words[0]["start"])
        )
        first_end = float(segment_words[0]["end"])
        overhang_sec = first_end - segment_start
        if overhang_sec < min_overhang_sec:
            continue
        rows.append(
            {
                "line_index": line_index + 1,
                "text": line_mapping.get("line_text", ""),
                "first_word": first_text,
                "first_score": round(float(first_score), 3),
                "segment_start": round(segment_start, 3),
                "first_word_end": round(first_end, 3),
                "leading_overhang_sec": round(overhang_sec, 3),
                "segment_end": round(float(line_mapping.get("segment_end", 0.0)), 3),
            }
        )

    return {
        "trace_path": str(trace_path),
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
