#!/usr/bin/env python3
"""Analyze why adjacent lines do or do not qualify for repetition-run realignment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from y2karaoke.core.models import Line, Word
from y2karaoke.core.components.whisper.whisper_mapping_post_text import (
    _normalize_match_token,
    _soft_token_overlap_ratio,
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _line_from_json(payload: dict[str, Any]) -> Line:
    return Line(
        words=[
            Word(
                text=word["text"],
                start_time=float(word["start"]),
                end_time=float(word["end"]),
            )
            for word in payload.get("words", [])
        ]
    )


def _normalized_tokens(line: Line) -> list[str]:
    return [
        _normalize_match_token(word.text)
        for word in line.words
        if _normalize_match_token(word.text)
    ]


def analyze(
    *,
    timing_path: Path,
    min_overlap: float = 0.4,
    min_run_len: int = 3,
) -> dict[str, Any]:
    payload = _load_json(timing_path)
    lines = [_line_from_json(line) for line in payload.get("lines", [])]
    rows: list[dict[str, Any]] = []
    for idx in range(len(lines) - 1):
        line = lines[idx]
        nxt = lines[idx + 1]
        line_tokens = _normalized_tokens(line)
        next_tokens = _normalized_tokens(nxt)
        overlap = _soft_token_overlap_ratio(line_tokens, next_tokens)
        exact_duplicate = line.text.strip().lower() == nxt.text.strip().lower()
        rows.append(
            {
                "line_index": idx + 1,
                "next_line_index": idx + 2,
                "line_text": line.text,
                "next_text": nxt.text,
                "line_tokens": line_tokens,
                "next_tokens": next_tokens,
                "overlap": round(overlap, 3),
                "exact_duplicate": exact_duplicate,
                "passes_pair_overlap_gate": overlap >= min_overlap,
                "would_still_fail_run_gate": not exact_duplicate or min_run_len > 2,
            }
        )
    return {"rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing", type=Path, required=True)
    parser.add_argument("--min-overlap", type=float, default=0.4)
    parser.add_argument("--min-run-len", type=int, default=3)
    args = parser.parse_args()
    print(
        json.dumps(
            analyze(
                timing_path=args.timing,
                min_overlap=args.min_overlap,
                min_run_len=args.min_run_len,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
