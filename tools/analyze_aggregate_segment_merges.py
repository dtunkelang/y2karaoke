#!/usr/bin/env python3
"""Detect adjacent gold-line merges in aggregate Whisper segments."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize(text: str) -> str:
    return " ".join(
        token for token in re.sub(r"[^a-z0-9]+", " ", text.lower()).split() if token
    )


def _segment_rows(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, segment in enumerate(segments, start=1):
        rows.append(
            {
                "segment_index": index,
                "start": float(segment.get("start", 0.0)),
                "end": float(segment.get("end", 0.0)),
                "text": str(segment.get("text", "")),
                "normalized_text": _normalize(str(segment.get("text", ""))),
            }
        )
    return rows


def analyze(
    *,
    aggregate_path: Path,
    vocals_path: Path,
    gold_path: Path,
) -> dict[str, Any]:
    aggregate = _segment_rows(_load_json(aggregate_path).get("segments", []))
    vocals = _segment_rows(_load_json(vocals_path).get("segments", []))
    gold_lines = _load_json(gold_path).get("lines", [])

    normalized_gold = [
        {
            "line_index": int(line["line_index"]),
            "text": str(line["text"]),
            "normalized_text": _normalize(str(line["text"])),
        }
        for line in gold_lines
    ]

    rows: list[dict[str, Any]] = []
    for idx in range(len(normalized_gold) - 1):
        left = normalized_gold[idx]
        right = normalized_gold[idx + 1]
        merged_text = f"{left['normalized_text']} {right['normalized_text']}".strip()
        aggregate_match = next(
            (
                segment
                for segment in aggregate
                if segment["normalized_text"] == merged_text
            ),
            None,
        )
        if aggregate_match is None:
            continue
        vocals_left_index = next(
            (
                i
                for i, segment in enumerate(vocals)
                if segment["normalized_text"] == left["normalized_text"]
            ),
            None,
        )
        if vocals_left_index is None or vocals_left_index + 1 >= len(vocals):
            continue
        vocals_left = vocals[vocals_left_index]
        vocals_right = vocals[vocals_left_index + 1]
        if vocals_right["normalized_text"] != right["normalized_text"]:
            continue
        rows.append(
            {
                "left_line_index": left["line_index"],
                "right_line_index": right["line_index"],
                "left_text": left["text"],
                "right_text": right["text"],
                "aggregate_segment_index": aggregate_match["segment_index"],
                "aggregate_start": round(aggregate_match["start"], 3),
                "aggregate_end": round(aggregate_match["end"], 3),
                "aggregate_text": aggregate_match["text"],
                "vocals_left_segment_index": vocals_left["segment_index"],
                "vocals_right_segment_index": vocals_right["segment_index"],
                "vocals_left_start": round(vocals_left["start"], 3),
                "vocals_left_end": round(vocals_left["end"], 3),
                "vocals_right_start": round(vocals_right["start"], 3),
                "vocals_right_end": round(vocals_right["end"], 3),
            }
        )

    return {
        "aggregate_path": str(aggregate_path),
        "vocals_path": str(vocals_path),
        "gold_path": str(gold_path),
        "merge_count": len(rows),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--vocals", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            analyze(
                aggregate_path=args.aggregate,
                vocals_path=args.vocals,
                gold_path=args.gold,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
