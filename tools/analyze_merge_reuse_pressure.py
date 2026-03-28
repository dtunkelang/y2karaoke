#!/usr/bin/env python3
"""Analyze whether merged adjacent lines are followed by immediate phrase reuse."""

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


def _overlap_ratio(left: str, right: str) -> float:
    left_tokens = set(_normalize_tokens(left))
    right_tokens = set(_normalize_tokens(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(len(left_tokens), len(right_tokens))


def analyze(
    *,
    timing_path: Path,
    gold_path: Path,
    aggregate_path: Path,
    vocals_path: Path,
    min_reuse_overlap: float = 0.4,
) -> dict[str, Any]:
    timing = _load_json(timing_path)
    gold = _load_json(gold_path)
    aggregate = _load_json(aggregate_path)
    vocals = _load_json(vocals_path)

    rows: list[dict[str, Any]] = []
    timing_lines = timing.get("lines", [])
    gold_lines = {int(line["line_index"]): line for line in gold.get("lines", [])}
    agg_segments = aggregate.get("segments", [])
    vocals_segments = vocals.get("segments", [])

    if len(timing_lines) < 3 or len(agg_segments) < 2:
        return {
            "timing_path": str(timing_path),
            "gold_path": str(gold_path),
            "aggregate_path": str(aggregate_path),
            "vocals_path": str(vocals_path),
            "triplet_count": 0,
            "rows": [],
        }

    for idx in range(len(timing_lines) - 2):
        left = timing_lines[idx]
        middle = timing_lines[idx + 1]
        right = timing_lines[idx + 2]
        left_gold = gold_lines.get(int(left.get("index", 0) or 0))
        middle_gold = gold_lines.get(int(middle.get("index", 0) or 0))
        right_gold = gold_lines.get(int(right.get("index", 0) or 0))
        if left_gold is None or middle_gold is None or right_gold is None:
            continue

        merged_text = f"{left['text']} {middle['text']}".strip()
        if not agg_segments:
            continue
        agg_first = agg_segments[0]
        agg_second = agg_segments[1] if len(agg_segments) > 1 else None
        if agg_second is None:
            continue
        if _normalize_tokens(agg_first.get("text", "")) != _normalize_tokens(
            merged_text
        ):
            continue

        overlap = _overlap_ratio(str(left.get("text", "")), str(right.get("text", "")))
        if overlap < min_reuse_overlap:
            continue

        vocals_split_ok = False
        if len(vocals_segments) >= 2:
            vocals_left = vocals_segments[0]
            vocals_mid = vocals_segments[1]
            vocals_split_ok = _normalize_tokens(
                vocals_left.get("text", "")
            ) == _normalize_tokens(str(left.get("text", ""))) and _normalize_tokens(
                vocals_mid.get("text", "")
            ) == _normalize_tokens(
                str(middle.get("text", ""))
            )

        rows.append(
            {
                "left_index": int(left.get("index", 0) or 0),
                "middle_index": int(middle.get("index", 0) or 0),
                "right_index": int(right.get("index", 0) or 0),
                "left_text": left["text"],
                "middle_text": middle["text"],
                "right_text": right["text"],
                "reuse_overlap_ratio": round(overlap, 3),
                "aggregate_first_end": round(float(agg_first.get("end", 0.0)), 3),
                "aggregate_second_start": round(float(agg_second.get("start", 0.0)), 3),
                "aggregate_segment_gap_sec": round(
                    float(agg_second.get("start", 0.0))
                    - float(agg_first.get("end", 0.0)),
                    3,
                ),
                "predicted_middle_end": round(float(middle["end"]), 3),
                "gold_middle_end": round(float(middle_gold["end"]), 3),
                "middle_end_error_sec": round(
                    float(middle["end"]) - float(middle_gold["end"]), 3
                ),
                "predicted_right_start": round(float(right["start"]), 3),
                "gold_right_start": round(float(right_gold["start"]), 3),
                "right_start_error_sec": round(
                    float(right["start"]) - float(right_gold["start"]), 3
                ),
                "vocals_split_ok": vocals_split_ok,
            }
        )

    return {
        "timing_path": str(timing_path),
        "gold_path": str(gold_path),
        "aggregate_path": str(aggregate_path),
        "vocals_path": str(vocals_path),
        "triplet_count": len(rows),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--vocals", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            analyze(
                timing_path=args.timing,
                gold_path=args.gold,
                aggregate_path=args.aggregate,
                vocals_path=args.vocals,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
