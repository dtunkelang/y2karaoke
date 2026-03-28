#!/usr/bin/env python3
"""Analyze boundary drift for adjacent high-overlap line pairs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_tokens(text: str) -> list[str]:
    return [token for token in re.sub(r"[^a-z]+", " ", text.lower()).split() if token]


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
    min_overlap_ratio: float = 0.4,
    max_line_gap: int = 2,
) -> dict[str, Any]:
    timing = _load_json(timing_path)
    gold = _load_json(gold_path)
    gold_lines = {int(line["line_index"]): line for line in gold.get("lines", [])}
    rows: list[dict[str, Any]] = []

    timing_lines = timing.get("lines", [])
    for idx in range(len(timing_lines) - 1):
        left = timing_lines[idx]
        for right_offset in range(1, max_line_gap + 1):
            if idx + right_offset >= len(timing_lines):
                continue
            right = timing_lines[idx + right_offset]
            overlap = _overlap_ratio(
                str(left.get("text", "")),
                str(right.get("text", "")),
            )
            if overlap < min_overlap_ratio:
                continue
            left_index = int(left.get("index", 0) or 0)
            right_index = int(right.get("index", 0) or 0)
            left_gold = gold_lines.get(left_index)
            right_gold = gold_lines.get(right_index)
            if left_gold is None or right_gold is None:
                continue
            row = {
                "left_index": left_index,
                "right_index": right_index,
                "line_gap": right_offset,
                "left_text": left["text"],
                "right_text": right["text"],
                "overlap_ratio": round(overlap, 3),
                "predicted_left_end": round(float(left["end"]), 3),
                "predicted_right_start": round(float(right["start"]), 3),
                "gold_left_end": round(float(left_gold["end"]), 3),
                "gold_right_start": round(float(right_gold["start"]), 3),
                "left_end_error_sec": round(
                    float(left["end"]) - float(left_gold["end"]), 3
                ),
                "right_start_error_sec": round(
                    float(right["start"]) - float(right_gold["start"]), 3
                ),
            }
            if right_offset == 1:
                row["boundary_drift_sec"] = round(
                    float(left["end"]) - float(left_gold["end"]), 3
                )
            else:
                middle = timing_lines[idx + 1]
                middle_index = int(middle.get("index", 0) or 0)
                middle_gold = gold_lines.get(middle_index)
                if middle_gold is not None:
                    row["middle_index"] = middle_index
                    row["middle_text"] = middle["text"]
                    row["predicted_middle_end"] = round(float(middle["end"]), 3)
                    row["predicted_middle_start"] = round(float(middle["start"]), 3)
                    row["gold_middle_end"] = round(float(middle_gold["end"]), 3)
                    row["gold_middle_start"] = round(float(middle_gold["start"]), 3)
                    row["middle_end_error_sec"] = round(
                        float(middle["end"]) - float(middle_gold["end"]), 3
                    )
                    row["middle_start_error_sec"] = round(
                        float(middle["start"]) - float(middle_gold["start"]), 3
                    )
            rows.append(row)

    return {
        "timing_path": str(timing_path),
        "gold_path": str(gold_path),
        "pair_count": len(rows),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--max-line-gap", type=int, default=2)
    args = parser.parse_args()
    print(
        json.dumps(
            analyze(
                timing_path=args.timing,
                gold_path=args.gold,
                max_line_gap=args.max_line_gap,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
