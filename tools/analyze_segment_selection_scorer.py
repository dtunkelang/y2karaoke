#!/usr/bin/env python3
"""Simulate segment-selection scorer penalties from trace output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [row for row in data.get("rows", []) if isinstance(row, dict)]


def _rescored_value(
    *,
    score: float,
    placeholder_ratio: float,
    bag_size: int,
    placeholder_penalty: float,
    placeholder_ratio_min: float,
    bag_size_min: int,
) -> float:
    adjusted = score
    if placeholder_ratio >= placeholder_ratio_min and bag_size >= bag_size_min:
        adjusted -= placeholder_penalty * placeholder_ratio
    return round(adjusted, 4)


def _collect_candidates(row: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for item in row.get("scores", []) or []:
        if not isinstance(item, dict):
            continue
        candidates.append(
            {
                "segment_index": item.get("segment_index"),
                "score": float(item.get("score", 0.0) or 0.0),
                "placeholder_ratio": float(item.get("placeholder_ratio", 0.0) or 0.0),
                "bag_size": int(item.get("bag_size", 0) or 0),
                "kind": "segment",
            }
        )
    for item in row.get("merged_candidates", []) or []:
        if not isinstance(item, dict):
            continue
        candidates.append(
            {
                "segment_index": item.get("chosen_segment"),
                "score": float(item.get("score", 0.0) or 0.0),
                "placeholder_ratio": float(item.get("placeholder_ratio", 0.0) or 0.0),
                "bag_size": int(item.get("bag_size", 0) or 0),
                "kind": "merged",
            }
        )
    return candidates


def _analyze_rows(
    rows: list[dict[str, Any]],
    *,
    placeholder_penalty: float,
    placeholder_ratio_min: float,
    bag_size_min: int,
) -> list[dict[str, Any]]:
    interesting: list[dict[str, Any]] = []
    for row in rows:
        original_seg = row.get("final_segment")
        if not isinstance(original_seg, int):
            continue
        candidates = _collect_candidates(row)
        if not candidates:
            continue
        rescored = []
        for item in candidates:
            rescored.append(
                {
                    **item,
                    "rescored": _rescored_value(
                        score=item["score"],
                        placeholder_ratio=item["placeholder_ratio"],
                        bag_size=item["bag_size"],
                        placeholder_penalty=placeholder_penalty,
                        placeholder_ratio_min=placeholder_ratio_min,
                        bag_size_min=bag_size_min,
                    ),
                }
            )
        best = max(rescored, key=lambda item: item["rescored"])
        if best["segment_index"] == original_seg:
            continue
        interesting.append(
            {
                "line_index": row.get("line_index"),
                "words": row.get("words", []),
                "original_segment": original_seg,
                "original_score": row.get("final_score"),
                "best_segment": best["segment_index"],
                "best_kind": best["kind"],
                "best_rescored": best["rescored"],
                "best_raw_score": round(best["score"], 4),
                "best_placeholder_ratio": round(best["placeholder_ratio"], 4),
                "best_bag_size": best["bag_size"],
            }
        )
    return interesting


def _format_markdown(
    *,
    trace_path: Path,
    rows: list[dict[str, Any]],
    placeholder_penalty: float,
    placeholder_ratio_min: float,
    bag_size_min: int,
) -> str:
    lines = [
        "# Segment Selection Scorer Analysis",
        "",
        f"- Trace: `{trace_path}`",
        f"- Placeholder penalty: `{placeholder_penalty}`",
        f"- Placeholder ratio min: `{placeholder_ratio_min}`",
        f"- Bag size min: `{bag_size_min}`",
        f"- Candidate line count: `{len(rows)}`",
        "",
    ]
    if not rows:
        lines.append("No line would change under this simulated scorer.")
        return "\n".join(lines) + "\n"
    for row in rows:
        lines.extend(
            [
                f"## Line {row['line_index']}",
                "",
                f"- Words: `{' '.join(row.get('words', []))}`",
                f"- Original segment/score: `{row.get('original_segment')}` / `{row.get('original_score')}`",
                f"- Simulated best segment: `{row.get('best_segment')}` via `{row.get('best_kind')}`",
                f"- Simulated rescored value: `{row.get('best_rescored')}`",
                f"- Raw score: `{row.get('best_raw_score')}`",
                f"- Placeholder ratio: `{row.get('best_placeholder_ratio')}`",
                f"- Bag size: `{row.get('best_bag_size')}`",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--placeholder-penalty", type=float, default=0.08)
    parser.add_argument("--placeholder-ratio-min", type=float, default=0.5)
    parser.add_argument("--bag-size-min", type=int, default=8)
    args = parser.parse_args()

    trace_path = Path(args.trace_json)
    rows = _load_rows(trace_path)
    interesting = _analyze_rows(
        rows,
        placeholder_penalty=args.placeholder_penalty,
        placeholder_ratio_min=args.placeholder_ratio_min,
        bag_size_min=args.bag_size_min,
    )
    output = _format_markdown(
        trace_path=trace_path,
        rows=interesting,
        placeholder_penalty=args.placeholder_penalty,
        placeholder_ratio_min=args.placeholder_ratio_min,
        bag_size_min=args.bag_size_min,
    )
    Path(args.output_md).write_text(output, encoding="utf-8")
    print(
        "segment_selection_scorer_analysis: OK\n"
        f"  trace={trace_path}\n"
        f"  output_md={args.output_md}\n"
        f"  changed_lines={len(interesting)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
