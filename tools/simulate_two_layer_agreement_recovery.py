#!/usr/bin/env python3
"""Simulate combined clipped-anchor and local-window agreement recovery."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any, cast

clip_tool = cast(
    Any, importlib.import_module("tools.simulate_clipped_agreement_recovery")
)
window_tool = cast(
    Any, importlib.import_module("tools.analyze_anchor_outside_window_recovery")
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def analyze(
    timing_report: dict[str, Any],
    *,
    min_text_similarity: float = 0.58,
    min_token_overlap: float = 0.5,
    min_line_words: int = 6,
    min_anchor_surplus_words: int = 15,
) -> dict[str, Any]:
    clipped = clip_tool.analyze(
        timing_report,
        min_text_similarity=min_text_similarity,
        min_token_overlap=min_token_overlap,
        min_line_words=min_line_words,
        min_anchor_surplus_words=min_anchor_surplus_words,
    )
    outside = window_tool.analyze(
        timing_report,
        min_text_similarity=min_text_similarity,
        min_token_overlap=min_token_overlap,
    )
    outside_rows = {
        int(row["line_index"]): row
        for row in outside.get("rows", [])
        if isinstance(row, dict)
    }

    rows: list[dict[str, Any]] = []
    baseline_eligible = int(clipped.get("baseline_eligible_lines", 0) or 0)
    baseline_matched = int(clipped.get("baseline_matched_lines", 0) or 0)
    recovered = 0
    extra_eligible = 0
    for row in clipped.get("rows", []) or []:
        if not isinstance(row, dict):
            continue
        line_index = int(row.get("line_index", 0) or 0)
        resolution = "baseline_match" if row.get("baseline_matched") else "unresolved"
        if row.get("recovered_by_clipping"):
            resolution = "recovered_by_clipping"
        else:
            outside_row = outside_rows.get(line_index, {})
            if row.get(
                "baseline_skip_reason"
            ) == "anchor_outside_window" and outside_row.get("candidate_would_match"):
                resolution = "recovered_by_window_phrase"
                extra_eligible += 1
        if resolution != "unresolved" and resolution != "baseline_match":
            recovered += 1
        rows.append(
            {
                "line_index": line_index,
                "text": str(row.get("text", "")),
                "baseline_skip_reason": row.get("baseline_skip_reason"),
                "resolution": resolution,
            }
        )

    adjusted_matched = baseline_matched + recovered
    adjusted_eligible = baseline_eligible + extra_eligible
    adjusted_coverage = (
        adjusted_matched / adjusted_eligible if adjusted_eligible else 0.0
    )
    return {
        "title": timing_report.get("title"),
        "artist": timing_report.get("artist"),
        "baseline_eligible_lines": baseline_eligible,
        "baseline_matched_lines": baseline_matched,
        "baseline_coverage_ratio": round(
            baseline_matched / baseline_eligible if baseline_eligible else 0.0, 4
        ),
        "adjusted_eligible_lines": adjusted_eligible,
        "adjusted_matched_lines": adjusted_matched,
        "adjusted_coverage_ratio": round(adjusted_coverage, 4),
        "recovered_lines": recovered,
        "min_line_words": min_line_words,
        "min_anchor_surplus_words": min_anchor_surplus_words,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("timing_report", help="Timing report JSON path")
    parser.add_argument(
        "--min-text-similarity",
        type=float,
        default=0.58,
        help="Agreement minimum text similarity",
    )
    parser.add_argument(
        "--min-token-overlap",
        type=float,
        default=0.5,
        help="Agreement minimum token overlap",
    )
    parser.add_argument(
        "--min-line-words",
        type=int,
        default=6,
        help="Guard for clipped-anchor recovery",
    )
    parser.add_argument(
        "--min-anchor-surplus-words",
        type=int,
        default=15,
        help="Guard for clipped-anchor recovery",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    args = parser.parse_args()

    payload = analyze(
        _load_json(Path(args.timing_report)),
        min_text_similarity=float(args.min_text_similarity),
        min_token_overlap=float(args.min_token_overlap),
        min_line_words=int(args.min_line_words),
        min_anchor_surplus_words=int(args.min_anchor_surplus_words),
    )
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    label = " - ".join(
        part for part in [payload.get("artist"), payload.get("title")] if part
    )
    print(label)
    print(
        f"baseline coverage: {int(payload['baseline_matched_lines'])}/"
        f"{int(payload['baseline_eligible_lines'])} "
        f"({float(payload['baseline_coverage_ratio']):.3f})"
    )
    print(
        f"adjusted coverage: {int(payload['adjusted_matched_lines'])}/"
        f"{int(payload['adjusted_eligible_lines'])} "
        f"({float(payload['adjusted_coverage_ratio']):.3f})"
    )
    for row in payload.get("rows", []) or []:
        if row.get("resolution") in {"baseline_match", "unresolved"}:
            continue
        print(f"line {int(row['line_index']):02d} {row['resolution']} {row['text']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
