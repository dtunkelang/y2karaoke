#!/usr/bin/env python3
"""Estimate agreement coverage gain from clipped merged anchors."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any, cast

skip_tool = cast(Any, importlib.import_module("tools.analyze_agreement_skip_reasons"))
clip_tool = cast(
    Any, importlib.import_module("tools.analyze_agreement_anchor_clipping")
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def analyze(
    timing_report: dict[str, Any],
    *,
    min_text_similarity: float = 0.58,
    min_token_overlap: float = 0.5,
    hook_boundary: bool = False,
    min_line_words: int = 6,
    min_anchor_surplus_words: int = 15,
    min_anchor_words: int = 20,
) -> dict[str, Any]:
    baseline = skip_tool.analyze(
        timing_report,
        min_text_similarity=min_text_similarity,
        min_token_overlap=min_token_overlap,
        hook_boundary=hook_boundary,
    )
    clipping = clip_tool.analyze(
        timing_report,
        min_text_similarity=min_text_similarity,
        min_token_overlap=min_token_overlap,
        hook_boundary=hook_boundary,
    )
    clip_rows = {
        int(row["line_index"]): row
        for row in clipping.get("rows", [])
        if isinstance(row, dict)
    }
    rows: list[dict[str, Any]] = []
    baseline_eligible = 0
    baseline_matched = 0
    recovered_matches = 0

    for row in baseline.get("rows", []) or []:
        if not isinstance(row, dict):
            continue
        line_index = int(row.get("line_index", 0) or 0)
        eligible = bool(row.get("eligible", False))
        matched = eligible and not row.get("skip_reason")
        recovered = False
        if row.get("skip_reason") == "low_text_similarity":
            clip_row = clip_rows.get(line_index, {})
            line_words = len(
                skip_tool.suite._normalize_agreement_text(row.get("text", "")).split()
            )
            anchor_words = len(
                skip_tool.suite._normalize_agreement_text(
                    clip_row.get("anchor_text", "")
                ).split()
            )
            recovered = (
                bool(clip_row.get("clipped_would_match", False))
                and line_words >= min_line_words
                and anchor_words >= min_anchor_words
                and (anchor_words - line_words) >= min_anchor_surplus_words
            )
        if eligible:
            baseline_eligible += 1
        if matched:
            baseline_matched += 1
        if recovered:
            recovered_matches += 1
        rows.append(
            {
                "line_index": line_index,
                "text": str(row.get("text", "")),
                "baseline_skip_reason": row.get("skip_reason"),
                "baseline_eligible": eligible,
                "baseline_matched": matched,
                "recovered_by_clipping": recovered,
            }
        )

    adjusted_matched = baseline_matched + recovered_matches
    adjusted_coverage = (
        adjusted_matched / baseline_eligible if baseline_eligible else 0.0
    )
    baseline_coverage = (
        baseline_matched / baseline_eligible if baseline_eligible else 0.0
    )
    return {
        "title": timing_report.get("title"),
        "artist": timing_report.get("artist"),
        "baseline_eligible_lines": baseline_eligible,
        "baseline_matched_lines": baseline_matched,
        "baseline_coverage_ratio": round(baseline_coverage, 4),
        "recovered_lines": recovered_matches,
        "adjusted_matched_lines": adjusted_matched,
        "adjusted_coverage_ratio": round(adjusted_coverage, 4),
        "min_line_words": min_line_words,
        "min_anchor_surplus_words": min_anchor_surplus_words,
        "min_anchor_words": min_anchor_words,
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
        "--hook-boundary",
        action="store_true",
        help="Use hook-boundary agreement normalization",
    )
    parser.add_argument(
        "--min-line-words",
        type=int,
        default=0,
        help="Only count clipped recoveries for lines at or above this word count",
    )
    parser.add_argument(
        "--min-anchor-surplus-words",
        type=int,
        default=15,
        help="Only count clipped recoveries when anchor is this many words longer",
    )
    parser.add_argument(
        "--min-anchor-words",
        type=int,
        default=20,
        help="Only count clipped recoveries when anchor has at least this many words",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    args = parser.parse_args()

    payload = analyze(
        _load_json(Path(args.timing_report)),
        min_text_similarity=float(args.min_text_similarity),
        min_token_overlap=float(args.min_token_overlap),
        hook_boundary=bool(args.hook_boundary),
        min_line_words=int(args.min_line_words),
        min_anchor_surplus_words=int(args.min_anchor_surplus_words),
        min_anchor_words=int(args.min_anchor_words),
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
        f"{int(payload['baseline_eligible_lines'])} "
        f"({float(payload['adjusted_coverage_ratio']):.3f})"
    )
    for row in payload.get("rows", []) or []:
        if not row.get("recovered_by_clipping"):
            continue
        print(f"line {int(row['line_index']):02d} recovered {row['text']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
