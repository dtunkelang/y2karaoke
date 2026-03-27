#!/usr/bin/env python3
"""Estimate whether clipping merged agreement anchors would recover matches."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any, cast

suite = cast(Any, importlib.import_module("tools.run_benchmark_suite"))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _best_clipped_anchor_text(
    line_text: Any,
    anchor_text: Any,
    *,
    normalize_fn: Any,
) -> str:
    line_tokens = normalize_fn(line_text).split()
    anchor_tokens = normalize_fn(anchor_text).split()
    if not line_tokens or not anchor_tokens:
        return ""
    if len(anchor_tokens) <= len(line_tokens):
        return " ".join(anchor_tokens)

    best_score: tuple[float, float, float] | None = None
    best_text = ""
    width = len(line_tokens)
    normalized_line = " ".join(line_tokens)
    for start_idx in range(0, len(anchor_tokens) - width + 1):
        candidate_tokens = anchor_tokens[start_idx : start_idx + width]
        candidate_text = " ".join(candidate_tokens)
        similarity = suite._agreement_text_similarity(
            normalized_line,
            candidate_text,
            normalize_fn=lambda value: str(value),
        )
        overlap = suite._agreement_token_overlap(
            normalized_line,
            candidate_text,
            normalize_fn=lambda value: str(value),
        )
        first_token_bonus = 1.0 if candidate_tokens[0] == line_tokens[0] else 0.0
        score = (overlap, similarity, first_token_bonus)
        if best_score is None or score > best_score:
            best_score = score
            best_text = candidate_text
    return best_text


def analyze(
    timing_report: dict[str, Any],
    *,
    min_text_similarity: float = 0.58,
    min_token_overlap: float = 0.5,
    hook_boundary: bool = False,
) -> dict[str, Any]:
    normalize_fn = (
        suite._normalize_agreement_text_hook_boundary
        if hook_boundary
        else suite._normalize_agreement_text
    )
    recovered_count = 0
    rows: list[dict[str, Any]] = []
    for line in timing_report.get("lines", []) or []:
        if not isinstance(line, dict):
            continue
        baseline = suite._evaluate_agreement_line(
            line,
            min_text_similarity,
            min_token_overlap,
            normalize_fn=normalize_fn,
        )
        clipped_anchor_text = _best_clipped_anchor_text(
            line.get("text"),
            line.get("nearest_segment_start_text"),
            normalize_fn=normalize_fn,
        )
        clipped_similarity = suite._agreement_text_similarity(
            line.get("text"),
            clipped_anchor_text,
            normalize_fn=normalize_fn,
        )
        clipped_overlap = suite._agreement_token_overlap(
            line.get("text"),
            clipped_anchor_text,
            normalize_fn=normalize_fn,
        )
        clipped_would_match = (
            clipped_similarity >= min_text_similarity
            and clipped_overlap >= min_token_overlap
        )
        if baseline.get("skip_reason") == "low_text_similarity" and clipped_would_match:
            recovered_count += 1
        rows.append(
            {
                "line_index": int(line.get("index", 0) or 0),
                "text": str(line.get("text", "")),
                "baseline_skip_reason": baseline.get("skip_reason"),
                "anchor_text": str(line.get("nearest_segment_start_text", "")),
                "baseline_text_similarity": round(
                    suite._agreement_text_similarity(
                        line.get("text"),
                        line.get("nearest_segment_start_text"),
                        normalize_fn=normalize_fn,
                    ),
                    4,
                ),
                "baseline_token_overlap": round(
                    suite._agreement_token_overlap(
                        line.get("text"),
                        line.get("nearest_segment_start_text"),
                        normalize_fn=normalize_fn,
                    ),
                    4,
                ),
                "clipped_anchor_text": clipped_anchor_text,
                "clipped_text_similarity": round(clipped_similarity, 4),
                "clipped_token_overlap": round(clipped_overlap, 4),
                "clipped_would_match": clipped_would_match,
            }
        )
    rows.sort(key=lambda row: int(row.get("line_index", 0)))
    return {
        "title": timing_report.get("title"),
        "artist": timing_report.get("artist"),
        "hook_boundary": hook_boundary,
        "min_text_similarity": min_text_similarity,
        "min_token_overlap": min_token_overlap,
        "recovered_low_text_similarity_lines": recovered_count,
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
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    args = parser.parse_args()

    payload = analyze(
        _load_json(Path(args.timing_report)),
        min_text_similarity=float(args.min_text_similarity),
        min_token_overlap=float(args.min_token_overlap),
        hook_boundary=bool(args.hook_boundary),
    )
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    label = " - ".join(
        part for part in [payload.get("artist"), payload.get("title")] if part
    )
    print(label)
    print(
        "recovered_low_text_similarity_lines:",
        payload.get("recovered_low_text_similarity_lines", 0),
    )
    for row in payload.get("rows", []) or []:
        if row.get("baseline_skip_reason") != "low_text_similarity":
            continue
        print(
            f"line {int(row['line_index']):02d} "
            f"baseline={float(row['baseline_text_similarity']):.3f} "
            f"clipped={float(row['clipped_text_similarity']):.3f} "
            f"match={'yes' if bool(row['clipped_would_match']) else 'no'}"
        )
        print(f"  {row['text']}")
        print(f"  clipped: {row['clipped_anchor_text']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
