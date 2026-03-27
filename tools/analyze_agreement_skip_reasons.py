#!/usr/bin/env python3
"""Explain benchmark agreement eligibility and skip reasons per line."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any, cast

suite = cast(Any, importlib.import_module("tools.run_benchmark_suite"))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _line_anchor_start(
    line: dict[str, Any],
    *,
    normalize_fn: Any,
) -> float | None:
    anchor_start = suite._select_agreement_anchor_start(
        line,
        normalize_fn=normalize_fn,
    )
    if not isinstance(anchor_start, (int, float)):
        return None
    return float(anchor_start)


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
    rows: list[dict[str, Any]] = []
    skip_reason_counts: dict[str, int] = {}

    for line in timing_report.get("lines", []) or []:
        if not isinstance(line, dict):
            continue
        evaluation = suite._evaluate_agreement_line(
            line,
            min_text_similarity,
            min_token_overlap,
            normalize_fn=normalize_fn,
        )
        anchor_start = _line_anchor_start(line, normalize_fn=normalize_fn)
        text_similarity = suite._agreement_text_similarity(
            line.get("text"),
            line.get("nearest_segment_start_text"),
            normalize_fn=normalize_fn,
        )
        token_overlap = suite._agreement_token_overlap(
            line.get("text"),
            line.get("nearest_segment_start_text"),
            normalize_fn=normalize_fn,
        )
        row = {
            "line_index": int(line.get("index", 0) or 0),
            "text": str(line.get("text", "")),
            "start": float(line.get("start", 0.0) or 0.0),
            "end": float(line.get("end", 0.0) or 0.0),
            "anchor_start": anchor_start,
            "anchor_text": str(line.get("nearest_segment_start_text", "")),
            "text_similarity": round(text_similarity, 4),
            "token_overlap": round(token_overlap, 4),
            "window_start": line.get("whisper_window_start"),
            "window_end": line.get("whisper_window_end"),
            "window_word_count": line.get("whisper_window_word_count"),
            "window_avg_prob": line.get("whisper_window_avg_prob"),
            "skip_reason": evaluation.get("skip_reason"),
            "eligible": bool(evaluation.get("eligible", False)),
            "anchor_start_delta": evaluation.get("anchor_start_delta"),
            "adaptive_rescue": bool(evaluation.get("adaptive_rescue", False)),
        }
        if row["skip_reason"]:
            skip_reason = str(row["skip_reason"])
            skip_reason_counts[skip_reason] = skip_reason_counts.get(skip_reason, 0) + 1
        rows.append(row)

    rows.sort(key=lambda row: int(row.get("line_index", 0)))
    return {
        "title": timing_report.get("title"),
        "artist": timing_report.get("artist"),
        "hook_boundary": hook_boundary,
        "min_text_similarity": min_text_similarity,
        "min_token_overlap": min_token_overlap,
        "skip_reason_counts": skip_reason_counts,
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
        "skip reasons:",
        ", ".join(
            f"{name}={count}"
            for name, count in sorted(payload.get("skip_reason_counts", {}).items())
        )
        or "none",
    )
    for row in payload.get("rows", []) or []:
        print(
            f"line {int(row['line_index']):02d} "
            f"{row['skip_reason'] or 'matched'} "
            f"sim={float(row['text_similarity']):.3f} "
            f"overlap={float(row['token_overlap']):.3f} "
            f"anchor={row['anchor_start']}"
        )
        print(f"  {row['text']}")
        if row.get("anchor_text"):
            print(f"  anchor: {row['anchor_text']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
