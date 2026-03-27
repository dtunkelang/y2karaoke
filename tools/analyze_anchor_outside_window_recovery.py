#!/usr/bin/env python3
"""Analyze whether anchor-outside-window lines have strong local window phrases."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any, cast

suite = cast(Any, importlib.import_module("tools.run_benchmark_suite"))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _best_window_phrase(line: dict[str, Any]) -> dict[str, Any]:
    line_tokens = suite._normalize_agreement_text(line.get("text")).split()
    window_tokens = [token for token, _ in suite._iter_agreement_window_tokens(line)]
    if not line_tokens or not window_tokens:
        return {
            "candidate_text": "",
            "candidate_overlap": 0.0,
            "candidate_similarity": 0.0,
        }

    best_overlap = 0.0
    best_similarity = 0.0
    best_text = ""
    normalized_line = " ".join(line_tokens)
    width = min(len(line_tokens), len(window_tokens))
    for candidate_width in range(max(1, width - 2), width + 1):
        for start_idx in range(0, len(window_tokens) - candidate_width + 1):
            candidate_text = " ".join(
                window_tokens[start_idx : start_idx + candidate_width]
            )
            overlap = suite._agreement_token_overlap(
                normalized_line,
                candidate_text,
                normalize_fn=lambda value: str(value),
            )
            similarity = suite._agreement_text_similarity(
                normalized_line,
                candidate_text,
                normalize_fn=lambda value: str(value),
            )
            score = (overlap, similarity, -candidate_width)
            best_score = (best_overlap, best_similarity, -len(best_text.split()))
            if score > best_score:
                best_overlap = overlap
                best_similarity = similarity
                best_text = candidate_text
    return {
        "candidate_text": best_text,
        "candidate_overlap": round(best_overlap, 4),
        "candidate_similarity": round(best_similarity, 4),
    }


def analyze(
    timing_report: dict[str, Any],
    *,
    min_text_similarity: float = 0.58,
    min_token_overlap: float = 0.5,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    recoverable = 0
    for line in timing_report.get("lines", []) or []:
        if not isinstance(line, dict):
            continue
        evaluation = suite._evaluate_agreement_line(
            line,
            min_text_similarity,
            min_token_overlap,
        )
        if evaluation.get("skip_reason") != "anchor_outside_window":
            continue
        candidate = _best_window_phrase(line)
        would_match = (
            float(candidate["candidate_similarity"]) >= min_text_similarity
            and float(candidate["candidate_overlap"]) >= min_token_overlap
        )
        if would_match:
            recoverable += 1
        rows.append(
            {
                "line_index": int(line.get("index", 0) or 0),
                "text": str(line.get("text", "")),
                "anchor_start": line.get("nearest_segment_start"),
                "window_start": line.get("whisper_window_start"),
                "window_end": line.get("whisper_window_end"),
                "candidate_text": candidate["candidate_text"],
                "candidate_overlap": candidate["candidate_overlap"],
                "candidate_similarity": candidate["candidate_similarity"],
                "candidate_would_match": would_match,
            }
        )
    rows.sort(key=lambda row: int(row["line_index"]))
    return {
        "title": timing_report.get("title"),
        "artist": timing_report.get("artist"),
        "recoverable_anchor_outside_window_lines": recoverable,
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
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    args = parser.parse_args()

    payload = analyze(
        _load_json(Path(args.timing_report)),
        min_text_similarity=float(args.min_text_similarity),
        min_token_overlap=float(args.min_token_overlap),
    )
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    label = " - ".join(
        part for part in [payload.get("artist"), payload.get("title")] if part
    )
    print(label)
    print(
        "recoverable_anchor_outside_window_lines:",
        payload.get("recoverable_anchor_outside_window_lines", 0),
    )
    for row in payload.get("rows", []) or []:
        print(
            f"line {int(row['line_index']):02d} "
            f"match={'yes' if bool(row['candidate_would_match']) else 'no'} "
            f"overlap={float(row['candidate_overlap']):.3f} "
            f"sim={float(row['candidate_similarity']):.3f}"
        )
        print(f"  {row['text']}")
        print(f"  candidate: {row['candidate_text']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
