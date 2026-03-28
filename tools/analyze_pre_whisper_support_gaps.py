#!/usr/bin/env python3
"""Analyze retained pre-whisper misses with weak local Whisper support."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _token_overlap_ratio(left: list[str], right: list[str]) -> float:
    if not left or not right:
        return 0.0
    left_set = set(left)
    right_set = set(right)
    overlap = len(left_set & right_set)
    return overlap / max(len(left_set), len(right_set))


def analyze_timing_report(
    *,
    report_path: Path,
    gold_path: Path,
    max_retained_delta_sec: float = 0.1,
    min_gold_start_delta_sec: float = 0.5,
    min_gold_end_delta_sec: float = 0.5,
    max_window_overlap_ratio: float = 0.25,
) -> dict[str, Any]:
    report = _load_json(report_path)
    gold = _load_json(gold_path)
    rows: list[dict[str, Any]] = []
    for report_line, gold_line in zip(report.get("lines", []), gold.get("lines", [])):
        final_start = float(report_line["start"])
        final_end = float(report_line["end"])
        pre_start = float(report_line.get("pre_whisper_start", final_start))
        pre_end = float(report_line.get("pre_whisper_end", final_end))
        if abs(final_start - pre_start) > max_retained_delta_sec:
            continue
        if abs(final_end - pre_end) > max_retained_delta_sec:
            continue
        gold_start = float(gold_line["start"])
        gold_end = float(gold_line["end"])
        gold_start_delta = abs(final_start - gold_start)
        gold_end_delta = abs(final_end - gold_end)
        if gold_start_delta < min_gold_start_delta_sec:
            continue
        if gold_end_delta < min_gold_end_delta_sec:
            continue
        line_tokens = _normalize_tokens(report_line["text"])
        window_tokens: list[str] = []
        for word in report_line.get("whisper_window_words", []):
            window_tokens.extend(_normalize_tokens(str(word.get("text", ""))))
        overlap_ratio = _token_overlap_ratio(line_tokens, window_tokens)
        if overlap_ratio > max_window_overlap_ratio:
            continue
        nearest_segment_end = float(
            report_line.get("nearest_segment_end", report_line.get("end", 0.0))
        )
        nearest_segment_start = float(
            report_line.get(
                "nearest_segment_end_start",
                report_line.get("nearest_segment_start", 0.0),
            )
        )
        rows.append(
            {
                "line_index": int(report_line["index"]),
                "text": report_line["text"],
                "pre_whisper_start": round(pre_start, 3),
                "pre_whisper_end": round(pre_end, 3),
                "final_start": round(final_start, 3),
                "final_end": round(final_end, 3),
                "gold_start": round(gold_start, 3),
                "gold_end": round(gold_end, 3),
                "gold_start_delta_sec": round(gold_start_delta, 3),
                "gold_end_delta_sec": round(gold_end_delta, 3),
                "window_overlap_ratio": round(overlap_ratio, 3),
                "whisper_window_word_count": int(
                    report_line.get("whisper_window_word_count", 0) or 0
                ),
                "nearest_segment_start": round(nearest_segment_start, 3),
                "nearest_segment_end": round(nearest_segment_end, 3),
            }
        )
    return {
        "report_path": str(report_path),
        "gold_path": str(gold_path),
        "candidate_count": len(rows),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            analyze_timing_report(report_path=args.report, gold_path=args.gold),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
