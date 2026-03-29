#!/usr/bin/env python3
"""Find lines whose final start regressed earlier despite a later segment start."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def analyze_timing_report(
    *,
    report_path: Path,
    gold_path: Path,
    min_pre_whisper_regression_sec: float = 0.25,
    min_segment_gain_sec: float = 0.4,
    min_gold_gain_sec: float = 0.25,
) -> dict[str, Any]:
    report = _load_json(report_path)
    gold = _load_json(gold_path)
    gold_lines = {int(line["line_index"]): line for line in gold.get("lines", [])}
    rows: list[dict[str, Any]] = []

    for line in report.get("lines", []):
        line_index = int(line.get("index", 0) or 0)
        gold_line = gold_lines.get(line_index)
        if gold_line is None:
            continue
        final_start = float(line["start"])
        pre_whisper_start = float(line.get("pre_whisper_start", final_start))
        nearest_segment_start = line.get("nearest_segment_start")
        if nearest_segment_start is None:
            continue
        nearest_segment_start = float(nearest_segment_start)
        gold_start = float(gold_line["start"])
        pre_whisper_regression = pre_whisper_start - final_start
        segment_gain = nearest_segment_start - final_start
        gold_gain = gold_start - final_start
        if pre_whisper_regression < min_pre_whisper_regression_sec:
            continue
        if segment_gain < min_segment_gain_sec:
            continue
        if gold_gain < min_gold_gain_sec:
            continue
        rows.append(
            {
                "line_index": line_index,
                "text": line["text"],
                "final_start": round(final_start, 3),
                "pre_whisper_start": round(pre_whisper_start, 3),
                "nearest_segment_start": round(nearest_segment_start, 3),
                "gold_start": round(gold_start, 3),
                "pre_whisper_regression_sec": round(pre_whisper_regression, 3),
                "segment_gain_sec": round(segment_gain, 3),
                "gold_gain_sec": round(gold_gain, 3),
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
