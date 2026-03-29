#!/usr/bin/env python3
"""Find lines whose final timing keeps the pre-whisper start but compresses the end."""

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
    max_start_drift_sec: float = 0.1,
    min_end_compression_sec: float = 0.2,
    min_gold_end_improvement_sec: float = 0.1,
) -> dict[str, Any]:
    report = _load_json(report_path)
    gold = _load_json(gold_path)
    report_lines = report.get("lines", [])
    gold_lines = gold.get("lines", [])
    rows: list[dict[str, Any]] = []

    for report_line, gold_line in zip(report_lines, gold_lines):
        pre_start = float(report_line.get("pre_whisper_start", report_line["start"]))
        pre_end = float(report_line.get("pre_whisper_end", report_line["end"]))
        final_start = float(report_line["start"])
        final_end = float(report_line["end"])
        gold_end = float(gold_line["end"])
        start_drift = abs(final_start - pre_start)
        end_compression = pre_end - final_end
        final_gold_end_delta = abs(final_end - gold_end)
        pre_gold_end_delta = abs(pre_end - gold_end)
        gold_end_improvement = final_gold_end_delta - pre_gold_end_delta
        if start_drift > max_start_drift_sec:
            continue
        if end_compression < min_end_compression_sec:
            continue
        if gold_end_improvement < min_gold_end_improvement_sec:
            continue
        rows.append(
            {
                "line_index": int(report_line["index"]),
                "text": report_line["text"],
                "pre_whisper_start": round(pre_start, 3),
                "pre_whisper_end": round(pre_end, 3),
                "final_start": round(final_start, 3),
                "final_end": round(final_end, 3),
                "gold_end": round(gold_end, 3),
                "start_drift_sec": round(start_drift, 3),
                "end_compression_sec": round(end_compression, 3),
                "pre_gold_end_delta_sec": round(pre_gold_end_delta, 3),
                "final_gold_end_delta_sec": round(final_gold_end_delta, 3),
                "gold_end_regression_sec": round(gold_end_improvement, 3),
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
