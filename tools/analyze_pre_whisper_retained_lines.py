#!/usr/bin/env python3
"""Find lines whose final timing still matches pre-whisper timing despite gold miss."""

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
    max_retained_delta_sec: float = 0.1,
    min_gold_start_delta_sec: float = 0.5,
    min_gold_end_delta_sec: float = 0.5,
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
        gold_start = float(gold_line["start"])
        gold_end = float(gold_line["end"])
        retained_start_delta = abs(final_start - pre_start)
        retained_end_delta = abs(final_end - pre_end)
        gold_start_delta = abs(final_start - gold_start)
        gold_end_delta = abs(final_end - gold_end)
        if retained_start_delta > max_retained_delta_sec:
            continue
        if retained_end_delta > max_retained_delta_sec:
            continue
        if (
            gold_start_delta < min_gold_start_delta_sec
            and gold_end_delta < min_gold_end_delta_sec
        ):
            continue
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
