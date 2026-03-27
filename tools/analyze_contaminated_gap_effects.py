"""Join interstitial contamination findings with line timing errors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools import analyze_interstitial_vocal_contamination as contamination_tool


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _safe_error(current: float | None, gold: float | None) -> float | None:
    if current is None or gold is None:
        return None
    return round(float(current) - float(gold), 3)


def _classify_effect(
    *,
    gap_classification: str,
    prev_end_error: float | None,
    next_start_error: float | None,
    threshold: float = 0.25,
) -> str:
    if gap_classification not in {
        "echo_fragment",
        "hallucinated_interstitial",
        "unclear_interstitial",
    }:
        return "non_contaminating_gap"
    if prev_end_error is None or next_start_error is None:
        return "insufficient_timing_data"
    prev_mag = abs(prev_end_error)
    next_mag = abs(next_start_error)
    if prev_end_error < -threshold and prev_mag > next_mag + 0.1:
        return "prev_line_truncated"
    if next_start_error > threshold and next_mag > prev_mag + 0.1:
        return "next_line_delayed"
    if prev_end_error < -threshold and next_start_error > threshold:
        return "both_sides_shifted_apart"
    return "mixed_or_small_effect"


def _analyze(
    *,
    gold_json: Path,
    timing_report_json: Path,
) -> dict[str, Any]:
    gold = _load_json(gold_json)
    report = _load_json(timing_report_json)
    contamination = contamination_tool.analyze_gold_json(gold_json)

    gold_lines = gold.get("lines", [])
    report_lines = report.get("lines", [])
    rows: list[dict[str, Any]] = []
    for gap in contamination["gaps"]:
        prev_idx = int(gap["gap_index"]) - 1
        next_idx = prev_idx + 1
        if prev_idx < 0 or next_idx >= len(gold_lines) or next_idx >= len(report_lines):
            continue
        gold_prev = gold_lines[prev_idx]
        gold_next = gold_lines[next_idx]
        report_prev = report_lines[prev_idx]
        report_next = report_lines[next_idx]
        prev_end_error = _safe_error(report_prev.get("end"), gold_prev.get("end"))
        next_start_error = _safe_error(report_next.get("start"), gold_next.get("start"))
        rows.append(
            {
                "gap_index": int(gap["gap_index"]),
                "classification": str(gap["classification"]),
                "aggressive_text": str(gap["aggressive_text"]),
                "prev_index": prev_idx + 1,
                "prev_text": str(gold_prev["text"]),
                "prev_current_end": report_prev.get("end"),
                "prev_gold_end": gold_prev.get("end"),
                "prev_end_error": prev_end_error,
                "next_index": next_idx + 1,
                "next_text": str(gold_next["text"]),
                "next_current_start": report_next.get("start"),
                "next_gold_start": gold_next.get("start"),
                "next_start_error": next_start_error,
                "effect": _classify_effect(
                    gap_classification=str(gap["classification"]),
                    prev_end_error=prev_end_error,
                    next_start_error=next_start_error,
                ),
            }
        )
    return {
        "gold_json": str(gold_json.resolve()),
        "timing_report_json": str(timing_report_json.resolve()),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gold_json", help="Gold timing JSON path")
    parser.add_argument("timing_report_json", help="Timing report JSON path")
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    args = parser.parse_args()

    payload = _analyze(
        gold_json=Path(args.gold_json),
        timing_report_json=Path(args.timing_report_json),
    )
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(payload["gold_json"])
    for row in payload["rows"]:
        print(f"gap {row['gap_index']}: {row['classification']} -> {row['effect']}")
        print(
            f"  prev line {row['prev_index']} end: "
            f"{row['prev_current_end']} vs gold {row['prev_gold_end']} "
            f"({row['prev_end_error']:+.3f}s)"
        )
        print(
            f"  next line {row['next_index']} start: "
            f"{row['next_current_start']} vs gold {row['next_gold_start']} "
            f"({row['next_start_error']:+.3f}s)"
        )
        if row["aggressive_text"]:
            print(f"  aggressive gap text: {row['aggressive_text']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
