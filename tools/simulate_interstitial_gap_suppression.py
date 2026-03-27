"""Simulate source-timing fallback after contaminated inter-line gaps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _gold_by_index(gold: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {idx + 1: line for idx, line in enumerate(gold.get("lines", []))}


def _contaminated_next_lines(
    contamination: dict[str, Any], *, classes: set[str]
) -> set[int]:
    return {
        int(gap["gap_index"]) + 1
        for gap in contamination.get("gaps", [])
        if str(gap.get("classification")) in classes
    }


def _simulate(
    *,
    timing_report: dict[str, Any],
    gold: dict[str, Any],
    contamination: dict[str, Any],
    contaminated_classes: set[str],
) -> dict[str, Any]:
    gold_map = _gold_by_index(gold)
    contaminated_next = _contaminated_next_lines(
        contamination, classes=contaminated_classes
    )
    lines: list[dict[str, Any]] = []
    current_errors: list[float] = []
    simulated_errors: list[float] = []
    for line in timing_report.get("lines", []):
        idx = int(line["index"])
        gold_line = gold_map.get(idx)
        if gold_line is None:
            continue
        current_start = float(line["start"])
        pre_start_raw = line.get("pre_whisper_start")
        pre_start = (
            float(pre_start_raw) if isinstance(pre_start_raw, (int, float)) else None
        )
        simulated_start = (
            pre_start
            if idx in contaminated_next and pre_start is not None
            else current_start
        )
        gold_start = float(gold_line["start"])
        current_error = abs(current_start - gold_start)
        simulated_error = abs(simulated_start - gold_start)
        current_errors.append(current_error)
        simulated_errors.append(simulated_error)
        lines.append(
            {
                "index": idx,
                "text": str(line["text"]),
                "contaminated_gap_predecessor": idx in contaminated_next,
                "current_start": current_start,
                "pre_start": pre_start,
                "simulated_start": simulated_start,
                "gold_start": gold_start,
                "current_start_error": current_error,
                "simulated_start_error": simulated_error,
            }
        )
    return {
        "current_start_mean": sum(current_errors) / max(1, len(current_errors)),
        "simulated_start_mean": sum(simulated_errors) / max(1, len(simulated_errors)),
        "lines": lines,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("timing_report", help="Timing report JSON path")
    parser.add_argument("gold_json", help="Gold timing JSON path")
    parser.add_argument(
        "contamination_json", help="Interstitial contamination JSON path"
    )
    parser.add_argument(
        "--class",
        action="append",
        dest="classes",
        help="Contamination classes to suppress; repeatable",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    args = parser.parse_args()

    result = _simulate(
        timing_report=_load_json(Path(args.timing_report)),
        gold=_load_json(Path(args.gold_json)),
        contamination=_load_json(Path(args.contamination_json)),
        contaminated_classes=set(
            args.classes or ["echo_fragment", "hallucinated_interstitial"]
        ),
    )
    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    print(
        f"start mean: {result['current_start_mean']:.3f} -> "
        f"{result['simulated_start_mean']:.3f}"
    )
    for line in result["lines"]:
        if not line["contaminated_gap_predecessor"]:
            continue
        print(
            f"line {line['index']}: {line['current_start_error']:.3f} -> "
            f"{line['simulated_start_error']:.3f} ({line['text']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
