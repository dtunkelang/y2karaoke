"""Scan a benchmark report for repeated-line start harmonization opportunities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools import simulate_repeated_line_start_harmonization as sim_tool


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _song_label(song: dict[str, Any]) -> str:
    return f"{song.get('artist', '')} - {song.get('title', '')}".strip(" -")


def _analyze_song(
    song: dict[str, Any], *, min_start_error_span: float
) -> dict[str, Any]:
    report = _load_json(Path(song["report_path"]))
    gold = _load_json(Path(song["gold_path"]))
    payload = sim_tool._simulate(
        report=report,
        gold=gold,
        min_start_error_span=min_start_error_span,
    )
    return {
        "song": _song_label(song),
        "current_start_mean": payload["current_start_mean"],
        "simulated_start_mean": payload["simulated_start_mean"],
        "improvement": payload["current_start_mean"] - payload["simulated_start_mean"],
        "changed_lines": [
            line
            for line in payload["lines"]
            if line["current_start"] != line["simulated_start"]
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_report", help="Benchmark report JSON path")
    parser.add_argument("--match", help="Optional song substring filter")
    parser.add_argument(
        "--min-start-error-span",
        type=float,
        default=0.3,
        help="Minimum repeated-line start error span to harmonize",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    args = parser.parse_args()

    report = _load_json(Path(args.benchmark_report))
    rows: list[dict[str, Any]] = []
    for song in report.get("songs", []):
        label = _song_label(song)
        if args.match and args.match.lower() not in label.lower():
            continue
        rows.append(_analyze_song(song, min_start_error_span=args.min_start_error_span))
    rows.sort(key=lambda row: (-float(row["improvement"]), row["song"]))

    if args.json:
        print(json.dumps({"songs": rows}, indent=2))
        return 0

    for row in rows:
        print(
            f"{row['song']}: {row['current_start_mean']:.3f} -> "
            f"{row['simulated_start_mean']:.3f} "
            f"(improvement {row['improvement']:.3f})"
        )
        for line in row["changed_lines"]:
            print(
                f"  line {line['index']}: {line['current_start_error']:.3f} -> "
                f"{line['simulated_start_error']:.3f} ({line['text']})"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
