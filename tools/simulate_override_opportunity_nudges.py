"""Simulate start-only nudges from ranked override opportunities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from faster_whisper import WhisperModel

from tools import analyze_override_opportunities as override_tool


def _simulate_song(
    rows: list[dict[str, Any]],
    *,
    families: set[str],
) -> dict[str, Any]:
    current_errors: list[float] = []
    simulated_errors: list[float] = []
    lines: list[dict[str, Any]] = []
    song = rows[0]["song"] if rows else ""
    for row in rows:
        gold_start = row["gold_start"]
        current_start = row["current_start"]
        simulated_start = current_start
        if row["opportunity"] in families and row["fuzzy_estimated_start"] is not None:
            simulated_start = float(row["fuzzy_estimated_start"])
        current_err = override_tool._estimate_error(current_start, gold_start)
        simulated_err = override_tool._estimate_error(simulated_start, gold_start)
        if current_err is not None:
            current_errors.append(current_err)
        if simulated_err is not None:
            simulated_errors.append(simulated_err)
        lines.append(
            {
                "index": int(row["index"]),
                "text": str(row["text"]),
                "opportunity": str(row["opportunity"]),
                "current_start": current_start,
                "simulated_start": simulated_start,
                "gold_start": gold_start,
                "current_start_error": current_err,
                "simulated_start_error": simulated_err,
            }
        )
    return {
        "song": song,
        "current_start_mean": sum(current_errors) / max(1, len(current_errors)),
        "simulated_start_mean": sum(simulated_errors) / max(1, len(simulated_errors)),
        "lines": lines,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_report", help="Benchmark report JSON path")
    parser.add_argument("--match", help="Optional song substring filter")
    parser.add_argument(
        "--family",
        action="append",
        dest="families",
        help="Opportunity family to simulate; repeatable",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    args = parser.parse_args()

    report = override_tool._load_json(Path(args.benchmark_report))
    model = WhisperModel("large-v3", device="cpu", compute_type="int8")
    rows: list[dict[str, Any]] = []
    for song in report.get("songs", []):
        label = f"{song.get('artist', '')} - {song.get('title', '')}".strip(" -")
        if args.match and args.match.lower() not in label.lower():
            continue
        rows.extend(override_tool._analyze_song(song=song, model=model))
    families = set(args.families or ["fuzzy_span_candidate"])
    songs: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        songs.setdefault(str(row["song"]), []).append(row)
    payload = {
        "families": sorted(families),
        "songs": [
            _simulate_song(song_rows, families=families)
            for _song, song_rows in sorted(songs.items())
        ],
    }
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    for song in payload["songs"]:
        print(song["song"])
        print(
            f"  start mean: {song['current_start_mean']:.3f} -> "
            f"{song['simulated_start_mean']:.3f}"
        )
        for line in song["lines"]:
            if line["opportunity"] not in families:
                continue
            print(
                f"  line {line['index']}: "
                f"{line['current_start_error']:.3f} -> "
                f"{line['simulated_start_error']:.3f} "
                f"({line['text']})"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
