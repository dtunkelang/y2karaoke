#!/usr/bin/env python3
"""Summarize contaminated inter-line gap effects across a benchmark report pack."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import re
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _resolve_report(path: Path) -> Path:
    return path / "benchmark_report.json" if path.is_dir() else path


def _load_report(path: Path) -> dict[str, Any]:
    return json.loads(_resolve_report(path).read_text(encoding="utf-8"))


def _safe_int(raw: Any) -> int:
    return int(raw or 0)


def _load_gap_tool() -> Any:
    return importlib.import_module("tools.analyze_contaminated_gap_effects")


def _match_song(song_label: str, pattern: re.Pattern[str] | None) -> bool:
    if pattern is None:
        return True
    return bool(pattern.search(song_label))


def analyze(
    report_doc: dict[str, Any],
    *,
    match: str | None = None,
) -> dict[str, Any]:
    pattern = re.compile(match, re.IGNORECASE) if match else None
    rows: list[dict[str, Any]] = []
    gap_tool = _load_gap_tool()
    totals: dict[str, int] = {
        "gaps_total": 0,
        "prev_line_truncated_total": 0,
        "next_line_delayed_total": 0,
        "both_sides_shifted_apart_total": 0,
        "mixed_or_small_effect_total": 0,
        "insufficient_timing_data_total": 0,
    }
    for song in report_doc.get("songs", []) or []:
        if not isinstance(song, dict):
            continue
        artist = str(song.get("artist", "")).strip()
        title = str(song.get("title", "")).strip()
        label = " - ".join(part for part in [artist, title] if part)
        if not _match_song(label, pattern):
            continue
        report_path = song.get("report_path")
        gold_path = song.get("gold_path")
        if not isinstance(report_path, str) or not isinstance(gold_path, str):
            continue
        payload = gap_tool._analyze(
            gold_json=Path(gold_path),
            timing_report_json=Path(report_path),
        )
        gap_rows = payload.get("rows", []) or []
        effect_counts: dict[str, int] = {}
        for row in gap_rows:
            if not isinstance(row, dict):
                continue
            effect = str(row.get("effect", ""))
            effect_counts[effect] = effect_counts.get(effect, 0) + 1
        gap_count = len(gap_rows)
        prev_line_truncated = effect_counts.get("prev_line_truncated", 0)
        next_line_delayed = effect_counts.get("next_line_delayed", 0)
        both_sides = effect_counts.get("both_sides_shifted_apart", 0)
        mixed = effect_counts.get("mixed_or_small_effect", 0)
        insufficient = effect_counts.get("insufficient_timing_data", 0)
        totals["gaps_total"] += gap_count
        totals["prev_line_truncated_total"] += prev_line_truncated
        totals["next_line_delayed_total"] += next_line_delayed
        totals["both_sides_shifted_apart_total"] += both_sides
        totals["mixed_or_small_effect_total"] += mixed
        totals["insufficient_timing_data_total"] += insufficient
        rows.append(
            {
                "song": label,
                "gold_path": gold_path,
                "report_path": report_path,
                "gap_count": gap_count,
                "prev_line_truncated_count": prev_line_truncated,
                "next_line_delayed_count": next_line_delayed,
                "both_sides_shifted_apart_count": both_sides,
                "mixed_or_small_effect_count": mixed,
                "insufficient_timing_data_count": insufficient,
                "effects": effect_counts,
            }
        )
    rows.sort(
        key=lambda row: (
            -_safe_int(row.get("prev_line_truncated_count")),
            -_safe_int(row.get("gap_count")),
            str(row.get("song", "")),
        )
    )
    return {
        **totals,
        "songs_analyzed": len(rows),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", help="Benchmark report JSON or run directory")
    parser.add_argument(
        "--match",
        default=None,
        help="Optional regex to filter songs by 'Artist - Title'",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    args = parser.parse_args()

    payload = analyze(_load_report(Path(args.report)), match=args.match)
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(
        "songs analyzed:",
        f"{int(payload.get('songs_analyzed', 0) or 0)}",
    )
    print(
        "gaps:",
        f"{int(payload.get('gaps_total', 0) or 0)}",
        "prev-truncated:",
        f"{int(payload.get('prev_line_truncated_total', 0) or 0)}",
        "next-delayed:",
        f"{int(payload.get('next_line_delayed_total', 0) or 0)}",
    )
    for row in payload.get("rows", []) or []:
        print(
            f"{row['song']}: "
            f"gaps={row['gap_count']} "
            f"prev_truncated={row['prev_line_truncated_count']} "
            f"next_delayed={row['next_line_delayed_count']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
