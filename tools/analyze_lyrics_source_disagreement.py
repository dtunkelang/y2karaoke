#!/usr/bin/env python3
"""Inspect timed-lyrics source disagreement for a song."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from y2karaoke.core.components.alignment.timing_evaluator_comparison import (
    analyze_source_disagreement,
    compare_sources,
    select_best_source,
)
from y2karaoke.core.components.lyrics.sync import fetch_from_all_sources


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch all available timed-lyrics sources, summarize disagreement, "
            "and optionally compare them against audio."
        )
    )
    parser.add_argument("--title", required=True, help="Song title")
    parser.add_argument("--artist", required=True, help="Artist name")
    parser.add_argument("--vocals-path", help="Path to vocals audio for source scoring")
    parser.add_argument(
        "--target-duration",
        type=int,
        help="Expected track duration in seconds for best-source tiebreaks",
    )
    parser.add_argument("--json-out", type=Path, help="Optional path to write JSON")
    return parser.parse_args()


def _serialize_reports(reports: dict[str, Any]) -> dict[str, dict[str, Any]]:
    serialized: dict[str, dict[str, Any]] = {}
    for source_name, report in reports.items():
        serialized[source_name] = {
            "overall_score": round(float(report.overall_score), 4),
            "line_alignment_score": round(float(report.line_alignment_score), 4),
            "pause_alignment_score": round(float(report.pause_alignment_score), 4),
            "summary": report.summary,
        }
    return serialized


def main() -> int:
    args = _parse_args()
    sources = fetch_from_all_sources(args.title, args.artist)
    disagreement = analyze_source_disagreement(args.title, args.artist, sources)

    payload: dict[str, Any] = {
        "title": args.title,
        "artist": args.artist,
        "disagreement": disagreement,
    }

    print(f"Song: {args.artist} - {args.title}")
    print(f"Sources fetched: {disagreement['source_count']}")
    print(f"Comparable timed sources: {disagreement['comparable_source_count']}")
    print(f"Flagged disagreement: {'yes' if disagreement['flagged'] else 'no'}")
    if disagreement["reasons"]:
        print("Reasons: " + "; ".join(disagreement["reasons"]))
    print()

    for source in disagreement["sources"]:
        print(
            f"- {source['source_name']}: "
            f"duration={source['duration']} "
            f"lines={source['line_count']} "
            f"first={source['first_start']} "
            f"last={source['last_start']}"
        )

    if args.vocals_path:
        reports = compare_sources(
            args.title,
            args.artist,
            args.vocals_path,
            sources=sources,
        )
        payload["timing_reports"] = _serialize_reports(reports)
        lrc_text, best_source, report = select_best_source(
            args.title,
            args.artist,
            args.vocals_path,
            target_duration=args.target_duration,
            sources=sources,
        )
        payload["selected_best_source"] = {
            "source": best_source,
            "has_lrc": bool(lrc_text),
            "overall_score": (
                round(float(report.overall_score), 4) if report is not None else None
            ),
        }
        print()
        if best_source and report is not None:
            print(
                "Audio-scored best source: "
                f"{best_source} (score={report.overall_score:.2f})"
            )
        else:
            print("Audio-scored best source: none")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Saved JSON: {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
