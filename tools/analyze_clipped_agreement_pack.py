#!/usr/bin/env python3
"""Summarize clipped-anchor agreement recovery across a benchmark report pack."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any, cast

sim_tool = cast(
    Any, importlib.import_module("tools.simulate_clipped_agreement_recovery")
)


def _resolve_report(path: Path) -> Path:
    return path / "benchmark_report.json" if path.is_dir() else path


def _load_report(path: Path) -> dict[str, Any]:
    return json.loads(_resolve_report(path).read_text(encoding="utf-8"))


def analyze(
    report_doc: dict[str, Any],
    *,
    min_text_similarity: float = 0.58,
    min_token_overlap: float = 0.5,
    min_line_words: int = 0,
    min_anchor_surplus_words: int = 0,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    total_baseline_eligible = 0
    total_baseline_matched = 0
    total_adjusted_matched = 0
    total_recovered = 0

    for song in report_doc.get("songs", []) or []:
        if not isinstance(song, dict):
            continue
        report_path = song.get("report_path")
        if not isinstance(report_path, str):
            continue
        timing_report = json.loads(Path(report_path).read_text(encoding="utf-8"))
        payload = sim_tool.analyze(
            timing_report,
            min_text_similarity=min_text_similarity,
            min_token_overlap=min_token_overlap,
            min_line_words=min_line_words,
            min_anchor_surplus_words=min_anchor_surplus_words,
        )
        baseline_eligible = int(payload.get("baseline_eligible_lines", 0) or 0)
        baseline_matched = int(payload.get("baseline_matched_lines", 0) or 0)
        adjusted_matched = int(payload.get("adjusted_matched_lines", 0) or 0)
        recovered = int(payload.get("recovered_lines", 0) or 0)
        total_baseline_eligible += baseline_eligible
        total_baseline_matched += baseline_matched
        total_adjusted_matched += adjusted_matched
        total_recovered += recovered
        rows.append(
            {
                "song": f"{song.get('artist', '')} - {song.get('title', '')}".strip(
                    " -"
                ),
                "report_path": report_path,
                "baseline_eligible_lines": baseline_eligible,
                "baseline_matched_lines": baseline_matched,
                "adjusted_matched_lines": adjusted_matched,
                "recovered_lines": recovered,
                "baseline_coverage_ratio": float(
                    payload.get("baseline_coverage_ratio", 0.0) or 0.0
                ),
                "adjusted_coverage_ratio": float(
                    payload.get("adjusted_coverage_ratio", 0.0) or 0.0
                ),
            }
        )

    rows.sort(key=lambda row: (-int(row["recovered_lines"]), str(row["song"])))
    return {
        "baseline_eligible_lines_total": total_baseline_eligible,
        "baseline_matched_lines_total": total_baseline_matched,
        "adjusted_matched_lines_total": total_adjusted_matched,
        "recovered_lines_total": total_recovered,
        "min_line_words": min_line_words,
        "min_anchor_surplus_words": min_anchor_surplus_words,
        "baseline_coverage_ratio_total": round(
            (
                total_baseline_matched / total_baseline_eligible
                if total_baseline_eligible
                else 0.0
            ),
            4,
        ),
        "adjusted_coverage_ratio_total": round(
            (
                total_adjusted_matched / total_baseline_eligible
                if total_baseline_eligible
                else 0.0
            ),
            4,
        ),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", help="Benchmark report JSON or run directory")
    parser.add_argument(
        "--min-text-similarity",
        type=float,
        default=0.58,
        help="Agreement minimum text similarity",
    )
    parser.add_argument(
        "--min-token-overlap",
        type=float,
        default=0.5,
        help="Agreement minimum token overlap",
    )
    parser.add_argument(
        "--min-line-words",
        type=int,
        default=0,
        help="Only count clipped recoveries for lines at or above this word count",
    )
    parser.add_argument(
        "--min-anchor-surplus-words",
        type=int,
        default=0,
        help="Only count clipped recoveries when anchor is this many words longer",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    args = parser.parse_args()

    payload = analyze(
        _load_report(Path(args.report)),
        min_text_similarity=float(args.min_text_similarity),
        min_token_overlap=float(args.min_token_overlap),
        min_line_words=int(args.min_line_words),
        min_anchor_surplus_words=int(args.min_anchor_surplus_words),
    )
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(
        "baseline coverage:",
        (
            f"{payload['baseline_matched_lines_total']}/"
            f"{payload['baseline_eligible_lines_total']}"
        ),
        f"({float(payload['baseline_coverage_ratio_total']):.3f})",
    )
    print(
        "adjusted coverage:",
        (
            f"{payload['adjusted_matched_lines_total']}/"
            f"{payload['baseline_eligible_lines_total']}"
        ),
        f"({float(payload['adjusted_coverage_ratio_total']):.3f})",
    )
    for row in payload.get("rows", []) or []:
        if int(row.get("recovered_lines", 0)) <= 0:
            continue
        print(
            f"{row['song']}: "
            f"{int(row['baseline_matched_lines'])}/"
            f"{int(row['baseline_eligible_lines'])}"
            f" -> {int(row['adjusted_matched_lines'])}/"
            f"{int(row['baseline_eligible_lines'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
