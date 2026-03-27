#!/usr/bin/env python3
"""Compare guarded two-layer agreement recovery settings across a report pack."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any, cast

pack_tool = cast(Any, importlib.import_module("tools.analyze_two_layer_agreement_pack"))


def _resolve_report(path: Path) -> Path:
    return path / "benchmark_report.json" if path.is_dir() else path


def _load_report(path: Path) -> dict[str, Any]:
    return json.loads(_resolve_report(path).read_text(encoding="utf-8"))


def analyze(
    report_doc: dict[str, Any],
    *,
    min_text_similarity: float = 0.58,
    min_token_overlap: float = 0.5,
    guard_candidates: list[tuple[int, int, int]] | None = None,
) -> dict[str, Any]:
    candidates = guard_candidates or [
        (6, 15, 15),
        (6, 15, 20),
        (8, 15, 20),
    ]
    rows: list[dict[str, Any]] = []
    for min_line_words, min_anchor_surplus_words, min_anchor_words in candidates:
        payload = pack_tool.analyze(
            report_doc,
            min_text_similarity=min_text_similarity,
            min_token_overlap=min_token_overlap,
            min_line_words=min_line_words,
            min_anchor_surplus_words=min_anchor_surplus_words,
            min_anchor_words=min_anchor_words,
        )
        con_calma_recovered = 0
        spillover_song_count = 0
        for row in payload.get("rows", []) or []:
            song_name = str(row.get("song", ""))
            recovered = int(row.get("recovered_lines", 0) or 0)
            if "Con Calma" in song_name:
                con_calma_recovered = recovered
            elif recovered > 0:
                spillover_song_count += 1
        score = (con_calma_recovered * 3.0) - spillover_song_count * 2.0
        rows.append(
            {
                "min_line_words": min_line_words,
                "min_anchor_surplus_words": min_anchor_surplus_words,
                "min_anchor_words": min_anchor_words,
                "baseline_coverage_ratio_total": payload[
                    "baseline_coverage_ratio_total"
                ],
                "adjusted_coverage_ratio_total": payload[
                    "adjusted_coverage_ratio_total"
                ],
                "con_calma_recovered_lines": con_calma_recovered,
                "spillover_song_count": spillover_song_count,
                "score": round(score, 4),
            }
        )
    rows.sort(
        key=lambda row: (
            -float(row["score"]),
            -int(row["con_calma_recovered_lines"]),
            int(row["spillover_song_count"]),
            -int(row["min_anchor_words"]),
            -int(row["min_anchor_surplus_words"]),
            -int(row["min_line_words"]),
        )
    )
    return {
        "rows": rows,
        "best_candidate": rows[0] if rows else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", help="Benchmark report JSON or run directory")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON",
    )
    args = parser.parse_args()

    payload = analyze(_load_report(Path(args.report)))
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    best = payload.get("best_candidate") or {}
    print("best candidate:")
    print(
        f"  line_words>={best.get('min_line_words')} "
        f"anchor_surplus>={best.get('min_anchor_surplus_words')} "
        f"anchor_words>={best.get('min_anchor_words')}"
    )
    for row in payload.get("rows", []) or []:
        print(
            f"guard=({row['min_line_words']},{row['min_anchor_surplus_words']},"
            f"{row['min_anchor_words']}) "
            f"coverage={float(row['adjusted_coverage_ratio_total']):.4f} "
            f"con_calma={int(row['con_calma_recovered_lines'])} "
            f"spillover={int(row['spillover_song_count'])} "
            f"score={float(row['score']):.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
