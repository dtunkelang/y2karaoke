#!/usr/bin/env python3
"""Simulate guard policies for trailing-extension end selection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _scoring_tool():
    from tools import simulate_trailing_extension_candidate_scoring

    return simulate_trailing_extension_candidate_scoring


def _current_choice(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: (
            row["matched_count"],
            row["last_end"] if row["last_end"] is not None else float("-inf"),
        ),
    )


def _guarded_choice(
    candidates: list[dict[str, Any]],
    *,
    min_score_margin: float = 1.0,
    max_end_distance: float = 2.5,
    forbid_crossing_next: bool = True,
    max_short_soft_only_matches: int = 0,
) -> dict[str, Any] | None:
    if not candidates:
        return None
    allowed = [
        row
        for row in candidates
        if row["end_distance"] <= max_end_distance
        and row["short_soft_only_matches"] <= max_short_soft_only_matches
        and (not forbid_crossing_next or not row["crosses_next_start"])
    ]
    if not allowed:
        return _current_choice(candidates)

    best = max(allowed, key=lambda row: (row["score"], row["matched_count"]))
    current = _current_choice(candidates)
    if current is None:
        return best
    if best["score"] >= current["score"] + min_score_margin:
        return best
    return current


def analyze(
    *,
    stage_trace: dict[str, Any],
    timing_report: dict[str, Any],
    line_index: int,
    stage_name: str = "postpass_shift_repeated",
    transcription_json: dict[str, Any] | None = None,
    min_score_margin: float = 1.0,
    max_end_distance: float = 2.5,
    forbid_crossing_next: bool = True,
    max_short_soft_only_matches: int = 0,
) -> dict[str, Any]:
    payload = _scoring_tool().analyze(
        stage_trace=stage_trace,
        timing_report=timing_report,
        line_index=line_index,
        stage_name=stage_name,
        transcription_json=transcription_json,
    )
    candidates = payload.get("candidates", [])
    current = _current_choice(candidates)
    guarded = _guarded_choice(
        candidates,
        min_score_margin=min_score_margin,
        max_end_distance=max_end_distance,
        forbid_crossing_next=forbid_crossing_next,
        max_short_soft_only_matches=max_short_soft_only_matches,
    )
    payload.update(
        {
            "current_choice": current,
            "guarded_choice": guarded,
            "guard_params": {
                "min_score_margin": min_score_margin,
                "max_end_distance": max_end_distance,
                "forbid_crossing_next": forbid_crossing_next,
                "max_short_soft_only_matches": max_short_soft_only_matches,
            },
        }
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage_trace_json", help="Whisper stage trace JSON")
    parser.add_argument("timing_report_json", help="Timing report JSON")
    parser.add_argument(
        "--transcription-json",
        help=(
            "Optional cached Whisper transcription JSON to use as the true "
            "word source"
        ),
    )
    parser.add_argument(
        "--line-index", type=int, required=True, help="1-based line index"
    )
    parser.add_argument(
        "--stage",
        default="postpass_shift_repeated",
        help="Stage to inspect before tail extension",
    )
    parser.add_argument("--min-score-margin", type=float, default=1.0)
    parser.add_argument("--max-end-distance", type=float, default=2.5)
    parser.add_argument("--allow-crossing-next", action="store_true")
    parser.add_argument("--max-short-soft-only-matches", type=int, default=0)
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    args = parser.parse_args()

    payload = analyze(
        stage_trace=_load_json(Path(args.stage_trace_json)),
        timing_report=_load_json(Path(args.timing_report_json)),
        line_index=args.line_index,
        stage_name=args.stage,
        transcription_json=(
            _load_json(Path(args.transcription_json))
            if args.transcription_json
            else None
        ),
        min_score_margin=args.min_score_margin,
        max_end_distance=args.max_end_distance,
        forbid_crossing_next=not args.allow_crossing_next,
        max_short_soft_only_matches=args.max_short_soft_only_matches,
    )
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(f"line {payload['line_index']}: {payload['line_text']}")
    print(f"current: {payload['current_choice']}")
    print(f"guarded: {payload['guarded_choice']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
