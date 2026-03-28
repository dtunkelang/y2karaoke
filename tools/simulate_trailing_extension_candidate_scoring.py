#!/usr/bin/env python3
"""Score trailing-extension candidates to compare early vs late phrase choices."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools import analyze_trailing_extension_candidates as candidate_tool  # noqa: E402


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _score_candidate(
    candidate: dict[str, Any],
    *,
    line_end: float,
    next_start: float | None,
    short_token_penalty: float = 1.5,
    end_distance_weight: float = 0.5,
    next_cross_penalty_weight: float = 0.8,
) -> dict[str, Any]:
    matched_pairs = candidate.get("matched_pairs") or []
    exact_matches = 0
    soft_only_matches = 0
    short_soft_only_matches = 0
    for pair in matched_pairs:
        line_token = str(pair["line_token"])
        cand_text = str(pair["candidate_text"])
        cand_token = candidate_tool._normalize_match_token(cand_text)
        if cand_token == line_token:
            exact_matches += 1
            continue
        soft_only_matches += 1
        if len(line_token) < 3 or len(cand_token) < 3:
            short_soft_only_matches += 1

    last_end = candidate.get("last_end")
    end_distance = (
        max(0.0, float(last_end) - float(line_end)) if last_end is not None else 0.0
    )
    crosses_next = (
        next_start is not None
        and last_end is not None
        and float(last_end) > float(next_start)
    )
    score = (
        float(candidate.get("matched_count", 0))
        + 0.25 * exact_matches
        - 0.15 * soft_only_matches
        - short_token_penalty * short_soft_only_matches
        - end_distance_weight * end_distance
        - (next_cross_penalty_weight if crosses_next else 0.0)
    )
    enriched = dict(candidate)
    enriched.update(
        {
            "exact_matches": exact_matches,
            "soft_only_matches": soft_only_matches,
            "short_soft_only_matches": short_soft_only_matches,
            "end_distance": round(end_distance, 3),
            "crosses_next_start": crosses_next,
            "score": round(score, 3),
        }
    )
    return enriched


def analyze(
    *,
    stage_trace: dict[str, Any],
    timing_report: dict[str, Any],
    line_index: int,
    stage_name: str = "postpass_shift_repeated",
    transcription_json: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = candidate_tool.analyze(
        stage_trace=stage_trace,
        timing_report=timing_report,
        line_index=line_index,
        stage_name=stage_name,
        transcription_json=transcription_json,
    )
    next_start = payload.get("next_start")
    scored = [
        _score_candidate(
            candidate,
            line_end=float(payload["line_end"]),
            next_start=float(next_start) if next_start is not None else None,
        )
        for candidate in payload.get("candidates", [])
    ]
    scored.sort(key=lambda row: (row["score"], row["matched_count"]), reverse=True)
    payload["candidates"] = scored
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
    )
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(f"line {payload['line_index']}: {payload['line_text']}")
    for candidate in payload["candidates"][:8]:
        start = candidate["start_word"]
        print(
            f"score={candidate['score']:+.3f} "
            f"start={start['text']}@{start['start']:.2f} "
            f"matched={candidate['matched_count']} last_end={candidate['last_end']} "
            f"short_soft={candidate['short_soft_only_matches']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
