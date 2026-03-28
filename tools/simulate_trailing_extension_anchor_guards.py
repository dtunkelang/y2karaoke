#!/usr/bin/env python3
"""Simulate anchor-shift guards for trailing-extension candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _end_guard_tool():
    from tools import simulate_trailing_extension_end_guards

    return simulate_trailing_extension_end_guards


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _first_anchor_shift(candidate: dict[str, Any], *, line_start: float) -> float:
    pairs = candidate.get("matched_pairs") or []
    if not pairs:
        return 0.0
    return round(float(pairs[0]["candidate_start"]) - float(line_start), 3)


def _guarded_choice(
    candidates: list[dict[str, Any]],
    *,
    line_start: float,
    max_first_anchor_shift: float = 1.0,
) -> dict[str, Any] | None:
    if not candidates:
        return None
    allowed = [
        row
        for row in candidates
        if _first_anchor_shift(row, line_start=line_start) <= max_first_anchor_shift
    ]
    if not allowed:
        return candidates[0]
    return max(allowed, key=lambda row: (row["score"], row["matched_count"]))


def analyze(
    *,
    stage_trace: dict[str, Any],
    timing_report: dict[str, Any],
    line_index: int,
    stage_name: str = "postpass_shift_repeated",
    transcription_json: dict[str, Any] | None = None,
    max_first_anchor_shift: float = 1.0,
) -> dict[str, Any]:
    payload = _end_guard_tool().analyze(
        stage_trace=stage_trace,
        timing_report=timing_report,
        line_index=line_index,
        stage_name=stage_name,
        transcription_json=transcription_json,
    )
    for row in payload.get("candidates", []):
        row["first_anchor_shift"] = _first_anchor_shift(
            row, line_start=float(payload["line_start"])
        )
    payload["anchor_guard_choice"] = _guarded_choice(
        payload.get("candidates", []),
        line_start=float(payload["line_start"]),
        max_first_anchor_shift=max_first_anchor_shift,
    )
    payload["anchor_guard_params"] = {
        "max_first_anchor_shift": max_first_anchor_shift,
    }
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
    parser.add_argument("--max-first-anchor-shift", type=float, default=1.0)
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
        max_first_anchor_shift=args.max_first_anchor_shift,
    )
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(f"line {payload['line_index']}: {payload['line_text']}")
    print(f"current: {payload['current_choice']}")
    print(f"anchor_guard: {payload['anchor_guard_choice']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
