#!/usr/bin/env python3
"""Inspect candidate matches for the trailing-extension postpass."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from y2karaoke.core.components.whisper.whisper_mapping_post_text import (  # noqa: E402
    _normalize_match_token,
    _soft_token_match,
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _find_snapshot(stage_trace: dict[str, Any], stage_name: str) -> dict[str, Any]:
    for snapshot in stage_trace.get("snapshots", []):
        if snapshot.get("stage") == stage_name:
            return snapshot
    return {}


def _snapshot_lines(snapshot: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(line["line_index"]): line for line in snapshot.get("lines", [])}


def _collect_report_words(timing_report: dict[str, Any]) -> list[dict[str, Any]]:
    seen: set[tuple[str, float, float]] = set()
    collected: list[dict[str, Any]] = []
    for line in timing_report.get("lines", []):
        for word in line.get("whisper_window_words") or []:
            key = (
                str(word.get("text", "")),
                float(word.get("start", 0.0)),
                float(word.get("end", 0.0)),
            )
            if key in seen:
                continue
            seen.add(key)
            collected.append(
                {
                    "text": key[0],
                    "start": key[1],
                    "end": key[2],
                    "probability": word.get("probability"),
                }
            )
    return sorted(
        collected, key=lambda word: (word["start"], word["end"], word["text"])
    )


def _collect_transcription_words(
    transcription_json: dict[str, Any]
) -> list[dict[str, Any]]:
    segments = transcription_json.get("segments") or []
    collected: list[dict[str, Any]] = []
    for segment in segments:
        for word in segment.get("words") or []:
            text = str(word.get("word") or word.get("text") or "")
            if not text:
                continue
            collected.append(
                {
                    "text": text,
                    "start": float(word["start"]),
                    "end": float(word["end"]),
                    "probability": word.get("probability"),
                }
            )
    return sorted(
        collected, key=lambda word: (word["start"], word["end"], word["text"])
    )


def analyze(
    *,
    stage_trace: dict[str, Any],
    timing_report: dict[str, Any],
    line_index: int,
    stage_name: str = "postpass_shift_repeated",
    transcription_json: dict[str, Any] | None = None,
) -> dict[str, Any]:
    snapshot = _find_snapshot(stage_trace, stage_name)
    if not snapshot:
        return {"stage": stage_name, "line_index": line_index, "candidates": []}
    lines = _snapshot_lines(snapshot)
    line = lines.get(line_index)
    if not line:
        return {"stage": stage_name, "line_index": line_index, "candidates": []}
    report_lines = {
        int(report_line["index"]): report_line
        for report_line in timing_report.get("lines", [])
    }
    report_line = report_lines.get(line_index, {})
    next_line = lines.get(line_index + 1)

    words = (
        _collect_transcription_words(transcription_json)
        if transcription_json is not None
        else _collect_report_words(timing_report)
    )
    token_pairs = [
        (idx, _normalize_match_token(word["text"]))
        for idx, word in enumerate(report_line.get("words") or line.get("words") or [])
    ]
    token_pairs = [(idx, tok) for idx, tok in token_pairs if tok]
    next_start = float(next_line["start"]) if next_line else float("inf")
    window_end = (
        next_start + 3.0 if next_start != float("inf") else float(line["end"]) + 3.0
    )
    candidates = [
        word
        for word in words
        if float(line["start"]) - 0.25 <= float(word["start"]) <= window_end
    ]

    rows: list[dict[str, Any]] = []
    for start_i in range(len(candidates)):
        wi = start_i
        matched = 0
        last_end = None
        matched_pairs: list[dict[str, Any]] = []
        for word_idx, tok in token_pairs:
            found = False
            while wi < len(candidates):
                ww_tok = _normalize_match_token(str(candidates[wi]["text"]))
                if _soft_token_match(tok, ww_tok):
                    matched += 1
                    last_end = float(candidates[wi]["end"])
                    matched_pairs.append(
                        {
                            "line_word_index": word_idx,
                            "line_token": tok,
                            "candidate_index": wi,
                            "candidate_text": candidates[wi]["text"],
                            "candidate_start": candidates[wi]["start"],
                            "candidate_end": candidates[wi]["end"],
                        }
                    )
                    wi += 1
                    found = True
                    break
                wi += 1
            if wi >= len(candidates):
                break
            if not found:
                continue
        rows.append(
            {
                "start_i": start_i,
                "start_word": candidates[start_i],
                "matched_count": matched,
                "last_end": last_end,
                "matched_pairs": matched_pairs,
            }
        )
    rows.sort(
        key=lambda row: (
            row["matched_count"],
            row["last_end"] if row["last_end"] is not None else float("-inf"),
        ),
        reverse=True,
    )
    return {
        "stage": stage_name,
        "line_index": line_index,
        "line_text": line.get("text"),
        "line_start": line.get("start"),
        "line_end": line.get("end"),
        "next_start": None if next_start == float("inf") else next_start,
        "candidates": rows,
    }


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
    for row in payload["candidates"][:8]:
        start = row["start_word"]
        print(
            f"start_i={row['start_i']} start={start['text']}@{start['start']:.2f} "
            f"matched={row['matched_count']} last_end={row['last_end']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
