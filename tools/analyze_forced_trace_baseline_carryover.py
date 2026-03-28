#!/usr/bin/env python3
"""Flag repeated short forced lines whose bad final timing comes from baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _normalized_tokens(text: str) -> list[str]:
    return [
        "".join(ch for ch in part.lower() if ch.isalnum() or ch == "'")
        for part in text.split()
        if "".join(ch for ch in part.lower() if ch.isalnum() or ch == "'")
    ]


def _overlap_ratio(tokens: list[str], segment_text: str) -> float:
    segment_tokens = set(_normalized_tokens(segment_text))
    if not tokens or not segment_tokens:
        return 0.0
    return sum(1 for token in tokens if token in segment_tokens) / len(tokens)


def _best_transcription_overlap(line: dict, metadata: dict) -> float:
    tokens = _normalized_tokens(str(line.get("text") or ""))
    segments = metadata.get("transcription_segment_preview") or []
    return max(
        (
            _overlap_ratio(tokens, str(segment.get("text") or ""))
            for segment in segments
            if isinstance(segment, dict)
        ),
        default=0.0,
    )


def _load_lines(payload: dict, stage: str) -> list[dict]:
    for snapshot in payload.get("snapshots", []):
        if snapshot.get("stage") == stage:
            return snapshot.get("lines", [])
    return []


def analyze_trace(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    metadata = payload.get("metadata", {})
    baseline_lines = _load_lines(payload, "baseline_lines")
    final_lines = _load_lines(payload, "final_forced_lines")
    hits: list[dict] = []
    pair_count = min(len(baseline_lines), len(final_lines))
    for idx in range(1, pair_count):
        baseline = baseline_lines[idx]
        previous = baseline_lines[idx - 1]
        final = final_lines[idx]
        text = str(baseline.get("text") or "")
        if len(_normalized_tokens(text)) > 2:
            continue
        if _normalized_tokens(text) != _normalized_tokens(
            str(previous.get("text") or "")
        ):
            continue
        if _best_transcription_overlap(baseline, metadata) > 0.0:
            continue
        start_delta = abs(
            float(final.get("start", 0.0)) - float(baseline.get("start", 0.0))
        )
        end_delta = abs(float(final.get("end", 0.0)) - float(baseline.get("end", 0.0)))
        if start_delta > 0.15 or end_delta > 0.15:
            continue
        hits.append(
            {
                "line_index": int(baseline.get("line_index", idx + 1)),
                "text": text,
                "baseline_start": float(baseline.get("start", 0.0)),
                "baseline_end": float(baseline.get("end", 0.0)),
                "final_start": float(final.get("start", 0.0)),
                "final_end": float(final.get("end", 0.0)),
                "transcription_overlap_ratio": _best_transcription_overlap(
                    baseline, metadata
                ),
            }
        )
    return hits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_json", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze_trace(args.trace_json), indent=2))


if __name__ == "__main__":
    main()
