#!/usr/bin/env python3
"""Explain why weak-evidence start-shift restoration likely fired per line."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _normalize_token(text: str) -> str:
    cleaned = re.sub(r"[^a-z]+", "", text.lower())
    if cleaned.endswith("s") and len(cleaned) > 3:
        cleaned = cleaned[:-1]
    return cleaned


def _first_substantive_support_token(words: list[dict[str, Any]]) -> str:
    fallback = ""
    for word in words:
        token = _normalize_token(str(word.get("text", "")))
        if not token:
            continue
        if not fallback:
            fallback = token
        if len(token) >= 2:
            return token
    return fallback


def _line_has_local_first_token_support(
    line_words: list[dict[str, Any]],
    whisper_words: list[dict[str, Any]],
    *,
    line_start: float,
    window_lead_sec: float = 0.3,
    window_follow_sec: float = 1.2,
) -> bool:
    first_token = _first_substantive_support_token(line_words)
    if not first_token:
        return True
    lo = line_start - window_lead_sec
    hi = line_start + window_follow_sec
    for word in whisper_words:
        start = float(word["start"])
        if str(word.get("text")) == "[VOCAL]" or start < lo or start > hi:
            continue
        token = _normalize_token(str(word.get("text", "")))
        if not token:
            continue
        if (
            token == first_token
            or token.startswith(first_token)
            or first_token.startswith(token)
            or (
                len(first_token) >= 2 and (token in first_token or first_token in token)
            )
        ):
            return True
    return False


def _line_window_has_low_confidence(
    whisper_words: list[dict[str, Any]],
    *,
    line_start: float,
    line_end: float,
    next_start: float | None,
    window_lead_sec: float = 1.0,
    low_prob_threshold: float = 0.5,
    low_avg_prob_threshold: float = 0.35,
    low_conf_ratio_threshold: float = 0.5,
) -> bool:
    window_start = line_start - window_lead_sec
    window_end = next_start if next_start is not None else line_end + window_lead_sec
    window_words = [
        word
        for word in whisper_words
        if window_start <= float(word["start"]) < window_end
    ]
    if not window_words:
        return False
    probs = [
        float(word["probability"])
        for word in window_words
        if word.get("probability") is not None
    ]
    if not probs:
        return False
    avg_prob = sum(probs) / len(probs)
    low_conf_count = sum(1 for prob in probs if prob < low_prob_threshold)
    low_conf_ratio = low_conf_count / len(probs)
    return (
        avg_prob < low_avg_prob_threshold or low_conf_ratio >= low_conf_ratio_threshold
    )


def _count_non_vocal_words_near_time(
    whisper_words: list[dict[str, Any]],
    *,
    center_time: float,
    window_sec: float = 1.0,
) -> int:
    lo = center_time - window_sec
    hi = center_time + window_sec
    count = 0
    for word in whisper_words:
        if str(word.get("text")) == "[VOCAL]":
            continue
        start = float(word["start"])
        if lo <= start <= hi:
            count += 1
    return count


def _stage_line_map(snapshot: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(line["line_index"]): line for line in snapshot.get("lines", [])}


def _find_stage_pair(
    payload: dict[str, Any],
    *,
    stage_name: str,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    snapshots = payload.get("snapshots", [])
    for idx, snapshot in enumerate(snapshots):
        if snapshot.get("stage") != stage_name or idx == 0:
            continue
        return snapshots[idx - 1], snapshot
    return None


def analyze(
    stage_trace: dict[str, Any],
    timing_report: dict[str, Any],
    *,
    stage_name: str = "after_restore_weak_evidence_large_start_shifts",
    min_shift_sec: float = 1.1,
    min_support_words: int = 3,
    support_window_sec: float = 1.0,
    lexical_support_shift_sec: float = 1.3,
) -> dict[str, Any]:
    stage_pair = _find_stage_pair(stage_trace, stage_name=stage_name)
    if stage_pair is None:
        return {"stage": stage_name, "rows": []}
    before_snapshot, after_snapshot = stage_pair
    before_lines = _stage_line_map(before_snapshot)
    after_lines = _stage_line_map(after_snapshot)
    report_lines = {int(line["index"]): line for line in timing_report.get("lines", [])}
    rows: list[dict[str, Any]] = []
    for line_index, after_line in after_lines.items():
        before_line = before_lines.get(line_index)
        report_line = report_lines.get(line_index)
        if before_line is None or report_line is None:
            continue
        start_delta = float(after_line["start"]) - float(before_line["start"])
        end_delta = float(after_line["end"]) - float(before_line["end"])
        if abs(start_delta) < 0.05 and abs(end_delta) < 0.05:
            continue

        whisper_words = report_line.get("whisper_window_words") or []
        line_words = report_line.get("words") or []
        abs_shift = abs(float(before_line["start"]) - float(after_line["start"]))
        has_lexical_support = _line_has_local_first_token_support(
            line_words,
            whisper_words,
            line_start=float(before_line["start"]),
        )
        next_start = None
        next_report_line = report_lines.get(line_index + 1)
        if next_report_line is not None:
            next_start = float(next_report_line["start"])
        low_confidence = _line_window_has_low_confidence(
            whisper_words,
            line_start=float(before_line["start"]),
            line_end=float(before_line["end"]),
            next_start=next_start,
        )
        nearby_support = _count_non_vocal_words_near_time(
            whisper_words,
            center_time=float(before_line["start"]),
            window_sec=support_window_sec,
        )

        if abs_shift >= lexical_support_shift_sec and not has_lexical_support:
            reason = "lexical_support_missing"
        elif low_confidence:
            reason = "low_confidence_window"
        elif nearby_support < min_support_words:
            reason = "sparse_support"
        else:
            reason = "other"

        rows.append(
            {
                "line_index": line_index,
                "text": str(after_line.get("text", "")),
                "before_start": float(before_line["start"]),
                "after_start": float(after_line["start"]),
                "before_end": float(before_line["end"]),
                "after_end": float(after_line["end"]),
                "start_delta": round(start_delta, 3),
                "end_delta": round(end_delta, 3),
                "abs_shift": round(abs_shift, 3),
                "has_lexical_support": has_lexical_support,
                "low_confidence_window": low_confidence,
                "nearby_support_words": nearby_support,
                "reason": reason,
                "whisper_window_avg_prob": report_line.get("whisper_window_avg_prob"),
                "whisper_window_low_conf_count": report_line.get(
                    "whisper_window_low_conf_count"
                ),
                "whisper_window_word_count": report_line.get(
                    "whisper_window_word_count"
                ),
                "whisper_window_words": whisper_words,
            }
        )
    rows.sort(key=lambda row: (-abs(row["start_delta"]), row["line_index"]))
    return {
        "stage": stage_name,
        "before_stage": before_snapshot["stage"],
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage_trace_json", help="Stage trace JSON path")
    parser.add_argument("timing_report_json", help="Timing report JSON path")
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    args = parser.parse_args()

    payload = analyze(
        _load_json(Path(args.stage_trace_json)),
        _load_json(Path(args.timing_report_json)),
    )
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(f"{payload['before_stage']} -> {payload['stage']}")
    for row in payload["rows"]:
        print(
            f"line {row['line_index']} {row['reason']}: "
            f"{row['before_start']:.3f}->{row['after_start']:.3f} "
            f"({row['start_delta']:+.3f}), support={row['nearby_support_words']}, "
            f"lexical={row['has_lexical_support']}, "
            f"low_conf={row['low_confidence_window']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
