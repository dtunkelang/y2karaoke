#!/usr/bin/env python3
"""Analyze hard alignment cases using onset, syllable, and repeat signals."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

from y2karaoke.core.audio_analysis import extract_audio_features
from y2karaoke.core.components.alignment.timing_evaluator_scoring import (
    find_closest_onset,
)

_TOKEN_RE = re.compile(r"[a-z0-9']+")
_VOWEL_RUN_RE = re.compile(r"[aeiouy]+", re.IGNORECASE)


def _normalize_text(text: str) -> str:
    return " ".join(_TOKEN_RE.findall(text.lower()))


def _estimate_token_syllables(token: str) -> int:
    cleaned = "".join(ch for ch in token.lower() if ch.isalpha())
    if not cleaned:
        return 0
    groups = _VOWEL_RUN_RE.findall(cleaned)
    count = len(groups)
    if cleaned.endswith("ed") and count > 1 and cleaned[-3:-2] not in "aeiouy":
        count -= 1
    if cleaned.endswith("e") and count > 1 and not cleaned.endswith(("le", "ye")):
        count -= 1
    return max(1, count)


def _estimate_text_syllables(text: str) -> int:
    tokens = _TOKEN_RE.findall(text.lower())
    return sum(_estimate_token_syllables(token) for token in tokens)


def _build_repeated_family_stats(
    lines: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    norm_texts = [_normalize_text(str(line.get("text") or "")) for line in lines]
    counts = Counter(norm_texts)
    durations_by_text: dict[str, list[float]] = {}
    for norm_text, line in zip(norm_texts, lines):
        if not norm_text:
            continue
        durations_by_text.setdefault(norm_text, []).append(
            float(line.get("end", 0.0)) - float(line.get("start", 0.0))
        )
    return {
        norm_text: {
            "count": count,
            "median_duration": median(durations_by_text.get(norm_text, [0.0])),
        }
        for norm_text, count in counts.items()
    }


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _gold_lines_by_index(gold_payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for line in gold_payload.get("lines", []):
        idx = int(line.get("line_index", 0))
        out[idx] = line
    return out


def _fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}"


def _analyze_line(
    report_line: dict[str, Any],
    *,
    gold_line: dict[str, Any] | None,
    onset_times: Any,
    family_stats: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    start = float(report_line["start"])
    end = float(report_line["end"])
    duration = end - start
    pre_start = report_line.get("pre_whisper_start")
    pre_end = report_line.get("pre_whisper_end")
    pre_duration = None
    if isinstance(pre_start, (int, float)) and isinstance(pre_end, (int, float)):
        pre_duration = float(pre_end) - float(pre_start)
    gold_start = None
    gold_end = None
    if gold_line is not None:
        gold_start = float(gold_line.get("start", 0.0))
        gold_end = float(gold_line.get("end", 0.0))

    syllables = _estimate_text_syllables(str(report_line.get("text") or ""))
    word_count = len(report_line.get("words", []))
    expected_min_duration = max(0.15 * syllables, 0.11 * word_count, 0.25)
    norm_text = _normalize_text(str(report_line.get("text") or ""))
    family = family_stats.get(norm_text, {"count": 1, "median_duration": duration})
    family_count = int(family.get("count", 1))
    family_median_duration = float(
        family["median_duration"]
        if isinstance(family.get("median_duration"), (int, float))
        else duration
    )

    pred_onset, pred_onset_delta = find_closest_onset(
        start, onset_times, max_distance=1.5
    )
    pre_onset = None
    pre_onset_delta = None
    if isinstance(pre_start, (int, float)):
        pre_onset, pre_onset_delta = find_closest_onset(
            float(pre_start), onset_times, max_distance=1.5
        )

    flags: list[str] = []
    if (
        isinstance(pre_start, (int, float))
        and start - float(pre_start) > 0.8
        and gold_start is not None
    ):
        flags.append("late_vs_source")
    if duration < expected_min_duration * 0.75:
        flags.append("compressed_for_syllables")
    if int(report_line.get("whisper_window_word_count", 0)) == 0:
        flags.append("zero_window_support")
    if family_count >= 2 and abs(duration - family_median_duration) > 0.6:
        flags.append("repeat_duration_outlier")
    if (
        pred_onset is not None
        and isinstance(pre_start, (int, float))
        and pre_onset is not None
        and abs(float(pre_onset_delta or 0.0)) + 0.08
        < abs(float(pred_onset_delta or 0.0))
    ):
        flags.append("source_nearer_onset")

    return {
        "index": int(report_line["index"]),
        "text": report_line["text"],
        "start": start,
        "end": end,
        "duration": duration,
        "pre_start": float(pre_start) if isinstance(pre_start, (int, float)) else None,
        "pre_end": float(pre_end) if isinstance(pre_end, (int, float)) else None,
        "pre_duration": pre_duration,
        "gold_start": gold_start,
        "gold_end": gold_end,
        "start_error": None if gold_start is None else abs(start - gold_start),
        "pre_start_error": (
            None
            if gold_start is None or not isinstance(pre_start, (int, float))
            else abs(float(pre_start) - gold_start)
        ),
        "syllables": syllables,
        "word_count": word_count,
        "expected_min_duration": expected_min_duration,
        "duration_vs_expected": duration / expected_min_duration,
        "whisper_window_word_count": int(
            report_line.get("whisper_window_word_count", 0)
        ),
        "whisper_window_avg_prob": report_line.get("whisper_window_avg_prob"),
        "pred_nearest_onset": None if pred_onset is None else float(pred_onset),
        "pred_onset_delta": (
            None
            if pred_onset is None or pred_onset_delta is None
            else float(pred_onset_delta)
        ),
        "pre_nearest_onset": None if pre_onset is None else float(pre_onset),
        "pre_onset_delta": (
            None
            if pre_onset is None or pre_onset_delta is None
            else float(pre_onset_delta)
        ),
        "repeat_family_count": family_count,
        "repeat_family_median_duration": family_median_duration,
        "flags": flags,
    }


def _iter_song_analyses(
    benchmark_report: dict[str, Any], pattern: re.Pattern[str] | None
):
    for song in benchmark_report.get("songs", []):
        title = str(song.get("title") or "")
        artist = str(song.get("artist") or "")
        label = f"{artist} - {title}"
        if pattern and not pattern.search(label):
            continue
        report_path = Path(song["report_path"])
        gold_path = Path(song["gold_path"])
        report_payload = _load_json(report_path)
        gold_payload = _load_json(gold_path)
        gold_by_index = _gold_lines_by_index(gold_payload)
        audio_path = Path(gold_payload["audio_path"])
        features = extract_audio_features(str(audio_path))
        onset_times = features.onset_times if features is not None else []
        family_stats = _build_repeated_family_stats(report_payload.get("lines", []))
        yield {
            "label": label,
            "report_path": str(report_path),
            "gold_path": str(gold_path),
            "audio_path": str(audio_path),
            "lines": [
                _analyze_line(
                    line,
                    gold_line=gold_by_index.get(int(line["index"])),
                    onset_times=onset_times,
                    family_stats=family_stats,
                )
                for line in report_payload.get("lines", [])
            ],
        }


def _render_song(song: dict[str, Any]) -> str:
    lines = [
        f"## {song['label']}",
        f"- report: `{song['report_path']}`",
        f"- gold: `{song['gold_path']}`",
        f"- audio: `{song['audio_path']}`",
    ]
    ranked = sorted(
        song["lines"],
        key=lambda line: (
            -float(line["start_error"] or 0.0),
            -len(line["flags"]),
            line["index"],
        ),
    )
    for line in ranked:
        if not line["flags"] and (line["start_error"] or 0.0) < 0.25:
            continue
        lines.append(
            "- line {idx} `{text}`: pred={pred}->{end}, pre={pre}->{pre_end}, "
            "gold={gold}->{gold_end}, err={err}, syll={syll}, "
            "dur/expected={ratio}, whisper_words={wwc}, onset_delta={odelta}, flags={flags}".format(
                idx=line["index"],
                text=line["text"],
                pred=_fmt(line["start"]),
                end=_fmt(line["end"]),
                pre=_fmt(line["pre_start"]),
                pre_end=_fmt(line["pre_end"]),
                gold=_fmt(line["gold_start"]),
                gold_end=_fmt(line["gold_end"]),
                err=_fmt(line["start_error"]),
                syll=line["syllables"],
                ratio=_fmt(line["duration_vs_expected"]),
                wwc=line["whisper_window_word_count"],
                odelta=_fmt(
                    None
                    if line["pred_onset_delta"] is None
                    else abs(line["pred_onset_delta"])
                ),
                flags=",".join(line["flags"]) or "-",
            )
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark_report", help="Path to benchmark_report.json")
    parser.add_argument(
        "--match",
        help="Regex filter on 'Artist - Title'",
    )
    args = parser.parse_args()

    benchmark_report = _load_json(Path(args.benchmark_report))
    pattern = re.compile(args.match, re.IGNORECASE) if args.match else None
    analyses = list(_iter_song_analyses(benchmark_report, pattern))
    for idx, song in enumerate(analyses):
        if idx:
            print()
        print(_render_song(song))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
