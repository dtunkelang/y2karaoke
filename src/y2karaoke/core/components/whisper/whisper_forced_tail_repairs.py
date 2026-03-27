"""Tail repair helpers for accepted forced-alignment lines."""

from __future__ import annotations

from typing import Any, List

from ... import models
from ...audio_analysis import (
    _check_for_silence_in_range,
    _check_vocal_activity_in_range,
)
from ..alignment import timing_models


def _forced_tail_target_end(
    *,
    baseline_end: float,
    next_start: float | None,
    last_word_start: float,
    forced_end: float,
    min_gap_sec: float,
    min_last_word_duration_sec: float,
    max_last_word_duration_sec: float,
) -> float | None:
    target_end = baseline_end
    if next_start is not None:
        target_end = min(target_end, next_start - min_gap_sec)
    if target_end <= last_word_start + min_last_word_duration_sec:
        return None
    if target_end - last_word_start > max_last_word_duration_sec:
        target_end = last_word_start + max_last_word_duration_sec
    if target_end <= forced_end + 0.05:
        return None
    return target_end


def _line_is_low_score_tail_extension_candidate(
    *,
    baseline_line: models.Line,
    forced_line: models.Line,
    segment: Any,
    min_word_count: int,
    max_word_count: int,
    min_end_shortfall_sec: float,
    max_end_shortfall_sec: float,
    max_low_score: float,
) -> bool:
    if not baseline_line.words or not forced_line.words:
        return False
    word_count = len(forced_line.words)
    if word_count < min_word_count or word_count > max_word_count:
        return False
    end_shortfall = baseline_line.end_time - forced_line.end_time
    if end_shortfall < min_end_shortfall_sec or end_shortfall > max_end_shortfall_sec:
        return False
    if not isinstance(segment, dict):
        return False
    seg_words = segment.get("words")
    if not isinstance(seg_words, list) or len(seg_words) != word_count:
        return False
    last_seg_word = seg_words[-1]
    if not isinstance(last_seg_word, dict):
        return False
    last_score = last_seg_word.get("score")
    return isinstance(last_score, (int, float)) and float(last_score) <= max_low_score


def extend_low_score_forced_line_tails_from_source(
    baseline_lines: List[models.Line],
    forced_lines: List[models.Line],
    aligned_segments: Any,
    *,
    min_word_count: int = 4,
    max_word_count: int = 6,
    min_end_shortfall_sec: float = 0.35,
    max_end_shortfall_sec: float = 0.95,
    max_low_score: float = 0.55,
    min_last_word_duration_sec: float = 0.16,
    max_last_word_duration_sec: float = 0.95,
    min_gap_sec: float = 0.05,
) -> tuple[List[models.Line], int]:
    if not isinstance(aligned_segments, list):
        return forced_lines, 0

    repaired = list(forced_lines)
    extended = 0
    for idx, (baseline_line, forced_line, segment) in enumerate(
        zip(baseline_lines, forced_lines, aligned_segments)
    ):
        if not _line_is_low_score_tail_extension_candidate(
            baseline_line=baseline_line,
            forced_line=forced_line,
            segment=segment,
            min_word_count=min_word_count,
            max_word_count=max_word_count,
            min_end_shortfall_sec=min_end_shortfall_sec,
            max_end_shortfall_sec=max_end_shortfall_sec,
            max_low_score=max_low_score,
        ):
            continue
        next_start = (
            repaired[idx + 1].start_time
            if idx + 1 < len(repaired) and repaired[idx + 1].words
            else None
        )
        target_end = _forced_tail_target_end(
            baseline_end=baseline_line.end_time,
            next_start=next_start,
            last_word_start=forced_line.words[-1].start_time,
            forced_end=forced_line.end_time,
            min_gap_sec=min_gap_sec,
            min_last_word_duration_sec=min_last_word_duration_sec,
            max_last_word_duration_sec=max_last_word_duration_sec,
        )
        if target_end is None:
            continue
        repaired[idx] = models.Line(
            words=[
                models.Word(
                    text=word.text,
                    start_time=word.start_time,
                    end_time=(
                        target_end
                        if word_idx == len(forced_line.words) - 1
                        else word.end_time
                    ),
                    singer=word.singer,
                )
                for word_idx, word in enumerate(forced_line.words)
            ],
            singer=forced_line.singer,
        )
        extended += 1
    return repaired, extended


def _final_held_tail_target_end(
    *,
    baseline_end: float,
    forced_end: float,
    vocal_end: float,
    min_extension_sec: float,
) -> float | None:
    target_end = min(baseline_end, vocal_end)
    if target_end <= forced_end + min_extension_sec:
        return None
    return target_end


def _is_final_line_with_words(
    *,
    idx: int,
    line_count: int,
    baseline_line: models.Line,
    forced_line: models.Line,
    audio_features: timing_models.AudioFeatures | None,
) -> bool:
    return (
        idx == line_count - 1
        and audio_features is not None
        and bool(baseline_line.words)
        and bool(forced_line.words)
    )


def _held_tail_word_count_and_duration_ok(
    *,
    baseline_line: models.Line,
    forced_line: models.Line,
    min_word_count: int,
    max_word_count: int,
    min_baseline_duration_sec: float,
    max_duration_ratio: float,
) -> bool:
    word_count = len(forced_line.words)
    if word_count < min_word_count or word_count > max_word_count:
        return False
    baseline_duration = baseline_line.end_time - baseline_line.start_time
    if baseline_duration < min_baseline_duration_sec:
        return False
    forced_duration = forced_line.end_time - forced_line.start_time
    return 0.0 < forced_duration <= baseline_duration * max_duration_ratio


def _held_tail_alignment_shortfall_ok(
    *,
    baseline_line: models.Line,
    forced_line: models.Line,
    min_end_shortfall_sec: float,
    max_end_shortfall_sec: float,
    max_start_delta_sec: float,
) -> bool:
    end_shortfall = baseline_line.end_time - forced_line.end_time
    if end_shortfall < min_end_shortfall_sec or end_shortfall > max_end_shortfall_sec:
        return False
    return abs(forced_line.start_time - baseline_line.start_time) <= max_start_delta_sec


def _held_tail_has_supported_activity(
    *,
    forced_line: models.Line,
    target_end: float,
    audio_features: timing_models.AudioFeatures,
    min_tail_activity: float,
    silence_min_duration: float,
) -> bool:
    activity = _check_vocal_activity_in_range(
        forced_line.end_time, target_end, audio_features
    )
    if activity < min_tail_activity:
        return False
    return not _check_for_silence_in_range(
        forced_line.end_time,
        target_end,
        audio_features,
        min_silence_duration=silence_min_duration,
    )


def _is_final_held_tail_candidate(
    *,
    idx: int,
    line_count: int,
    baseline_line: models.Line,
    forced_line: models.Line,
    audio_features: timing_models.AudioFeatures | None,
    min_word_count: int,
    max_word_count: int,
    min_baseline_duration_sec: float,
    min_end_shortfall_sec: float,
    max_end_shortfall_sec: float,
    max_start_delta_sec: float,
    max_duration_ratio: float,
    min_tail_activity: float,
    silence_min_duration: float,
    min_extension_sec: float,
) -> bool:
    if not _is_final_line_with_words(
        idx=idx,
        line_count=line_count,
        baseline_line=baseline_line,
        forced_line=forced_line,
        audio_features=audio_features,
    ):
        return False
    if not _held_tail_word_count_and_duration_ok(
        baseline_line=baseline_line,
        forced_line=forced_line,
        min_word_count=min_word_count,
        max_word_count=max_word_count,
        min_baseline_duration_sec=min_baseline_duration_sec,
        max_duration_ratio=max_duration_ratio,
    ):
        return False
    if not _held_tail_alignment_shortfall_ok(
        baseline_line=baseline_line,
        forced_line=forced_line,
        min_end_shortfall_sec=min_end_shortfall_sec,
        max_end_shortfall_sec=max_end_shortfall_sec,
        max_start_delta_sec=max_start_delta_sec,
    ):
        return False
    assert audio_features is not None
    target_end = _final_held_tail_target_end(
        baseline_end=baseline_line.end_time,
        forced_end=forced_line.end_time,
        vocal_end=audio_features.vocal_end,
        min_extension_sec=min_extension_sec,
    )
    if target_end is None:
        return False
    return _held_tail_has_supported_activity(
        forced_line=forced_line,
        target_end=target_end,
        audio_features=audio_features,
        min_tail_activity=min_tail_activity,
        silence_min_duration=silence_min_duration,
    )


def extend_final_held_tail_lines_from_activity(
    baseline_lines: List[models.Line],
    forced_lines: List[models.Line],
    audio_features: timing_models.AudioFeatures | None,
    *,
    min_word_count: int = 5,
    max_word_count: int = 7,
    min_baseline_duration_sec: float = 6.0,
    min_end_shortfall_sec: float = 2.5,
    max_end_shortfall_sec: float = 8.0,
    max_start_delta_sec: float = 0.35,
    max_duration_ratio: float = 0.5,
    min_tail_activity: float = 0.9,
    silence_min_duration: float = 0.35,
    min_extension_sec: float = 0.5,
) -> tuple[List[models.Line], int]:
    repaired = list(forced_lines)
    extended = 0
    for idx, (baseline_line, forced_line) in enumerate(
        zip(baseline_lines, forced_lines)
    ):
        if not _is_final_held_tail_candidate(
            idx=idx,
            line_count=len(forced_lines),
            baseline_line=baseline_line,
            forced_line=forced_line,
            audio_features=audio_features,
            min_word_count=min_word_count,
            max_word_count=max_word_count,
            min_baseline_duration_sec=min_baseline_duration_sec,
            min_end_shortfall_sec=min_end_shortfall_sec,
            max_end_shortfall_sec=max_end_shortfall_sec,
            max_start_delta_sec=max_start_delta_sec,
            max_duration_ratio=max_duration_ratio,
            min_tail_activity=min_tail_activity,
            silence_min_duration=silence_min_duration,
            min_extension_sec=min_extension_sec,
        ):
            continue
        assert audio_features is not None
        target_end = _final_held_tail_target_end(
            baseline_end=baseline_line.end_time,
            forced_end=forced_line.end_time,
            vocal_end=audio_features.vocal_end,
            min_extension_sec=min_extension_sec,
        )
        if target_end is None:
            continue
        repaired[idx] = models.Line(
            words=[
                models.Word(
                    text=word.text,
                    start_time=word.start_time,
                    end_time=(
                        target_end
                        if word_idx == len(forced_line.words) - 1
                        else word.end_time
                    ),
                    singer=word.singer,
                )
                for word_idx, word in enumerate(forced_line.words)
            ],
            singer=forced_line.singer,
        )
        extended += 1
    return repaired, extended
