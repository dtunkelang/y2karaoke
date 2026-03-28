"""Sparse-support repair helpers for forced alignment fallback."""

from __future__ import annotations

from typing import Callable, List

import numpy as np

from ... import models
from ..alignment import timing_models
from .whisper_forced_word_redistribution import (
    redistribute_line_with_word_weights as _redistribute_line_with_word_weights,
    sustained_word_layout_weights as _sustained_word_layout_weights,
)


def restore_sparse_support_line_durations_from_source(
    baseline_lines: List[models.Line],
    forced_lines: List[models.Line],
    whisper_words: List[timing_models.TranscriptionWord] | None,
    *,
    non_placeholder_whisper_word_count_fn: Callable[..., int],
    shift_line_fn: Callable[[models.Line, float], models.Line],
    min_baseline_duration_sec: float = 3.0,
    max_baseline_words: int = 5,
    max_duration_ratio: float = 0.7,
    min_effective_word_count: int = 5,
) -> tuple[List[models.Line], int]:
    populated_lines = [line for line in forced_lines if line.words]
    if not populated_lines:
        return forced_lines, 0
    if non_placeholder_whisper_word_count_fn(whisper_words) > max(
        3, int(len(populated_lines) * 0.6)
    ):
        return forced_lines, 0

    repaired = list(forced_lines)
    restored = 0
    for idx, (baseline_line, forced_line) in enumerate(
        zip(baseline_lines, forced_lines)
    ):
        if not baseline_line.words or not forced_line.words:
            continue
        effective_word_count = max(
            len(baseline_line.words),
            len([part for part in baseline_line.text.split() if part.strip()]),
        )
        if effective_word_count < min_effective_word_count:
            continue
        if len(baseline_line.words) > max_baseline_words:
            continue
        baseline_duration = baseline_line.end_time - baseline_line.start_time
        if baseline_duration < min_baseline_duration_sec:
            continue
        forced_duration = forced_line.end_time - forced_line.start_time
        if forced_duration >= baseline_duration * max_duration_ratio:
            continue
        delta = forced_line.start_time - baseline_line.start_time
        repaired[idx] = shift_line_fn(baseline_line, delta)
        restored += 1
    return repaired, restored


def shift_sparse_support_lines_toward_better_onsets(
    baseline_lines: List[models.Line],
    forced_lines: List[models.Line],
    whisper_words: List[timing_models.TranscriptionWord] | None,
    audio_features: timing_models.AudioFeatures | None,
    *,
    non_placeholder_whisper_word_count_fn: Callable[..., int],
    shift_line_fn: Callable[[models.Line, float], models.Line],
    min_late_shift_sec: float = 0.2,
    max_late_shift_sec: float = 0.8,
    min_forced_onset_distance: float = 0.1,
    min_onset_distance_gain: float = 0.03,
) -> tuple[List[models.Line], int]:
    if (
        audio_features is None
        or audio_features.onset_times is None
        or len(audio_features.onset_times) == 0
    ):
        return forced_lines, 0
    if non_placeholder_whisper_word_count_fn(whisper_words) > max(
        3, int(max(1, len([line for line in forced_lines if line.words])) * 0.6)
    ):
        return forced_lines, 0

    onset_times = audio_features.onset_times
    repaired = list(forced_lines)
    shifted = 0
    for idx, (baseline_line, forced_line) in enumerate(
        zip(baseline_lines, forced_lines)
    ):
        if not baseline_line.words or not forced_line.words:
            continue
        if len(forced_line.words) > 5:
            continue
        baseline_candidates = onset_times[
            (onset_times >= baseline_line.start_time - 0.35)
            & (onset_times <= baseline_line.start_time + 0.35)
        ]
        if len(baseline_candidates) == 0:
            continue
        target_start = float(
            baseline_candidates[
                int(np.argmin(np.abs(baseline_candidates - baseline_line.start_time)))
            ]
        )
        late_shift = target_start - forced_line.start_time
        if late_shift < min_late_shift_sec or late_shift > max_late_shift_sec:
            continue
        forced_onset_distance = float(min(abs(onset_times - forced_line.start_time)))
        target_onset_distance = float(min(abs(onset_times - target_start)))
        if forced_onset_distance < min_forced_onset_distance:
            continue
        if forced_onset_distance - target_onset_distance < min_onset_distance_gain:
            continue
        repaired[idx] = shift_line_fn(forced_line, late_shift)
        shifted += 1
    return repaired, shifted


def restore_sparse_support_line_starts_from_source(
    baseline_lines: List[models.Line],
    forced_lines: List[models.Line],
    whisper_words: List[timing_models.TranscriptionWord] | None,
    audio_features: timing_models.AudioFeatures | None,
    *,
    non_placeholder_whisper_word_count_fn: Callable[..., int],
    shift_line_fn: Callable[[models.Line, float], models.Line],
    min_late_shift_sec: float = 0.12,
    max_late_shift_sec: float = 0.9,
    onset_distance_tolerance: float = 0.03,
) -> tuple[List[models.Line], int]:
    if (
        audio_features is None
        or audio_features.onset_times is None
        or len(audio_features.onset_times) == 0
    ):
        return forced_lines, 0
    populated_count = len([line for line in forced_lines if line.words])
    if non_placeholder_whisper_word_count_fn(whisper_words) > max(
        3, int(max(1, populated_count) * 0.6)
    ):
        return forced_lines, 0

    repaired = list(forced_lines)
    onset_times = audio_features.onset_times
    restored = 0
    for idx, (baseline_line, forced_line) in enumerate(
        zip(baseline_lines, forced_lines)
    ):
        if not baseline_line.words or not forced_line.words:
            continue
        late_shift = baseline_line.start_time - forced_line.start_time
        if late_shift < min_late_shift_sec or late_shift > max_late_shift_sec:
            continue
        forced_onset_distance = float(min(abs(onset_times - forced_line.start_time)))
        baseline_onset_distance = float(
            min(abs(onset_times - baseline_line.start_time))
        )
        if baseline_onset_distance > forced_onset_distance + onset_distance_tolerance:
            continue
        repaired[idx] = shift_line_fn(forced_line, late_shift)
        restored += 1
    return repaired, restored


def restore_compact_two_word_lines_from_source(
    baseline_lines: List[models.Line],
    forced_lines: List[models.Line],
    *,
    min_baseline_duration_sec: float = 0.9,
    max_duration_ratio: float = 0.8,
    min_early_shift_sec: float = 0.15,
) -> tuple[List[models.Line], int]:
    repaired = list(forced_lines)
    restored = 0
    for idx, (baseline_line, forced_line) in enumerate(
        zip(baseline_lines, forced_lines)
    ):
        if not baseline_line.words or not forced_line.words:
            continue
        if len(baseline_line.words) != 2 or len(forced_line.words) != 2:
            continue
        baseline_duration = baseline_line.end_time - baseline_line.start_time
        forced_duration = forced_line.end_time - forced_line.start_time
        if baseline_duration < min_baseline_duration_sec or forced_duration <= 0.0:
            continue
        if forced_line.start_time >= baseline_line.start_time - min_early_shift_sec:
            continue
        if forced_duration >= baseline_duration * max_duration_ratio:
            continue
        repaired[idx] = models.Line(
            words=[
                models.Word(
                    text=word.text,
                    start_time=word.start_time,
                    end_time=word.end_time,
                    singer=word.singer,
                )
                for word in baseline_line.words
            ],
            singer=baseline_line.singer,
        )
        restored += 1
    return repaired, restored


def redistribute_sparse_support_sustained_words(
    baseline_lines: List[models.Line],
    forced_lines: List[models.Line],
    whisper_words: List[timing_models.TranscriptionWord] | None,
    *,
    non_placeholder_whisper_word_count_fn: Callable[..., int],
    min_baseline_duration_sec: float = 3.0,
    max_words_per_line: int = 5,
) -> tuple[List[models.Line], int]:
    populated_lines = [line for line in forced_lines if line.words]
    if not populated_lines:
        return forced_lines, 0
    if non_placeholder_whisper_word_count_fn(whisper_words) > max(
        3, int(len(populated_lines) * 0.6)
    ):
        return forced_lines, 0

    repaired = list(forced_lines)
    redistributed = 0
    for idx, (baseline_line, forced_line) in enumerate(
        zip(baseline_lines, forced_lines)
    ):
        if not baseline_line.words or not forced_line.words:
            continue
        if len(forced_line.words) != len(baseline_line.words):
            continue
        word_count = len(forced_line.words)
        if word_count < 3 or word_count > max_words_per_line:
            continue
        baseline_duration = baseline_line.end_time - baseline_line.start_time
        line_duration = forced_line.end_time - forced_line.start_time
        if baseline_duration < min_baseline_duration_sec or line_duration <= 0.0:
            continue

        weights = _sustained_word_layout_weights(forced_line)
        if weights is None:
            continue
        repaired[idx] = _redistribute_line_with_word_weights(forced_line, weights)
        redistributed += 1
    return repaired, redistributed
