"""Weak-evidence restoration helpers for mapped line start shifts."""

from __future__ import annotations

from typing import List, Tuple

from ... import models
from ..alignment import timing_models


def _count_non_vocal_words_near_time(
    words: List[timing_models.TranscriptionWord],
    center_time: float,
    *,
    window_sec: float = 1.0,
) -> int:
    lo = center_time - window_sec
    hi = center_time + window_sec
    count = 0
    for word in words:
        if word.text == "[VOCAL]":
            continue
        if lo <= word.start <= hi:
            count += 1
    return count


def restore_weak_evidence_large_start_shifts(
    mapped_lines: List[models.Line],
    baseline_lines: List[models.Line],
    whisper_words: List[timing_models.TranscriptionWord],
    *,
    min_shift_sec: float = 1.1,
    min_support_words: int = 3,
    support_window_sec: float = 1.0,
) -> Tuple[List[models.Line], int]:
    repaired = list(mapped_lines)
    restored = 0
    limit = min(len(mapped_lines), len(baseline_lines))
    for idx in range(limit):
        mapped = repaired[idx]
        base = baseline_lines[idx]
        if not mapped.words or not base.words:
            continue
        shift = mapped.start_time - base.start_time
        if shift < min_shift_sec:
            continue
        support = _count_non_vocal_words_near_time(
            whisper_words,
            mapped.start_time,
            window_sec=support_window_sec,
        )
        if support >= min_support_words:
            continue
        repaired[idx] = base
        restored += 1
    return repaired, restored
