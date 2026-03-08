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


def _line_window_has_low_confidence(
    mapped_lines: List[models.Line],
    line_index: int,
    whisper_words: List[timing_models.TranscriptionWord],
    *,
    window_lead_sec: float = 1.0,
    low_prob_threshold: float = 0.5,
    low_avg_prob_threshold: float = 0.35,
    low_conf_ratio_threshold: float = 0.5,
) -> bool:
    line = mapped_lines[line_index]
    if not line.words:
        return False
    next_start = (
        mapped_lines[line_index + 1].start_time
        if line_index + 1 < len(mapped_lines) and mapped_lines[line_index + 1].words
        else line.end_time + window_lead_sec
    )
    window_start = line.start_time - window_lead_sec
    window_words = [
        word for word in whisper_words if window_start <= word.start < next_start
    ]
    if not window_words:
        return False
    probs = [word.probability for word in window_words if word.probability is not None]
    if not probs:
        return False
    avg_prob = sum(probs) / len(probs)
    low_conf_count = sum(1 for prob in probs if prob < low_prob_threshold)
    low_conf_ratio = low_conf_count / len(probs)
    return (
        avg_prob < low_avg_prob_threshold or low_conf_ratio >= low_conf_ratio_threshold
    )


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
        if _line_window_has_low_confidence(repaired, idx, whisper_words):
            repaired[idx] = base
            restored += 1
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
