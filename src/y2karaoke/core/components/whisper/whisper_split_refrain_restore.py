"""Shared helpers for restoring short repeated refrain lines to matching segments."""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from ... import models
from ..alignment import timing_models

_TOKEN_RE = re.compile(r"[^a-z0-9\s]")


def normalize_refrain_text(text: str) -> str:
    return _TOKEN_RE.sub("", text.lower()).strip()


def _normalized_refrain_counts(
    aligned_lines: List[models.Line],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for line in aligned_lines:
        if not line.words or not line.text.strip():
            continue
        key = normalize_refrain_text(line.text)
        if key:
            counts[key] = counts.get(key, 0) + 1
    return counts


def _line_neighbors(
    adjusted: List[models.Line],
    idx: int,
) -> tuple[float | None, float | None]:
    prev_end = (
        adjusted[idx - 1].end_time if idx > 0 and adjusted[idx - 1].words else None
    )
    next_start = (
        adjusted[idx + 1].start_time
        if idx + 1 < len(adjusted) and adjusted[idx + 1].words
        else None
    )
    return prev_end, next_start


def _find_best_matching_segment(
    *,
    key: str,
    line: models.Line,
    sorted_segments: List[timing_models.TranscriptionSegment],
    prev_end: float | None,
    min_gap: float,
    min_late_shift: float,
    max_late_shift: float,
) -> timing_models.TranscriptionSegment | None:
    best_seg: Optional[timing_models.TranscriptionSegment] = None
    best_delta: Optional[float] = None
    for seg in sorted_segments:
        if normalize_refrain_text(seg.text) != key:
            continue
        delta = seg.start - line.start_time
        if delta < min_late_shift or delta > max_late_shift:
            continue
        if prev_end is not None and seg.start <= prev_end + min_gap:
            continue
        if best_delta is None or delta < best_delta:
            best_seg = seg
            best_delta = delta
    return best_seg


def _clamp_target_window(
    *,
    seg: timing_models.TranscriptionSegment,
    prev_end: float | None,
    next_start: float | None,
    min_gap: float,
) -> tuple[float, float] | None:
    target_start = seg.start if prev_end is None else max(seg.start, prev_end + min_gap)
    target_end = seg.end if next_start is None else min(seg.end, next_start - min_gap)
    if target_end <= target_start + 0.2:
        return None
    return target_start, target_end


def _rebuild_line_to_window(
    line: models.Line,
    *,
    target_start: float,
    target_end: float,
) -> models.Line:
    spacing = (target_end - target_start) / len(line.words)
    rebuilt_words: list[models.Word] = []
    for word_idx, word in enumerate(line.words):
        start = target_start + word_idx * spacing
        end = start + spacing * 0.9
        if word_idx == len(line.words) - 1:
            end = target_end
        rebuilt_words.append(
            models.Word(
                text=word.text,
                start_time=start,
                end_time=end,
                singer=word.singer,
            )
        )
    return models.Line(words=rebuilt_words, singer=line.singer)


def restore_split_short_refrains_to_matching_segments(
    aligned_lines: List[models.Line],
    transcription: List[timing_models.TranscriptionSegment],
    *,
    min_gap: float = 0.05,
    max_words: int = 4,
    min_late_shift: float = 0.8,
    max_late_shift: float = 3.0,
) -> Tuple[List[models.Line], int]:
    if not aligned_lines or not transcription:
        return aligned_lines, 0

    normalized_counts = _normalized_refrain_counts(aligned_lines)
    adjusted = list(aligned_lines)
    restored = 0
    sorted_segments = sorted(transcription, key=lambda seg: seg.start)
    for idx, line in enumerate(adjusted):
        if not line.words or len(line.words) > max_words:
            continue
        key = normalize_refrain_text(line.text)
        if not key or normalized_counts.get(key, 0) < 2:
            continue

        prev_end, next_start = _line_neighbors(adjusted, idx)
        best_seg = _find_best_matching_segment(
            key=key,
            line=line,
            sorted_segments=sorted_segments,
            prev_end=prev_end,
            min_gap=min_gap,
            min_late_shift=min_late_shift,
            max_late_shift=max_late_shift,
        )
        if best_seg is None:
            continue

        target_window = _clamp_target_window(
            seg=best_seg,
            prev_end=prev_end,
            next_start=next_start,
            min_gap=min_gap,
        )
        if target_window is None:
            continue

        adjusted[idx] = _rebuild_line_to_window(
            line,
            target_start=target_window[0],
            target_end=target_window[1],
        )
        restored += 1

    return adjusted, restored
