"""Forced-fallback repairs for exact adjacent transcription segment boundaries."""

from __future__ import annotations

from typing import List

from ... import models
from ..alignment import timing_models
from . import whisper_forced_local_repairs as _forced_local_repairs

_normalize_token = _forced_local_repairs.normalize_token
_shift_line = _forced_local_repairs.shift_line


def _normalized_token_list(text: str) -> list[str]:
    return [
        token for token in (_normalize_token(part) for part in text.split()) if token
    ]


def _segment_text_matches_line(
    segment: timing_models.TranscriptionSegment,
    line: models.Line,
) -> bool:
    return _normalized_token_list(segment.text) == _normalized_token_list(line.text)


def _set_line_end(line: models.Line, end_time: float) -> models.Line:
    if not line.words or end_time <= line.start_time:
        return line
    words = [
        models.Word(
            text=word.text,
            start_time=word.start_time,
            end_time=(end_time if idx == len(line.words) - 1 else word.end_time),
            singer=word.singer,
        )
        for idx, word in enumerate(line.words)
    ]
    return models.Line(words=words, singer=line.singer)


def restore_forced_exact_adjacent_segment_boundaries(
    forced_lines: List[models.Line],
    segments: List[timing_models.TranscriptionSegment] | None,
    *,
    max_segment_gap_sec: float = 0.1,
    min_tail_shortfall_sec: float = 0.3,
    min_next_early_start_sec: float = 0.3,
    min_gap: float = 0.05,
) -> tuple[List[models.Line], int]:
    if not segments or len(segments) < 2:
        return forced_lines, 0

    repaired = list(forced_lines)
    restored = 0
    pair_count = min(len(repaired), len(segments)) - 1
    for idx in range(pair_count):
        left = repaired[idx]
        right = repaired[idx + 1]
        left_segment = segments[idx]
        right_segment = segments[idx + 1]
        if not left.words or not right.words:
            continue
        if not _segment_text_matches_line(left_segment, left):
            continue
        if not _segment_text_matches_line(right_segment, right):
            continue
        if abs(right_segment.start - left_segment.end) > max_segment_gap_sec:
            continue
        if (left_segment.end - left.end_time) < min_tail_shortfall_sec:
            continue
        if (right_segment.start - right.start_time) < min_next_early_start_sec:
            continue
        repaired_left = _set_line_end(left, left_segment.end - min_gap)
        shifted_right = _shift_line(right, right_segment.start - right.start_time)
        repaired_right = _set_line_end(shifted_right, right_segment.end)
        if repaired_left.end_time > repaired_right.start_time - min_gap:
            continue
        repaired[idx] = repaired_left
        repaired[idx + 1] = repaired_right
        restored += 1
    return repaired, restored
