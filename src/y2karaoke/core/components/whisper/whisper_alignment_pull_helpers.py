"""Shared helper utilities for Whisper pull/merge timing rules."""

from __future__ import annotations

from typing import List, Optional, Tuple

from ...models import Line, Word
from ..alignment.timing_models import TranscriptionSegment


def nearest_segment_by_start(
    start_time: float,
    segments: List[TranscriptionSegment],
    max_time_window: float,
) -> Optional[TranscriptionSegment]:
    """Return segment with start closest to start_time within max window."""
    nearest = None
    nearest_gap = None
    for cand in segments:
        gap = abs(cand.start - start_time)
        if gap > max_time_window:
            continue
        if nearest_gap is None or gap < nearest_gap:
            nearest_gap = gap
            nearest = cand
    return nearest


def nearest_prior_segment_by_end(
    start_time: float,
    segments: List[TranscriptionSegment],
    max_time_window: float,
) -> Optional[Tuple[TranscriptionSegment, float]]:
    """Return nearest segment ending before start_time and the late_by amount."""
    nearest = None
    nearest_late = None
    for cand in segments:
        if abs(cand.start - start_time) > max_time_window:
            continue
        late_by = start_time - cand.end
        if late_by < 0:
            continue
        if nearest_late is None or late_by < nearest_late:
            nearest_late = late_by
            nearest = cand
    if nearest is None or nearest_late is None:
        return None
    return nearest, nearest_late


def line_neighbors(
    lines: List[Line], idx: int
) -> Tuple[Optional[float], Optional[float]]:
    """Return previous line end and next line start around idx when available."""
    prev_end = None
    if idx > 0 and lines[idx - 1].words:
        prev_end = lines[idx - 1].end_time
    next_start = None
    if idx + 1 < len(lines) and lines[idx + 1].words:
        next_start = lines[idx + 1].start_time
    return prev_end, next_start


def reflow_words_to_window(
    words: List[Word], window_start: float, window_end: float
) -> List[Word]:
    """Spread words evenly into [window_start, window_end]."""
    if not words:
        return []
    total_duration = max(window_end - window_start, 0.2)
    spacing = total_duration / len(words)
    out: List[Word] = []
    for i, word in enumerate(words):
        start = window_start + i * spacing
        end = start + spacing * 0.9
        out.append(
            Word(
                text=word.text,
                start_time=start,
                end_time=end,
                singer=word.singer,
            )
        )
    return out


def reflow_two_lines_to_segment(
    line: Line,
    next_line: Line,
    segment: TranscriptionSegment,
    prev_end: Optional[float] = None,
) -> Optional[Tuple[Line, Line]]:
    """Reflow two lines into one segment window preserving word order."""
    total_words = len(line.words) + len(next_line.words)
    if total_words <= 0:
        return None
    window_start = segment.start
    if prev_end is not None:
        window_start = max(window_start, prev_end + 0.01)
    if window_start >= segment.end:
        return None
    total_duration = max(segment.end - window_start, 0.2)
    spacing = total_duration / total_words
    new_line_words: List[Word] = []
    new_next_words: List[Word] = []
    for i, word in enumerate(line.words):
        start = window_start + i * spacing
        end = start + spacing * 0.9
        new_line_words.append(
            Word(
                text=word.text,
                start_time=start,
                end_time=end,
                singer=word.singer,
            )
        )
    for j, word in enumerate(next_line.words):
        idx_in_seg = len(line.words) + j
        start = window_start + idx_in_seg * spacing
        end = start + spacing * 0.9
        new_next_words.append(
            Word(
                text=word.text,
                start_time=start,
                end_time=end,
                singer=word.singer,
            )
        )
    return Line(words=new_line_words, singer=line.singer), Line(
        words=new_next_words, singer=next_line.singer
    )
