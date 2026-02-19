"""Interjection-specific post-processing for Whisper-mapped lines."""

from __future__ import annotations

from typing import Callable, List, Optional

from ... import models
from ..alignment import timing_models


def _retime_short_interjection_lines(
    mapped_lines: List[models.Line],
    segments: List[timing_models.TranscriptionSegment],
    *,
    is_interjection_line_fn: Callable[[str], bool],
    interjection_similarity_fn: Callable[[str, str], float],
    min_similarity: float = 0.8,
    max_shift: float = 8.0,
    min_gap: float = 0.05,
) -> List[models.Line]:
    """Retarget short interjection lines (for example, 'Oooh') to matching segments."""
    if not mapped_lines or not segments:
        return mapped_lines

    adjusted = list(mapped_lines)
    sorted_segments = sorted(segments, key=lambda s: s.start)

    for idx, line in enumerate(adjusted):
        if not line.words or not is_interjection_line_fn(line.text):
            continue

        prev_end = (
            adjusted[idx - 1].end_time if idx > 0 and adjusted[idx - 1].words else None
        )
        next_start = (
            adjusted[idx + 1].start_time
            if idx + 1 < len(adjusted) and adjusted[idx + 1].words
            else None
        )

        best_seg: Optional[timing_models.TranscriptionSegment] = None
        best_score = 0.0
        for seg in sorted_segments:
            if abs(seg.start - line.start_time) > max_shift:
                continue
            score = interjection_similarity_fn(line.text, seg.text)
            if score > best_score:
                best_score = score
                best_seg = seg

        if best_seg is None or best_score < min_similarity:
            continue

        window_start = best_seg.start
        if prev_end is not None:
            window_start = max(window_start, prev_end + min_gap)

        window_end = best_seg.end
        if next_start is not None:
            window_end = min(window_end, next_start - min_gap)

        if window_end - window_start <= 0.05:
            continue

        total_duration = max(window_end - window_start, 0.2)
        spacing = total_duration / len(line.words)
        new_words = []
        for word_idx, w in enumerate(line.words):
            start = window_start + word_idx * spacing
            end = start + spacing * 0.9
            new_words.append(
                models.Word(
                    text=w.text,
                    start_time=start,
                    end_time=end,
                    singer=w.singer,
                )
            )
        adjusted[idx] = models.Line(words=new_words, singer=line.singer)

    return adjusted
