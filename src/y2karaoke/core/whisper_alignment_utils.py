"""Utility functions for Whisper-based alignment."""

import logging
from typing import List, Optional, Tuple

from .models import Line, Word
from .timing_models import TranscriptionSegment
from .phonetic_utils import _phonetic_similarity

logger = logging.getLogger(__name__)


def _find_best_whisper_segment(
    line_text: str,
    line_start: float,
    sorted_segments: List[TranscriptionSegment],
    language: str,
    min_similarity: float,
) -> Tuple[Optional[TranscriptionSegment], float, float]:
    """Find best matching Whisper segment for a line."""
    best_segment = None
    best_similarity = 0.0
    best_offset = 0.0

    for seg in sorted_segments:
        if seg.start > line_start + 15 or seg.end < line_start - 15:
            continue

        similarity = _phonetic_similarity(line_text, seg.text, language)
        if similarity > best_similarity and similarity >= min_similarity:
            best_similarity = similarity
            best_segment = seg
            best_offset = seg.start - line_start

    return best_segment, best_similarity, best_offset


def _drop_duplicate_lines(
    lines: List[Line],
    segments: List[TranscriptionSegment],
    language: str,
    min_similarity: float = 0.7,
) -> Tuple[List[Line], int]:
    """Drop adjacent duplicate lines when they map to the same Whisper segment."""
    if not lines:
        return lines, 0

    sorted_segments = sorted(segments, key=lambda s: s.start) if segments else []
    deduped: List[Line] = []
    dropped = 0

    for line in lines:
        if not deduped:
            deduped.append(line)
            continue

        prev = deduped[-1]
        if not prev.words or not line.words:
            deduped.append(line)
            continue

        if prev.text.strip() != line.text.strip():
            deduped.append(line)
            continue

        prev_seg, prev_sim, _ = _find_best_whisper_segment(
            prev.text, prev.start_time, sorted_segments, language, min_similarity
        )
        seg, sim, _ = _find_best_whisper_segment(
            line.text, line.start_time, sorted_segments, language, min_similarity
        )

        if (
            prev_seg is not None
            and seg is not None
            and prev_seg is seg
            and prev_sim >= min_similarity
            and sim >= min_similarity
        ):
            dropped += 1
            continue

        deduped.append(line)

    return deduped, dropped


def _clamp_repeated_line_duration(
    lines: List[Line],
    max_duration: float = 1.5,
    min_gap: float = 0.01,
) -> Tuple[List[Line], int]:
    """Clamp duration for immediately repeated lines so they don't hang too long."""
    if not lines:
        return lines, 0

    adjusted = list(lines)
    fixes = 0

    for idx in range(1, len(adjusted)):
        prev = adjusted[idx - 1]
        line = adjusted[idx]
        if not prev.words or not line.words:
            continue

        if prev.text.strip() != line.text.strip():
            continue

        duration = line.end_time - line.start_time
        if duration <= max_duration:
            continue

        new_end = line.start_time + max_duration
        if idx + 1 < len(adjusted) and adjusted[idx + 1].words:
            next_start = adjusted[idx + 1].start_time
            new_end = min(new_end, next_start - min_gap)
        if new_end <= line.start_time:
            continue

        total_duration = max(new_end - line.start_time, 0.2)
        word_count = max(len(line.words), 1)
        spacing = total_duration / word_count
        new_words = []
        for i, word in enumerate(line.words):
            start = line.start_time + i * spacing
            end = start + spacing * 0.9
            new_words.append(
                Word(
                    text=word.text,
                    start_time=start,
                    end_time=end,
                    singer=word.singer,
                )
            )
        adjusted[idx] = Line(words=new_words, singer=line.singer)
        fixes += 1

    return adjusted, fixes
