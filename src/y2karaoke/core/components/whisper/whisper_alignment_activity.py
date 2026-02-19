"""Vocal-activity gap fill and timing dedupe helpers for Whisper alignment."""

from __future__ import annotations

import logging
from typing import Callable, List, Optional, Tuple

from ...models import Line
from ..alignment.timing_models import (
    AudioFeatures,
    TranscriptionSegment,
    TranscriptionWord,
)

logger = logging.getLogger(__name__)


def _build_vocal_block_words(
    start: float,
    end: float,
    chunk_duration: float,
) -> List[TranscriptionWord]:
    words: List[TranscriptionWord] = []
    curr = start
    while curr + chunk_duration <= end:
        words.append(
            TranscriptionWord(
                start=curr,
                end=curr + chunk_duration,
                text="[VOCAL]",
                probability=0.0,
            )
        )
        curr += chunk_duration
    return words


def _fill_vocal_activity_gaps(
    whisper_words: List[TranscriptionWord],
    audio_features: AudioFeatures,
    *,
    check_vocal_activity_in_range_fn: Callable[[float, float, AudioFeatures], float],
    threshold: float = 0.3,
    min_gap: float = 1.0,
    chunk_duration: float = 0.5,
    segments: Optional[List[TranscriptionSegment]] = None,
) -> Tuple[List[TranscriptionWord], Optional[List[TranscriptionSegment]]]:
    """Inject pseudo-words and segments where vocal activity is high but transcription missing."""
    if not whisper_words:
        return whisper_words, segments

    filled_words = list(whisper_words)
    filled_words.sort(key=lambda w: w.start)

    filled_segments = list(segments) if segments is not None else None
    if filled_segments:
        filled_segments.sort(key=lambda s: s.start)

    new_words: List[TranscriptionWord] = []
    new_segments: List[TranscriptionSegment] = []

    def add_vocal_block(start: float, end: float) -> None:
        seg_words = _build_vocal_block_words(start, end, chunk_duration)
        if not seg_words:
            return
        new_words.extend(seg_words)
        if filled_segments is not None:
            new_segments.append(
                TranscriptionSegment(
                    start=seg_words[0].start,
                    end=seg_words[-1].end,
                    text="[VOCAL]",
                    words=seg_words,
                )
            )

    vocal_start = audio_features.vocal_start
    if filled_words[0].start - vocal_start >= min_gap:
        activity = check_vocal_activity_in_range_fn(
            vocal_start, filled_words[0].start, audio_features
        )
        if activity > threshold:
            add_vocal_block(vocal_start, filled_words[0].start)

    for idx in range(len(filled_words) - 1):
        gap_start = filled_words[idx].end
        gap_end = filled_words[idx + 1].start
        if gap_end - gap_start < min_gap:
            continue
        activity = check_vocal_activity_in_range_fn(gap_start, gap_end, audio_features)
        if activity > threshold:
            add_vocal_block(gap_start, gap_end)

    vocal_end = audio_features.vocal_end
    if vocal_end - filled_words[-1].end >= min_gap:
        activity = check_vocal_activity_in_range_fn(
            filled_words[-1].end, vocal_end, audio_features
        )
        if activity > threshold:
            add_vocal_block(filled_words[-1].end, vocal_end)

    if new_words:
        logger.info(
            "Vocal gap filler: injected %d [VOCAL] pseudo-words", len(new_words)
        )
        filled_words.extend(new_words)
        filled_words.sort(key=lambda w: w.start)
        if filled_segments is not None and new_segments:
            filled_segments.extend(new_segments)
            filled_segments.sort(key=lambda s: s.start)

    return filled_words, filled_segments


def _drop_duplicate_lines_by_timing(
    lines: List[Line],
    max_gap: float = 0.2,
) -> Tuple[List[Line], int]:
    """Drop adjacent duplicate lines that overlap or are nearly contiguous."""
    if not lines:
        return lines, 0

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
        gap = line.start_time - prev.end_time
        if gap <= max_gap:
            dropped += 1
            continue
        deduped.append(line)
    return deduped, dropped
