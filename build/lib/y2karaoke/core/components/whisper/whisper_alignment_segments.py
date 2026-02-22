"""Functions for aligning lyrics lines to Whisper segments."""

import logging
from typing import List, Tuple

from ...models import Line, Word
from ..alignment.timing_models import TranscriptionSegment
from . import whisper_alignment_utils
from . import whisper_alignment_retime
from . import whisper_alignment_pull

logger = logging.getLogger(__name__)

# Re-export utility functions
_find_best_whisper_segment = whisper_alignment_utils._find_best_whisper_segment
_drop_duplicate_lines = whisper_alignment_utils._drop_duplicate_lines
_clamp_repeated_line_duration = whisper_alignment_utils._clamp_repeated_line_duration

# Re-export retiming functions
_retime_line_to_segment = whisper_alignment_retime._retime_line_to_segment
_retime_line_to_segment_with_min_start = (
    whisper_alignment_retime._retime_line_to_segment_with_min_start
)
_retime_line_to_window = whisper_alignment_retime._retime_line_to_window
_retime_adjacent_lines_to_whisper_window = (
    whisper_alignment_retime._retime_adjacent_lines_to_whisper_window
)
_retime_adjacent_lines_to_segment_window = (
    whisper_alignment_retime._retime_adjacent_lines_to_segment_window
)

# Re-export pulling and merging functions
_merge_lines_to_whisper_segments = (
    whisper_alignment_pull._merge_lines_to_whisper_segments
)
_pull_next_line_into_segment_window = (
    whisper_alignment_pull._pull_next_line_into_segment_window
)
_pull_next_line_into_same_segment = (
    whisper_alignment_pull._pull_next_line_into_same_segment
)
_merge_short_following_line_into_segment = (
    whisper_alignment_pull._merge_short_following_line_into_segment
)
_pull_lines_near_segment_end = whisper_alignment_pull._pull_lines_near_segment_end
_pull_lines_to_best_segments = whisper_alignment_pull._pull_lines_to_best_segments


def _merge_first_two_lines_if_segment_matches(
    lines: List[Line],
    segments: List[TranscriptionSegment],
    language: str,
    min_similarity: float = 0.8,
) -> Tuple[List[Line], bool]:
    """Merge the first two non-empty lines if Whisper sees them as one phrase."""
    if not lines or not segments:
        return lines, False

    first_idx = None
    second_idx = None
    for idx, line in enumerate(lines):
        if line.words:
            if first_idx is None:
                first_idx = idx
            else:
                second_idx = idx
                break

    if first_idx is None or second_idx is None:
        return lines, False

    first_line = lines[first_idx]
    second_line = lines[second_idx]
    combined_text = f"{first_line.text} {second_line.text}".strip()

    seg, sim, _ = _find_best_whisper_segment(
        combined_text,
        first_line.start_time,
        sorted(segments, key=lambda s: s.start),
        language,
        min_similarity,
    )

    if seg is None or sim < min_similarity:
        return lines, False

    merged_words = list(first_line.words) + list(second_line.words)
    word_count = max(len(merged_words), 1)
    total_duration = max(seg.end - seg.start, 0.2)
    spacing = total_duration / word_count
    new_words = []
    for i, word in enumerate(merged_words):
        start = seg.start + i * spacing
        end = start + spacing * 0.9
        new_words.append(
            Word(
                text=word.text,
                start_time=start,
                end_time=end,
                singer=word.singer,
            )
        )

    lines[first_idx] = Line(words=new_words, singer=first_line.singer)
    lines[second_idx] = Line(words=[], singer=second_line.singer)
    return lines, True


def _tighten_lines_to_whisper_segments(
    lines: List[Line],
    segments: List[TranscriptionSegment],
    language: str,
    min_similarity: float = 0.5,
) -> Tuple[List[Line], int]:
    """Pull consecutive lines to Whisper segment boundaries when phrased as one."""
    if not lines or not segments:
        return lines, 0

    adjusted = list(lines)
    fixes = 0
    sorted_segments = sorted(segments, key=lambda s: s.start)

    for idx in range(1, len(adjusted)):
        prev_line = adjusted[idx - 1]
        line = adjusted[idx]
        if not prev_line.words or not line.words:
            continue

        line_text = line.text.strip()
        if not line_text:
            continue

        prev_seg, prev_sim, _ = _find_best_whisper_segment(
            prev_line.text,
            prev_line.start_time,
            sorted_segments,
            language,
            min_similarity,
        )
        seg, sim, _ = _find_best_whisper_segment(
            line_text, line.start_time, sorted_segments, language, min_similarity
        )

        if seg is None or prev_seg is None:
            continue

        # If both lines map to the same segment, pull line start to segment end.
        if seg is prev_seg:
            shift = seg.end - line.start_time
        else:
            # If line maps to next segment but starts far after it, pull to segment start.
            if line.start_time - seg.start <= 0:
                continue
            if seg.start - prev_seg.end > 2.5:
                continue
            if sim < min_similarity:
                continue
            shift = seg.start - line.start_time

        if shift >= -0.2:
            continue

        new_words = [
            Word(
                text=w.text,
                start_time=w.start_time + shift,
                end_time=w.end_time + shift,
                singer=w.singer,
            )
            for w in line.words
        ]
        adjusted[idx] = Line(words=new_words, singer=line.singer)
        fixes += 1

    return adjusted, fixes
