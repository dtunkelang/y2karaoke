"""Functions for retiming lyrics lines within Whisper segments/windows."""

import logging
from typing import List, Optional, Tuple

from ...models import Line, Word
from ..alignment.timing_models import TranscriptionSegment
from ...phonetic_utils import _phonetic_similarity
from .whisper_alignment_utils import _find_best_whisper_segment

logger = logging.getLogger(__name__)


def _retime_line_to_segment(line: Line, seg: TranscriptionSegment) -> Line:
    """Retime a line's words evenly within a Whisper segment."""
    if not line.words:
        return line

    word_count = max(len(line.words), 1)
    total_duration = max(seg.end - seg.start, 0.2)
    spacing = total_duration / word_count
    new_words = []
    for i, word in enumerate(line.words):
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
    return Line(words=new_words, singer=line.singer)


def _retime_line_to_segment_with_min_start(
    line: Line, seg: TranscriptionSegment, min_start: float
) -> Line:
    """Retime a line within a segment, respecting a minimum start time."""
    start_base = max(seg.start, min_start)
    if start_base >= seg.end:
        return line

    word_count = max(len(line.words), 1)
    total_duration = max(seg.end - start_base, 0.2)
    spacing = total_duration / word_count
    new_words = []
    for i, word in enumerate(line.words):
        start = start_base + i * spacing
        end = start + spacing * 0.9
        new_words.append(
            Word(
                text=word.text,
                start_time=start,
                end_time=end,
                singer=word.singer,
            )
        )
    return Line(words=new_words, singer=line.singer)


def _retime_line_to_window(line: Line, window_start: float, window_end: float) -> Line:
    """Retime a line's words evenly within a custom window."""
    if window_end <= window_start:
        return line

    if not line.words:
        return line

    word_count = max(len(line.words), 1)
    total_duration = max(window_end - window_start, 0.2)
    spacing = total_duration / word_count
    new_words = []
    for i, word in enumerate(line.words):
        start = window_start + i * spacing
        end = start + spacing * 0.9
        new_words.append(
            Word(
                text=word.text,
                start_time=start,
                end_time=end,
                singer=word.singer,
            )
        )
    return Line(words=new_words, singer=line.singer)


def _retime_adjacent_lines_to_whisper_window(  # noqa: C901
    lines: List[Line],
    segments: List[TranscriptionSegment],
    language: str,
    min_similarity: float = 0.35,
    max_gap: float = 2.5,
    min_similarity_late: float = 0.25,
    max_late: float = 3.0,
    max_late_short: float = 6.0,
    max_time_window: float = 15.0,
    max_window_duration: Optional[float] = None,
    max_start_offset: Optional[float] = None,
) -> Tuple[List[Line], int]:
    """Retune adjacent lines into a single Whisper window without merging them."""
    if not lines or not segments:
        return lines, 0

    adjusted = list(lines)
    fixes = 0
    sorted_segments = sorted(segments, key=lambda s: s.start)

    for idx in range(len(adjusted) - 1):
        line = adjusted[idx]
        next_line = adjusted[idx + 1]
        if not line.words or not next_line.words:
            continue

        combined_text = f"{line.text} {next_line.text}".strip()
        prev_end = None
        if idx > 0 and adjusted[idx - 1].words:
            prev_end = adjusted[idx - 1].end_time

        seg, combined_sim, _ = _find_best_whisper_segment(
            combined_text, line.start_time, sorted_segments, language, min_similarity
        )
        if seg is None or combined_sim < min_similarity:
            # Fallback: if the pair is late, try the nearest segment by time.
            nearest = None
            nearest_gap = None
            for cand in sorted_segments:
                gap = abs(cand.start - line.start_time)
                if gap > max_time_window:
                    continue
                if nearest_gap is None or gap < nearest_gap:
                    nearest_gap = gap
                    nearest = cand
            if nearest is None:
                continue
            if line.start_time - nearest.start < max_late:
                continue

            combined_sim = _phonetic_similarity(combined_text, nearest.text, language)
            seg = nearest

            # If the nearest segment is after the line start, try the prior segment instead.
            prior = None
            prior_gap = None
            for cand in sorted_segments:
                if (
                    cand.end <= line.start_time
                    and cand.start >= line.start_time - max_time_window
                ):
                    gap = line.start_time - cand.end
                    if prior_gap is None or gap < prior_gap:
                        prior_gap = gap
                        prior = cand
            if prior is not None and prior_gap is not None:
                prior_sim = _phonetic_similarity(combined_text, prior.text, language)
                if (
                    prior_gap <= max_late_short
                    and (line.start_time - prior.start) > max_late
                    and prior_sim >= min_similarity_late
                ):
                    seg = prior
                    combined_sim = prior_sim

            if combined_sim < min_similarity_late:
                continue
        if (
            max_window_duration is not None
            and (seg.end - seg.start) > max_window_duration
        ):
            continue
        if (
            max_start_offset is not None
            and abs(line.start_time - seg.start) > max_start_offset
        ):
            continue

        total_words = len(line.words) + len(next_line.words)
        if total_words <= 0:
            continue

        window_start = seg.start
        if prev_end is not None:
            window_start = max(window_start, prev_end + 0.01)
        if window_start >= seg.end:
            continue
        if (
            max_window_duration is not None
            and (seg.end - window_start) > max_window_duration
        ):
            continue

        total_duration = max(seg.end - window_start, 0.2)
        spacing = total_duration / total_words
        new_line_words = []
        new_next_words = []

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

        gap = new_next_words[0].start_time - new_line_words[-1].end_time
        if gap > max_gap:
            continue

        adjusted[idx] = Line(words=new_line_words, singer=line.singer)
        adjusted[idx + 1] = Line(words=new_next_words, singer=next_line.singer)
        fixes += 1

    return adjusted, fixes


def _retime_adjacent_lines_to_segment_window(
    lines: List[Line],
    segments: List[TranscriptionSegment],
    language: str,
    min_similarity: float = 0.35,
    max_gap: float = 2.5,
    max_time_window: float = 15.0,
) -> Tuple[List[Line], int]:
    """Retune adjacent lines into a two-segment Whisper window."""
    if not lines or len(segments) < 2:
        return lines, 0

    adjusted = list(lines)
    fixes = 0
    sorted_segments = sorted(segments, key=lambda s: s.start)

    for idx in range(len(adjusted) - 1):
        line = adjusted[idx]
        next_line = adjusted[idx + 1]
        if not line.words or not next_line.words:
            continue

        combined_text = f"{line.text} {next_line.text}".strip()

        for s_idx in range(len(sorted_segments) - 1):
            seg_a = sorted_segments[s_idx]
            seg_b = sorted_segments[s_idx + 1]
            if seg_b.start - seg_a.end > 1.0:
                continue
            if abs(seg_a.start - line.start_time) > max_time_window:
                continue

            window_text = f"{seg_a.text} {seg_b.text}".strip()
            combined_sim = _phonetic_similarity(combined_text, window_text, language)
            if combined_sim < min_similarity:
                continue

            total_words = len(line.words) + len(next_line.words)
            if total_words <= 0:
                continue

            window_start = seg_a.start
            window_end = seg_b.end
            total_duration = max(window_end - window_start, 0.2)
            spacing = total_duration / total_words

            new_line_words = []
            new_next_words = []
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

            gap = new_next_words[0].start_time - new_line_words[-1].end_time
            if gap > max_gap:
                continue

            adjusted[idx] = Line(words=new_line_words, singer=line.singer)
            adjusted[idx + 1] = Line(words=new_next_words, singer=next_line.singer)
            fixes += 1
            break

    return adjusted, fixes
