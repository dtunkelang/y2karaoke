"""Core timing adjustment functions for lyrics lines and words."""

import logging
from typing import List, Optional

from ...models import Line, Word

logger = logging.getLogger(__name__)


def _enforce_monotonic_line_starts(
    lines: List[Line], min_gap: float = 0.01, max_forward_shift: float = 3.0
) -> List[Line]:
    """Make line starts non-decreasing, preferring backward pull on large inversions."""
    adjusted: List[Line] = []
    prev_start = None
    for line in lines:
        if not line.words:
            adjusted.append(line)
            continue

        start = line.start_time
        if prev_start is not None and start < prev_start:
            needed = (prev_start - start) + min_gap
            # Large single inversions are often caused by one outlier line pulled too late.
            # Pull the previous line back first to avoid cascading forward shifts.
            if needed > max_forward_shift and adjusted:
                prev_line = adjusted[-1]
                if prev_line.words:
                    prior_start: Optional[float] = None
                    for j in range(len(adjusted) - 2, -1, -1):
                        if adjusted[j].words:
                            prior_start = adjusted[j].start_time
                            break
                    target_prev_start = start - min_gap
                    if prior_start is not None:
                        target_prev_start = max(
                            target_prev_start, prior_start + min_gap
                        )
                    backward = prev_line.start_time - target_prev_start
                    if backward > 0.0:
                        adjusted[-1] = _shift_line(prev_line, -backward)
                        prev_start = adjusted[-1].start_time
                        needed = (
                            (prev_start - start) + min_gap
                            if start < prev_start
                            else 0.0
                        )
            if needed > 0.0:
                line = _shift_line(line, needed)

        adjusted.append(line)
        prev_start = line.start_time

    return adjusted


def _scale_line_to_duration(line: Line, target_duration: float) -> Line:
    """Scale word timings so the line fits within target_duration."""
    if not line.words:
        return line

    start = line.start_time
    end = line.end_time
    duration = end - start
    if duration <= 0 or target_duration <= 0:
        return line

    scale = target_duration / duration
    new_words: List[Word] = []
    for w in line.words:
        rel_start = w.start_time - start
        rel_end = w.end_time - start
        new_start = start + rel_start * scale
        new_end = start + rel_end * scale
        new_words.append(
            Word(
                text=word_text if (word_text := getattr(w, "text", None)) else "",
                start_time=new_start,
                end_time=new_end,
                singer=getattr(w, "singer", ""),
            )
        )
    return Line(words=new_words, singer=line.singer)


def _enforce_non_overlapping_lines(
    lines: List[Line], min_gap: float = 0.02, min_duration: float = 0.2
) -> List[Line]:
    """Shift/scale lines to avoid overlaps while preserving order."""
    if not lines:
        return lines

    adjusted: List[Line] = []
    prev_end: Optional[float] = None
    # Precompute next starts for lookahead
    next_starts: List[Optional[float]] = [None] * len(lines)
    next_start = None
    for i in range(len(lines) - 1, -1, -1):
        next_starts[i] = next_start
        if lines[i].words:
            next_start = lines[i].start_time

    for i, line in enumerate(lines):
        if not line.words:
            adjusted.append(line)
            continue

        start = line.start_time
        if prev_end is not None and start < prev_end + min_gap:
            shift = (prev_end + min_gap) - start
            new_words = [
                Word(
                    text=w.text,
                    start_time=w.start_time + shift,
                    end_time=w.end_time + shift,
                    singer=w.singer,
                )
                for w in line.words
            ]
            line = Line(words=new_words, singer=line.singer)

        next_start = next_starts[i]
        if next_start is not None and line.end_time >= next_start - min_gap:
            target = max(next_start - min_gap - line.start_time, min_duration)
            if target < (line.end_time - line.start_time):
                line = _scale_line_to_duration(line, target)

        adjusted.append(line)
        prev_end = line.end_time

    return adjusted


def _normalize_line_word_timings(
    lines: List[Line],
    min_word_duration: float = 0.05,
    min_gap: float = 0.01,
) -> List[Line]:
    """Ensure each line has monotonic word timings with valid durations."""
    normalized: List[Line] = []
    for line in lines:
        if not line.words:
            normalized.append(line)
            continue

        new_words = []
        last_end = None
        for word in line.words:
            start = float(word.start_time)
            end = float(word.end_time)

            if end < start:
                end = start + min_word_duration

            if last_end is not None and start < last_end:
                shift = (last_end - start) + min_gap
                start += shift
                end += shift

            if end < start:
                end = start + min_word_duration

            new_words.append(
                Word(
                    text=word.text,
                    start_time=start,
                    end_time=end,
                    singer=word.singer,
                )
            )
            last_end = end

        normalized.append(Line(words=new_words, singer=line.singer))

    return normalized


def _apply_offset_to_line(line: Line, offset: float) -> Line:
    """Apply timing offset to all words in a line."""
    new_words = [
        Word(
            text=word.text,
            start_time=word.start_time + offset,
            end_time=word.end_time + offset,
            singer=word.singer,
        )
        for word in line.words
    ]
    return Line(words=new_words, singer=line.singer)


def _shift_line(line: Line, shift: float) -> Line:
    if not line.words:
        return line
    shifted_words = [
        Word(
            text=w.text,
            start_time=w.start_time + shift,
            end_time=w.end_time + shift,
            singer=w.singer,
        )
        for w in line.words
    ]
    return Line(words=shifted_words, singer=line.singer)


def _line_duration(line: Line) -> float:
    duration = line.end_time - line.start_time
    return duration if duration > 0.01 else 0.5
