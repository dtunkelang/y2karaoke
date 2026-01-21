"""Build timing segments for forced alignment."""

from typing import List, Tuple
from ..utils.logging import get_logger

logger = get_logger(__name__)


def build_segments_from_lrc(
    text_lines: List[str],
    line_timings: List[Tuple[float, str]]
) -> List[dict]:
    """
    Build WhisperX segments from Genius text and LRC timing.

    We use Genius for text content but LRC for approximate timing.
    """
    from rapidfuzz import fuzz

    segments = []
    used_timings = set()

    for text in text_lines:
        if not text.strip():
            continue

        # Find best matching LRC line for timing
        best_timing = None
        best_score = 0

        for i, (timestamp, lrc_text) in enumerate(line_timings):
            if i in used_timings:
                continue
            score = fuzz.ratio(text.lower(), lrc_text.lower())
            if score > best_score:
                best_score = score
                best_timing = (i, timestamp)

        if best_timing and best_score > 50:
            used_timings.add(best_timing[0])
            start_time = best_timing[1]
        else:
            # No good match - estimate based on previous segment
            if segments:
                start_time = segments[-1].get("end", 0) + 0.5
            else:
                start_time = 0.0

        # Estimate end time
        end_time = start_time + max(len(text.split()) * 0.4, 1.0)

        segments.append({
            "text": text,
            "start": start_time,
            "end": end_time,
        })

    return segments


def build_segments_estimated(
    text_lines: List[str],
    duration: float,
    offset: float
) -> List[dict]:
    """Build segments with evenly distributed timing."""
    segments = []
    non_empty_lines = [t for t in text_lines if t.strip()]

    if not non_empty_lines:
        return segments

    # Leave some buffer at start and end
    usable_duration = duration - offset - 5.0
    time_per_line = usable_duration / len(non_empty_lines)

    for i, text in enumerate(non_empty_lines):
        start_time = offset + i * time_per_line
        end_time = start_time + time_per_line * 0.9

        segments.append({
            "text": text,
            "start": start_time,
            "end": end_time,
        })

    return segments


def build_lines_estimated(
    text_lines: List[str],
    duration: float,
    offset: float
) -> List:
    """Build lines with evenly distributed word timing (fallback)."""
    from .models import Line, Word

    lines = []
    non_empty = [t for t in text_lines if t.strip()]

    if not non_empty:
        return lines

    usable_duration = duration - offset - 5.0
    time_per_line = usable_duration / len(non_empty)

    for i, text in enumerate(non_empty):
        line_start = offset + i * time_per_line
        word_texts = text.split()

        if not word_texts:
            continue

        word_duration = (time_per_line * 0.9) / len(word_texts)
        words = []

        for j, word_text in enumerate(word_texts):
            word_start = line_start + j * word_duration
            words.append(Word(
                text=word_text,
                start_time=word_start,
                end_time=word_start + word_duration * 0.9,
            ))

        lines.append(Line(words=words))

    return lines
