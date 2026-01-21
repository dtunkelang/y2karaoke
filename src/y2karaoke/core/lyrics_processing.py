"""Lyrics processing: creating Line objects and splitting lines."""

from typing import List
from .models import Line, Word
from .romanization import romanize_line
from .lrc_utils import parse_lrc_with_timing

def create_lines_from_lrc(
    lrc_text: str,
    romanize: bool = True,
    title: str = "",
    artist: str = "",
) -> List[Line]:
    """Create Line objects from LRC text with evenly distributed word timings."""
    timed_lines = parse_lrc_with_timing(lrc_text, title, artist)
    if not timed_lines:
        return []

    lines: List[Line] = []
    for i, (start_time, text) in enumerate(timed_lines):
        if romanize:
            text = romanize_line(text)

        if i + 1 < len(timed_lines):
            end_time = timed_lines[i + 1][0]
            if end_time - start_time > 10.0:
                end_time = start_time + 5.0
        else:
            end_time = start_time + 3.0

        word_texts = text.split()
        if not word_texts:
            continue

        line_duration = end_time - start_time
        word_duration = (line_duration * 0.95) / len(word_texts)

        words: List[Word] = []
        for j, word_text in enumerate(word_texts):
            word_start = start_time + j * (line_duration / len(word_texts))
            word_end = word_start + word_duration
            words.append(Word(text=word_text, start_time=word_start, end_time=word_end))

        lines.append(Line(words=words))

    return lines

def split_long_lines(lines: List[Line], max_width_ratio: float = 0.75) -> List[Line]:
    """
    Split lines that are too long for display.

    Args:
        lines: List of Line objects
        max_width_ratio: Maximum ratio of screen width (0.75 = 75%)

    Returns:
        List of Line objects with long lines split
    """
    # Estimate max chars based on width ratio (assuming ~60 char screen width)
    max_chars = int(60 * max_width_ratio)
    result: List[Line] = []

    for line in lines:
        text = line.text
        if len(text) <= max_chars:
            result.append(line)
            continue

        # Split into roughly equal halves at word boundary
        words = line.words
        mid = len(words) // 2

        if mid == 0:
            result.append(line)
            continue

        # First half
        first_words = words[:mid]
        first_line = Line(words=first_words, singer=line.singer)

        # Second half
        second_words = words[mid:]
        second_line = Line(words=second_words, singer=line.singer)

        result.append(first_line)
        result.append(second_line)

    return result