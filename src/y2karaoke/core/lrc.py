"""LRC parsing and Line creation for karaoke lyrics.

This module handles:
- LRC timestamp parsing
- Extracting lyrics text from LRC format
- Creating Line objects from LRC timings
- Splitting long lines for display
"""

import re
from typing import List, Tuple, Optional
from difflib import SequenceMatcher
from .models import Line, Word
from .romanization import romanize_line

# ----------------------
# LRC timestamp regex
# ----------------------
_LRC_TS_RE = re.compile(
    r"""
    \[                      # opening bracket
    (?P<min>\d+)            # minutes
    :
    (?P<sec>[0-5]?\d)       # seconds
    (?:\.(?P<frac>\d{1,3}))?  # optional fractional seconds
    \]                      # closing bracket
    """,
    re.VERBOSE
)


# ----------------------
# Metadata filtering
# ----------------------
def _is_metadata_line(text: str, title: str = "", artist: str = "") -> bool:
    """
    Determine if a line is metadata rather than actual lyrics.
    Skips obvious labels and title/artist lines.
    """
    if not text:
        return True

    text_lower = text.lower().strip()

    metadata_prefixes = [
        "artist:", "song:", "title:", "album:", "writer:", "composer:",
        "lyricist:", "lyrics by", "written by", "produced by", "music by",
    ]
    for prefix in metadata_prefixes:
        if text_lower.startswith(prefix):
            return True

    if title:
        title_lower = title.lower().replace(" ", "")
        line_normalized = text_lower.replace(" ", "")
        if line_normalized == title_lower:
            return True

    if artist:
        artist_lower = artist.lower().replace(" ", "")
        line_normalized = text_lower.replace(" ", "")
        if line_normalized == artist_lower:
            return True

    return False


# ----------------------
# LRC timestamp parsing
# ----------------------
def parse_lrc_timestamp(ts: str) -> Optional[float]:
    """Parse a single LRC timestamp like [01:23.45] to seconds."""
    if not ts:
        return None
    match = _LRC_TS_RE.match(ts.strip())
    if not match:
        return None
    minutes = int(match.group("min"))
    seconds = int(match.group("sec"))
    if seconds >= 60:
        return None
    frac = match.group("frac")
    frac_seconds = int(frac) / (10 ** len(frac)) if frac else 0.0
    return minutes * 60 + seconds + frac_seconds


def extract_lyrics_text(lrc_text: str, title: str = "", artist: str = "") -> List[str]:
    """Extract plain text lines from LRC format (ignore timing and metadata)."""
    if not lrc_text:
        return []

    lines: List[str] = []
    for line in lrc_text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        match = _LRC_TS_RE.match(line)
        if match:
            text_part = line[match.end():].strip()
            if text_part and not _is_metadata_line(text_part, title, artist):
                lines.append(text_part)
    return lines


def parse_lrc_with_timing(lrc_text: str, title: str = "", artist: str = "") -> List[Tuple[float, str]]:
    """Parse LRC format and extract (timestamp, text) tuples."""
    if not lrc_text:
        return []

    lines: List[Tuple[float, str]] = []
    for line in lrc_text.strip().splitlines():
        line = line.strip()
        if not line:
            continue

        match = _LRC_TS_RE.match(line)
        if not match:
            continue

        timestamp = parse_lrc_timestamp(match.group(0))
        if timestamp is None:
            continue

        text_part = line[match.end():].strip()
        if text_part and not _is_metadata_line(text_part, title, artist):
            lines.append((timestamp, text_part))

    return lines


# ----------------------
# Line creation from LRC
# ----------------------
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


def create_lines_from_lrc_timings(
    lrc_timings: List[Tuple[float, str]],
    genius_lines: List[str],
) -> List[Line]:
    """Create Line objects from LRC timings + Genius canonical text.

    Matches LRC lines to Genius text via fuzzy matching, using Genius
    as the canonical source for lyrics text.
    """
    lines: List[Line] = []
    used_genius = set()

    for i, (start_time, lrc_text) in enumerate(lrc_timings):
        if i + 1 < len(lrc_timings):
            end_time = lrc_timings[i + 1][0]
            if end_time - start_time > 10.0:
                end_time = start_time + 5.0
        else:
            end_time = start_time + 3.0

        # Find best matching Genius line
        best_match = None
        best_score = 0.0
        lrc_normalized = lrc_text.lower().strip()
        for j, genius_text in enumerate(genius_lines):
            if j in used_genius:
                continue
            genius_normalized = genius_text.lower().strip()
            score = SequenceMatcher(None, lrc_normalized, genius_normalized).ratio()
            if score > best_score:
                best_score = score
                best_match = (j, genius_text)

        line_text = best_match[1] if best_match and best_score > 0.5 else lrc_text
        if best_match:
            used_genius.add(best_match[0])

        word_texts = line_text.split()
        if not word_texts:
            continue

        line_duration = end_time - start_time
        word_count = len(word_texts)
        word_duration = (line_duration * 0.95) / word_count

        words = []
        for j, word_text in enumerate(word_texts):
            word_start = start_time + j * (line_duration / word_count)
            word_end = word_start + word_duration
            words.append(Word(text=word_text, start_time=word_start, end_time=word_end))

        # Skip duplicate consecutive lines
        line_text_str = " ".join([w.text for w in words]).strip()
        if lines and " ".join([w.text for w in lines[-1].words]).strip() == line_text_str:
            continue

        lines.append(Line(words=words))

    return lines


# ----------------------
# Line splitting for display
# ----------------------
def split_long_lines(lines: List[Line], max_width_ratio: float = 0.75) -> List[Line]:
    """Split lines that are too long for display.

    Args:
        lines: List of Line objects
        max_width_ratio: Maximum ratio of screen width (0.75 = 75%)

    Returns:
        List of Line objects with long lines split
    """
    max_chars = int(60 * max_width_ratio)
    result: List[Line] = []

    for line in lines:
        text = line.text
        if len(text) <= max_chars:
            result.append(line)
            continue

        words = line.words
        mid = len(words) // 2

        if mid == 0:
            result.append(line)
            continue

        first_words = words[:mid]
        first_line = Line(words=first_words, singer=line.singer)

        second_words = words[mid:]
        second_line = Line(words=second_words, singer=line.singer)

        result.append(first_line)
        result.append(second_line)

    return result
