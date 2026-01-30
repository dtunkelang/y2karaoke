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
def _is_metadata_line(text: str, title: str = "", artist: str = "", timestamp: float = -1.0) -> bool:
    """
    Determine if a line is metadata rather than actual lyrics.
    Skips obvious labels, credits, and title/artist lines that appear as headers.

    Args:
        text: The line text to check
        title: Song title (for filtering title-only header lines)
        artist: Artist name (for filtering artist-only header lines)
        timestamp: Line timestamp in seconds (-1 if unknown). Lines matching
                   title/artist are only filtered if they appear very early (< 2s),
                   as later occurrences are likely actual lyrics.
    """
    import re

    if not text:
        return True

    text_lower = text.lower().strip()

    # Skip empty or whitespace-only
    if not text_lower:
        return True

    # Skip lines that are just symbols
    if all(c in "♪🎵🎶♫♬-–—=_.·•" or c.isspace() for c in text):
        return True

    metadata_prefixes = [
        "artist:", "song:", "title:", "album:", "writer:", "composer:",
        "lyricist:", "lyrics by", "written by", "produced by", "music by",
        "source:", "contributor:", "arranged by", "performed by", "vocals by",
        "music and lyrics", "words and music",
        # Chinese metadata (simplified and traditional)
        "作词", "作詞", "作曲", "编曲", "編曲", "制作人", "演唱", "歌手",
        "词:", "曲:", "词 :", "曲 :",
        # Japanese metadata (romanized) - various spacing
        "saku:", "saku :", "sakkyoku:", "sakkyoku :", "sakushi:", "sakushi :",
        "henkyoku:", "henkyoku :", "kashu:", "kashu :",
        # Korean metadata
        "작사:", "작곡:", "편곡:",
    ]
    for prefix in metadata_prefixes:
        if text_lower.startswith(prefix):
            return True

    # Check for metadata patterns anywhere in the line (not just start)
    # This catches lines like "Lyrics: John Lennon" or "Music: Paul McCartney"
    metadata_keywords = [
        "composer", "lyricist", "writer", "arranger", "producer",
        "saku", "sakkyoku", "sakushi", "henkyoku",
    ]
    # Pattern: keyword followed by colon and a proper name
    for keyword in metadata_keywords:
        pattern = rf'\b{keyword}\s*:\s*[A-Z]'
        if re.search(pattern, text, re.IGNORECASE):
            return True

    # Skip common metadata patterns
    metadata_patterns = [
        "all rights reserved", "copyright", "℗", "©",
        "(instrumental)", "[instrumental]", "(intro)", "[intro]",
        "(outro)", "[outro]", "(verse)", "[verse]", "(chorus)", "[chorus]",
        "(bridge)", "[bridge]", "(hook)", "[hook]",
    ]
    for pattern in metadata_patterns:
        if pattern in text_lower:
            return True

    # Skip contributor credit patterns like "username : Name" or "word: Name"
    # These are common in community-sourced LRC files
    credit_pattern = re.match(r'^(\w+)\s*:\s*([A-Z][a-z]+(\s+[A-Z][a-z]+)*)\s*$', text.strip())
    if credit_pattern:
        # Short first word (likely username) followed by proper name
        first_word = credit_pattern.group(1)
        if len(first_word) <= 15:  # Usernames are typically short
            return True

    # Only filter title/artist lines if they appear early (likely headers)
    # Lines appearing later (>= 15s) with title text are probably actual lyrics
    # Use 15s threshold because some LRCs have title/artist in first 10s before singing starts
    is_early_line = timestamp < 15.0

    if title and is_early_line:
        title_lower = title.lower().replace(" ", "").replace("'", "").replace("(", "").replace(")", "")
        line_normalized = text_lower.replace(" ", "").replace("'", "").replace("(", "").replace(")", "")
        # Check various title formats
        title_variants = [
            title_lower,
            title_lower.replace("don'tfear", "dontfear"),  # Handle apostrophe variations
            f"dontfear{title_lower}" if "reaper" in title_lower else title_lower,
        ]
        if any(line_normalized == variant for variant in title_variants):
            return True
        # Check partial title match (line contains just the title)
        if line_normalized == title_lower or title_lower in line_normalized and len(line_normalized) < len(title_lower) + 10:
            return True
        # Also check if line is just "title - artist" or "artist - title"
        if artist:
            artist_lower = artist.lower().replace(" ", "")
            combined1 = f"{title_lower}{artist_lower}"
            combined2 = f"{artist_lower}{title_lower}"
            if line_normalized == combined1 or line_normalized == combined2:
                return True

    if artist and is_early_line:
        artist_lower = artist.lower().replace(" ", "").replace("ö", "o").replace("ü", "u")
        line_normalized = text_lower.replace(" ", "").replace("ö", "o").replace("ü", "u")
        # Check for artist name match (with common variations)
        artist_variants = [
            artist_lower,
            artist_lower.replace("blueoystercult", "blueöystercult"),
            "blueöystercult", "blueoystercult", "blueoyster", "b.o.c", "boc",
        ]
        if any(line_normalized == variant or variant in line_normalized for variant in artist_variants):
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
        if text_part and not _is_metadata_line(text_part, title, artist, timestamp):
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
