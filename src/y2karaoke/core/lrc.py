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
    re.VERBOSE,
)


# ----------------------
# Metadata filtering
# ----------------------

# Prefixes that indicate metadata lines
_METADATA_PREFIXES = [
    "artist:",
    "song:",
    "title:",
    "album:",
    "writer:",
    "composer:",
    "lyricist:",
    "lyrics by",
    "written by",
    "produced by",
    "music by",
    "source:",
    "contributor:",
    "arranged by",
    "performed by",
    "vocals by",
    "music and lyrics",
    "words and music",
    # Chinese metadata
    "作词",
    "作詞",
    "作曲",
    "编曲",
    "編曲",
    "制作人",
    "演唱",
    "歌手",
    "词:",
    "曲:",
    "词 :",
    "曲 :",
    # Japanese metadata (romanized)
    "saku:",
    "saku :",
    "sakkyoku:",
    "sakkyoku :",
    "sakushi:",
    "sakushi :",
    "henkyoku:",
    "henkyoku :",
    "kashu:",
    "kashu :",
    # Korean metadata
    "작사:",
    "작곡:",
    "편곡:",
]

# Keywords that indicate metadata when followed by colon and name
_METADATA_KEYWORDS = [
    "composer",
    "lyricist",
    "writer",
    "arranger",
    "producer",
    "saku",
    "sakkyoku",
    "sakushi",
    "henkyoku",
]

# Patterns that indicate non-lyric content
_METADATA_PATTERNS = [
    "all rights reserved",
    "copyright",
    "℗",
    "©",
    "(instrumental)",
    "[instrumental]",
    "(intro)",
    "[intro]",
    "(outro)",
    "[outro]",
    "(verse)",
    "[verse]",
    "(chorus)",
    "[chorus]",
    "(bridge)",
    "[bridge]",
    "(hook)",
    "[hook]",
]

# Promo/CTA patterns commonly found in synced lyric dumps
_PROMO_PATTERNS = [
    r"https?://",
    r"www\.",
    r"@\w",
    r"\bmp3\b",
    r"\bsms\b",
    r"\bdownload\b",
    r"\bsubscribe\b",
    r"\bmember\b",
    r"\bring( ?)?tone\b",
    r"\bmobile\b",
    r"\bphone\b",
    r"\btext\b",
]

_PROMO_RE = re.compile("|".join(_PROMO_PATTERNS), re.IGNORECASE)

# Simple phone/shortcode pattern (e.g., "12345", "+33 6 12 34 56 78")
_PHONE_RE = re.compile(r"\+?\d[\d\s\-().]{4,}\d")


def _is_empty_or_symbols(text: str) -> bool:
    """Check if line is empty or contains only musical symbols."""
    if not text or not text.strip():
        return True
    return all(c in "♪🎵🎶♫♬-–—=_.·•" or c.isspace() for c in text)


def _has_metadata_prefix(text_lower: str) -> bool:
    """Check if line starts with a metadata prefix."""
    return any(text_lower.startswith(prefix) for prefix in _METADATA_PREFIXES)


def _has_metadata_keyword(text: str) -> bool:
    """Check if line contains metadata keyword pattern."""
    import re

    for keyword in _METADATA_KEYWORDS:
        if re.search(rf"\b{keyword}\s*:\s*[A-Z]", text, re.IGNORECASE):
            return True
    return False


def _has_metadata_pattern(text_lower: str) -> bool:
    """Check if line contains metadata patterns like copyright or section markers."""
    return any(pattern in text_lower for pattern in _METADATA_PATTERNS)


def _is_credit_pattern(text: str) -> bool:
    """Check if line matches contributor credit pattern."""
    import re

    match = re.match(r"^(\w+)\s*:\s*([A-Z][a-z]+(\s+[A-Z][a-z]+)*)\s*$", text.strip())
    return match is not None and len(match.group(1)) <= 15


def _is_promo_line(text: str) -> bool:
    """Check for common promo/CTA patterns in early LRC lines."""
    if not text:
        return False
    text_lower = text.lower()
    if _PROMO_RE.search(text_lower):
        return True
    if _PHONE_RE.search(text_lower):
        return True
    return False


def _is_promo_like_title_line(text: str, title: str) -> bool:
    """Heuristic: early full-sentence line that repeats the song title."""
    if not text or not title:
        return False
    text_lower = text.lower()
    title_lower = title.lower()

    if title_lower not in text_lower:
        return False
    if text_lower.count(title_lower) > 1:
        return True

    stripped = text_lower.lstrip(" \"'“”‘’(")
    starts_with_title = stripped.startswith(title_lower)
    has_quoted_title = re.search(
        rf"[\"“”].*{re.escape(title_lower)}.*[\"“”]", text_lower
    )
    has_sentence_punct = re.search(r"[.!?]$", text.strip()) is not None
    long_enough = len(text_lower) >= max(24, len(title_lower) + 8)

    if (
        not starts_with_title
        and (has_quoted_title or has_sentence_punct)
        and long_enough
    ):
        return True
    return False


def _normalize_for_comparison(text: str) -> str:
    """Normalize text for title/artist comparison."""
    return (
        text.lower().replace(" ", "").replace("'", "").replace("(", "").replace(")", "")
    )


def _is_title_header(text_lower: str, title: str, artist: str) -> bool:
    """Check if line is a title header."""
    title_norm = _normalize_for_comparison(title)
    line_norm = _normalize_for_comparison(text_lower)

    # Direct title match
    if line_norm == title_norm:
        return True
    # Title contained with little extra
    if title_norm in line_norm and len(line_norm) < len(title_norm) + 10:
        return True
    # Combined title-artist patterns
    if artist:
        artist_norm = _normalize_for_comparison(artist)
        if line_norm in (f"{title_norm}{artist_norm}", f"{artist_norm}{title_norm}"):
            return True
    return False


def _is_artist_header(text_lower: str, artist: str) -> bool:
    """Check if line is an artist header."""
    artist_norm = artist.lower().replace(" ", "").replace("ö", "o").replace("ü", "u")
    line_norm = text_lower.replace(" ", "").replace("ö", "o").replace("ü", "u")

    artist_variants = [
        artist_norm,
        artist_norm.replace("blueoystercult", "blueöystercult"),
        "blueöystercult",
        "blueoystercult",
        "blueoyster",
        "b.o.c",
        "boc",
    ]
    return any(line_norm == v or v in line_norm for v in artist_variants)


def _is_metadata_line(
    text: str,
    title: str = "",
    artist: str = "",
    timestamp: float = -1.0,
    filter_promos: bool = True,
) -> bool:
    """
    Determine if a line is metadata rather than actual lyrics.

    Args:
        text: The line text to check
        title: Song title (for filtering title-only header lines)
        artist: Artist name (for filtering artist-only header lines)
        timestamp: Line timestamp in seconds (-1 if unknown)
    """
    if _is_empty_or_symbols(text):
        return True

    text_lower = text.lower().strip()

    if _has_metadata_prefix(text_lower):
        return True
    if _has_metadata_keyword(text):
        return True
    if _has_metadata_pattern(text_lower):
        return True
    if _is_credit_pattern(text):
        return True

    # Only filter title/artist headers if they appear extremely early (<= 1s).
    # Song titles often appear in real lyrics, so avoid dropping them by default.
    is_header_time = timestamp <= 1.0
    if title and is_header_time and _is_title_header(text_lower, title, artist):
        return True
    if artist and is_header_time and _is_artist_header(text_lower, artist):
        return True

    # Filter obvious promo/CTA lines only at the very beginning.
    if filter_promos:
        if timestamp <= 12.0 and _is_promo_line(text):
            return True
        if timestamp <= 15.0 and _is_promo_like_title_line(text, title):
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


def parse_lrc_with_timing(
    lrc_text: str,
    title: str = "",
    artist: str = "",
    filter_promos: bool = True,
) -> List[Tuple[float, str]]:
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

        text_part = line[match.end() :].strip()
        if text_part and not _is_metadata_line(
            text_part, title, artist, timestamp, filter_promos=filter_promos
        ):
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
    filter_promos: bool = True,
) -> List[Line]:
    """Create Line objects from LRC text with evenly distributed word timings."""
    timed_lines = parse_lrc_with_timing(lrc_text, title, artist, filter_promos)
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
        if (
            lines
            and " ".join([w.text for w in lines[-1].words]).strip() == line_text_str
        ):
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
