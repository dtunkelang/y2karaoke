# src/y2karaoke/core/lrc_utils.py

"""
LRC parsing utilities for karaoke lyrics.

Handles timestamp parsing, line extraction, and basic metadata filtering.
"""

import re
from typing import List, Tuple, Optional

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
# Metadata helper
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
# Parse individual LRC timestamp
# ----------------------
def parse_lrc_timestamp(ts: str) -> Optional[float]:
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

# ----------------------
# Extract plain text lines from LRC
# ----------------------
def extract_lyrics_text(lrc_text: str, title: str = "", artist: str = "") -> List[str]:
    """
    Extract plain text lines from LRC format (ignore timing and metadata).
    """
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

# ----------------------
# Parse LRC lines with timing
# ----------------------
def parse_lrc_with_timing(lrc_text: str, title: str = "", artist: str = "") -> List[Tuple[float, str]]:
    """
    Parse LRC format and extract (timestamp, text) tuples.
    Skips metadata lines.
    """
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
