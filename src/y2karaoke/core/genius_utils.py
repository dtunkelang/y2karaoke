"""
Genius-specific helper functions.

Contains utilities for:
- Detecting Genius metadata lines
- Stripping artist prefixes
- Filtering singer-only lines
- Resolving Genius URLs
"""

import re
from typing import List, Tuple, Optional

from .models import SongMetadata
from .text_utils import normalize_text, make_slug, clean_title_for_search

# ----------------------
# Constants / Patterns
# ----------------------
TITLE_CLEANUP_PATTERNS = [
    r'\s*[|｜]\s*.*$',  # Remove after | or ｜
    r'\s*[\(\[]?\s*(ft\.?|feat\.?|featuring).*?[\)\]]?\s*$',  # Featuring
    r'\s*[\(\[].*?[\)\]]\s*',  # Parentheses/brackets
]

YOUTUBE_SUFFIXES = [
    ' Lyrics', ' Official Video', ' Official Audio',
    ' Official Music Video', ' Audio', ' Video'
]

DESCRIPTION_PATTERNS = [
    'is the first track', 'is the second track', 'is the third track',
    'is a song by', 'was released as', 'Read More', 'studio album',
    'music video featuring'
]
DESCRIPTION_REGEX = re.compile("|".join(map(re.escape, DESCRIPTION_PATTERNS)), re.IGNORECASE)

TRANSLATION_LANGUAGES = ['Türkçe','Français','Español','Deutsch','Português']

# ----------------------
# Utility functions
# ----------------------
def _is_genius_metadata(line: str) -> bool:
    """Detect lines that are likely Genius metadata rather than lyrics."""
    if re.match(r'^\d+\s*Contributor', line):
        return True
    if 'Translations' in line and any(lang in line for lang in TRANSLATION_LANGUAGES):
        return True
    if DESCRIPTION_REGEX.search(line):
        return True
    if len(line) > 300:
        return True
    return False


def strip_leading_artist_from_line(text: str, artist: str) -> str:
    """Remove leading artist prefixes from a lyric line."""
    if not artist:
        return text
    pattern = re.compile(
        rf'^(?:\[{re.escape(artist)}\]\s*|{re.escape(artist)}\s*[-–]\s*)',
        re.IGNORECASE
    )
    return pattern.sub('', text).strip()


def filter_singer_only_lines(
    lines: List[Tuple[str, str]],
    known_singers: List[str]
) -> List[Tuple[str, str]]:
    """Remove lines that only contain known singer names."""
    known_set = {s.lower() for s in known_singers}
    filtered = []
    for text, singer in lines:
        text_clean = strip_leading_artist_from_line(text, artist='')
        parts = re.split(r'[\/,]', text_clean.lower())
        if any(p.strip() not in known_set for p in parts):
            filtered.append((text_clean, singer))
    return filtered


# ----------------------
# URL resolution
# ----------------------
def resolve_genius_url(title: str, artist: str) -> str:
    """
    Construct candidate Genius URLs for a song.

    Returns the first plausible URL as a string.
    """
    artist_slug = make_slug(artist)
    title_slug = make_slug(clean_title_for_search(title, TITLE_CLEANUP_PATTERNS, YOUTUBE_SUFFIXES))

    candidate_urls = [
        f"https://genius.com/{artist_slug}-{title_slug}-lyrics",
        f"https://genius.com/{title_slug}-lyrics",
        f"https://genius.com/Genius-romanizations-{artist_slug}-{title_slug}-romanized-lyrics",
    ]

    # Return the first candidate URL (the caller can validate existence)
    return candidate_urls[0]
