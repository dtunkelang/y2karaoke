"""Genius utility functions and constants."""

import re
import unicodedata
from typing import List, Tuple

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
# Helper functions
# ----------------------
def make_slug(text: str) -> str:
    """Convert text to lowercase slug suitable for URLs."""
    text = unicodedata.normalize("NFKD", text)
    text = text.lower()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'\s+', '-', text)
    return text.strip('-')


def clean_title_for_search(title: str, title_cleanup_patterns: list, youtube_suffixes: list) -> str:
    """Clean a song title for Genius search."""
    cleaned = title
    for pattern in title_cleanup_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    for suffix in youtube_suffixes:
        if cleaned.endswith(suffix):
            cleaned = cleaned[:-len(suffix)].strip()
    return cleaned.strip()


def is_genius_metadata(line: str) -> bool:
    """Determine if a line from Genius is metadata rather than lyrics."""
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
    """Remove leading artist labels from a lyric line."""
    if not artist:
        return text
    pattern = re.compile(
        rf'^(?:\[{re.escape(artist)}\]\s*|{re.escape(artist)}\s*[-–]\s*)', re.IGNORECASE
    )
    return pattern.sub('', text).strip()


def filter_singer_only_lines(
    lines: List[Tuple[str, str]],
    known_singers: List[str]
) -> List[Tuple[str, str]]:
    """Filter out lines that only mention singers and contain no lyrics."""
    known_set = {s.lower() for s in known_singers}
    filtered = []
    for text, singer in lines:
        text_clean = strip_leading_artist_from_line(text, artist='')
        parts = re.split(r'[\/,]', text_clean.lower())
        if any(p.strip() not in known_set for p in parts):
            filtered.append((text_clean, singer))
    return filtered


def normalize_text(text: str) -> str:
    """Normalize text for fuzzy matching."""
    if not text:
        return ""
    text = text.lower()
    text = unicodedata.normalize('NFKD', text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
