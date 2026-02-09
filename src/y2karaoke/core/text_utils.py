"""
Text utilities for slug generation, normalization, and karaoke-specific line cleaning.

These are general-purpose helpers used across the y2karaoke core modules.
"""

import re
import unicodedata
from typing import List, Tuple


# ----------------------
# Slug and title utilities
# ----------------------
def make_slug(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"\s+", "-", text)
    return text.strip("-")


def clean_title_for_search(
    title: str, title_cleanup_patterns: List[str], youtube_suffixes: List[str]
) -> str:
    cleaned = title
    for pattern in title_cleanup_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    for suffix in youtube_suffixes:
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)].strip()
    return cleaned.strip()


# ----------------------
# Genius-specific line helpers
# ----------------------
def strip_leading_artist_from_line(text: str, artist: str) -> str:
    if not artist:
        return text
    pattern = re.compile(
        rf"^(?:\[{re.escape(artist)}\]\s*|{re.escape(artist)}\s*[-–]\s*)", re.IGNORECASE
    )
    return pattern.sub("", text).strip()


def filter_singer_only_lines(
    lines: List[Tuple[str, str]], known_singers: List[str]
) -> List[Tuple[str, str]]:
    known_set = {s.lower() for s in known_singers}
    filtered = []
    for text, singer in lines:
        text_clean = strip_leading_artist_from_line(text, artist="")
        parts = re.split(r"[\/,]", text_clean.lower())
        if any(p.strip() not in known_set for p in parts):
            filtered.append((text_clean, singer))
    return filtered


def normalize_title(title: str, remove_stopwords: bool = False) -> str:
    """Normalize title for comparison.

    Args:
        title: Title string to normalize
        remove_stopwords: If True, remove common stopwords (the, el, los, etc.)
    """
    normalized = re.sub(r"[,.\-:;\'\"!?()]", " ", title.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()

    if remove_stopwords:
        words = [w for w in normalized.split() if w not in STOP_WORDS]
        normalized = " ".join(words)

    return normalized


STOP_WORDS = {
    # English
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "with",
    "in",
    "to",
    "for",
    "by",
    "&",
    "+",
    # Spanish
    "el",
    "la",
    "los",
    "las",
    "un",
    "una",
    "unos",
    "unas",
    "y",
    "de",
    "del",
    "con",
    # French
    "le",
    "la",
    "les",
    "un",
    "une",
    "des",
    "et",
    "de",
    "du",
    "au",
    "aux",
    # German
    "der",
    "die",
    "das",
    "ein",
    "eine",
    "und",
    "von",
    "mit",
    # Italian
    "il",
    "lo",
    "la",
    "i",
    "gli",
    "le",
    "un",
    "uno",
    "una",
    "e",
    "di",
    "del",
    "della",
    # Portuguese
    "o",
    "a",
    "os",
    "as",
    "um",
    "uma",
    "uns",
    "umas",
    "e",
    "de",
    "do",
    "da",
    "dos",
    "das",
}
