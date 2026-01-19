"""Genius lyrics fetching with singer annotations."""

import re
import unicodedata
from typing import List, Tuple, Optional
from ..utils.logging import get_logger
from .constants import TITLE_CLEANUP_PATTERNS, YOUTUBE_SUFFIXES, DESCRIPTION_REGEX, TRANSLATION_LANGUAGES

logger = get_logger(__name__)


# ----------------------
# Utility functions
# ----------------------
def _make_slug(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.lower()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'\s+', '-', text)
    return text.strip('-')


def _clean_title_for_search(title: str) -> str:
    cleaned = title
    for pattern in TITLE_CLEANUP_PATTERNS:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    for suffix in YOUTUBE_SUFFIXES:
        if cleaned.endswith(suffix):
            cleaned = cleaned[:-len(suffix)].strip()
    return cleaned.strip()


def _is_genius_metadata(line: str) -> bool:
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
    known_set = {s.lower() for s in known_singers}
    filtered = []
    for text, singer in lines:
        text_clean = strip_leading_artist_from_line(text, artist='')
        parts = re.split(r'[\/,]', text_clean.lower())
        if any(p.strip() not in known_set for p in parts):
            filtered.append((text_clean, singer))
    return filtered


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = unicodedata.normalize('NFKD', text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
