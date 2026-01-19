"""
Text utilities for slug generation and title cleaning.

These are general-purpose helpers used across the y2karaoke core modules.
"""

import re
import unicodedata
from typing import List

# ----------------------
# Slug and title utilities
# ----------------------
def make_slug(text: str) -> str:
    """
    Convert text to a URL-friendly slug.

    - Normalizes Unicode
    - Lowercases
    - Removes punctuation
    - Replaces whitespace with dashes
    """
    text = unicodedata.normalize("NFKD", text)
    text = text.lower()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'\s+', '-', text)
    return text.strip('-')


def clean_title_for_search(
    title: str,
    title_cleanup_patterns: List[str],
    youtube_suffixes: List[str]
) -> str:
    """
    Clean a song title for searching.

    - Removes patterns like "(feat. ...)", "| ...", etc.
    - Strips common YouTube suffixes
    """
    cleaned = title
    for pattern in title_cleanup_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    for suffix in youtube_suffixes:
        if cleaned.endswith(suffix):
            cleaned = cleaned[:-len(suffix)].strip()
    return cleaned.strip()

def normalize_text(text: str) -> str:
    """
    Normalize text for fuzzy matching.
    - Lowercase
    - Remove accents
    - Remove punctuation
    - Collapse whitespace
    """
    if not text:
        return ""
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
