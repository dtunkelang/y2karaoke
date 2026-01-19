"""
Genius-specific utility functions and constants.
"""

import re
from typing import Optional
from .fetch import fetch_html, fetch_json

# ----------------------
# Constants for title cleaning
# ----------------------
TITLE_CLEANUP_PATTERNS = [
    r"\(feat\..*?\)",
    r"\(ft\..*?\)",
    r"\(with .*?\)",
    r"\[.*?\]",
    r"\|.*",
]

YOUTUBE_SUFFIXES = [
    "official video",
    "official music video",
    "lyrics",
]

# ----------------------
# Helper functions
# ----------------------
def is_genius_metadata(line: str) -> bool:
    """
    Detect lines that are Genius metadata or section headers
    that should be ignored for karaoke purposes.
    """
    if not line:
        return True
    line_lower = line.lower()
    if line_lower.startswith(("intro", "verse", "chorus", "bridge", "outro", "pre-chorus")):
        return True
    if line_lower in ("instrumental", "music", "spoken"):
        return True
    if re.match(r'^\[.*\]$', line):
        return True
    return False


def resolve_genius_url(artist_slug: str, title_slug: str, headers: Optional[dict] = None) -> Optional[str]:
    """
    Attempt to resolve the best Genius URL for a song.

    Skips translated pages and prefers romanized / English pages.
    """
    candidate_urls = [
        f"https://genius.com/{artist_slug}-{title_slug}-lyrics",
        f"https://genius.com/{title_slug}-lyrics",
        f"https://genius.com/Genius-romanizations-{artist_slug}-{title_slug}-romanized-lyrics",
    ]

    for url in candidate_urls:
        html = fetch_html(url, headers=headers)
        if html and "translation" not in url.lower():
            return url

    # Fallback: could implement Genius search API here if needed
    return None
