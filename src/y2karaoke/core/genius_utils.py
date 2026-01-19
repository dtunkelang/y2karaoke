"""Genius utility functions and constants."""

import re
import unicodedata
from typing import List, Tuple, Optional
from .fetch import fetch_html, fetch_json

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
    text = unicodedata.normalize("NFKD", text)
    text = text.lower()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'\s+', '-', text)
    return text.strip('-')


def clean_title_for_search(title: str, title_cleanup_patterns: list, youtube_suffixes: list) -> str:
    cleaned = title
    for pattern in title_cleanup_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    for suffix in youtube_suffixes:
        if cleaned.endswith(suffix):
            cleaned = cleaned[:-len(suffix)].strip()
    return cleaned.strip()


def is_genius_metadata(line: str) -> bool:
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


# ----------------------
# URL resolution
# ----------------------
def resolve_genius_url(title: str, artist: str, romanized: bool = True) -> Optional[str]:
    from .genius_utils import make_slug, clean_title_for_search, TITLE_CLEANUP_PATTERNS, YOUTUBE_SUFFIXES

    cleaned_title = clean_title_for_search(title, TITLE_CLEANUP_PATTERNS, YOUTUBE_SUFFIXES)
    artist_slug = make_slug(artist)
    title_slug = make_slug(cleaned_title)

    candidate_urls = [
        f"https://genius.com/{artist_slug}-{title_slug}-lyrics",
        f"https://genius.com/{title_slug}-lyrics",
    ]
    if romanized:
        candidate_urls.insert(0, f"https://genius.com/Genius-romanizations-{artist_slug}-{title_slug}-romanized-lyrics")

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                      'AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/143.0.0.0 Safari/537.36',
        'Accept': '*/*'
    }

    # Try candidate URLs first
    for url in candidate_urls:
        html = fetch_html(url, headers=headers)
        if html and not any(tag in url.lower() for tag in ["translation", "übersetzung"]):
            return url

    # Fallback: Genius API search
    search_queries = [f"{artist} {cleaned_title}", f"{cleaned_title} {artist}"]
    for query in search_queries:
        api_url = f"https://genius.com/api/search/song?per_page=5&q={query.replace(' ', '%20')}"
        data = fetch_json(api_url, headers=headers)
        if not data:
            continue

        sections = data.get('response', {}).get('sections', [])
        for section in sections:
            if section.get('type') != 'song':
                continue
            for hit in section.get('hits', []):
                result = hit.get('result', {})
                url = result.get('url')
                if url and url.endswith("-lyrics") and '/artists/' not in url and \
                   not any(tag in url.lower() for tag in ["translation", "übersetzung"]):
                    return url

    return None
