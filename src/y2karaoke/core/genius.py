"""Genius lyrics fetching with singer annotations."""

import re
import time
import random
import unicodedata
from difflib import SequenceMatcher
from typing import List, Tuple, Optional
import logging

import requests
from bs4 import BeautifulSoup

from ..utils.logging import get_logger
from .models import SingerID, Word, Line, SongMetadata
from .romanization import romanize_line

logger = get_logger(__name__)

# ----------------------
# Retry constants
# ----------------------
DEFAULT_MAX_RETRIES = 5
BACKOFF_FACTOR = 1.0

# ----------------------
# Patterns
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
def _make_request_with_retry(
    url: str,
    headers: Optional[dict] = None,
    timeout: int = 10,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_factor: float = BACKOFF_FACTOR
) -> Optional[requests.Response]:
    headers = headers or {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/143.0.0.0 Safari/537.36"
    }

    delay = backoff_factor
    for attempt in range(max_retries + 1):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request failed (attempt {attempt}/{max_retries}) for {url}: {e}")
            if attempt < max_retries:
                time.sleep(delay + random.uniform(0, 0.5))
                delay = min(delay * 2, 30.0)
        except Exception as e:
            logger.error(f"Unexpected error during request to {url}: {e}")
            break
    return None

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

# ----------------------
# Main lyrics fetching
# ----------------------
def fetch_genius_lyrics_with_singers(
    title: str,
    artist: str
) -> Tuple[Optional[List[Tuple[str, str]]], Optional[SongMetadata]]:
    """
    Fetch lyrics from Genius with singer annotations.

    Captures all lines, including those before any section markers.
    Strips artist prefixes and filters singer-only lines.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36',
        'Accept': '*/*'
    }

    cleaned_title = _clean_title_for_search(title)
    artist_slug = _make_slug(artist)
    title_slug = _make_slug(cleaned_title)

    candidate_urls = [
        f"https://genius.com/{artist_slug}-{title_slug}-lyrics",
        f"https://genius.com/{title_slug}-lyrics",
        f"https://genius.com/Genius-romanizations-{artist_slug}-{title_slug}-romanized-lyrics",
    ]

    # Try candidate URLs first
    song_url = None
    for url in candidate_urls:
        response = _make_request_with_retry(url, headers)
        if response and response.status_code == 200 and "translation" not in url.lower():
            song_url = url
            break

    # Fallback: search API
    if not song_url:
        search_queries = [f"{artist} {cleaned_title}", f"{cleaned_title} {artist}"]
        for query in search_queries:
            api_url = f"https://genius.com/api/search/song?per_page=5&q={query.replace(' ', '%20')}"
            response = _make_request_with_retry(api_url, headers)
            if not response:
                continue
            try:
                data = response.json()
                sections = data.get('response', {}).get('sections', [])
                for section in sections:
                    if section.get('type') == 'song':
                        for hit in section.get('hits', []):
                            result = hit.get('result', {})
                            url = result.get('url')
                            if url and url.endswith('-lyrics') and '/artists/' not in url:
                                song_url = url
                                break
                    if song_url:
                        break
            except Exception:
                continue
            if song_url:
                break

    if not song_url:
        return None, None

    # Fetch lyrics page
    response = _make_request_with_retry(song_url, headers)
    if not response or response.status_code != 200:
        return None, None

    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract Genius title/artist from page
    page_title = soup.find('title').get_text().strip() if soup.find('title') else ""
    genius_title = cleaned_title
    genius_artist = artist
    if page_title and "|" in page_title:
        parts = page_title.split("|")[0].split("–") if "–" in page_title else page_title.split("-")
        if len(parts) == 2:
            genius_artist = parts[0].strip()
            genius_title = parts[1].strip()
            if genius_title.endswith(" Lyrics"):
                genius_title = genius_title[:-7].strip()

    # Extract lyrics containers
    lyrics_containers = soup.find_all('div', {'data-lyrics-container': 'true'})
    if not lyrics_containers:
        return None, None

    lines_with_singers: List[Tuple[str, str]] = []
    current_singer = ""
    singers_found: set = set()
    section_pattern = re.compile(r'\[([^\]]+)\]')

    for container in lyrics_containers:
        # Remove structural metadata
        for elem in container.find_all(['div', 'span', 'a'], class_=lambda x: x and any(
            pattern in ' '.join(x) for pattern in ['LyricsHeader', 'SongBioPreview', 'ContributorsCredit']
        )):
            elem.decompose()

        for br in container.find_all('br'):
            br.replace_with('\n')

        text = container.get_text()
        for line in text.split('\n'):
            line = line.strip()
            if not line or _is_genius_metadata(line):
                continue

            # Section marker indicates singer
            section_match = section_pattern.match(line)
            if section_match and ':' in section_match.group(1):
                singer_part = section_match.group(1).split(':', 1)[1].strip()
                current_singer = singer_part
                singers_found.add(singer_part)
                continue

            # Append line regardless of current_singer
            lines_with_singers.append((line, current_singer))

    if not lines_with_singers:
        return None, None

    # Determine unique singers
    unique_singers: List[str] = []
    for singer in singers_found:
        if '&' in singer or ',' in singer:
            for part in re.split(r'[&,]', singer):
                name = part.strip()
                if name and name not in unique_singers:
                    unique_singers.append(name)
        elif singer and singer.lower() != 'both':
            if singer not in unique_singers:
                unique_singers.append(singer)

    is_duet = len(unique_singers) >= 2

    metadata = SongMetadata(
        singers=unique_singers[:2] if is_duet else [],
        is_duet=is_duet,
        title=genius_title,
        artist=genius_artist
    )

    # --- Strip artist prefixes ---
    lines_with_singers = [
        (strip_leading_artist_from_line(text, artist), singer)
        for text, singer in lines_with_singers
    ]

    # --- Filter singer-only lines ---
    lines_with_singers = filter_singer_only_lines(
        lines_with_singers,
        known_singers=metadata.singers if metadata else [artist]
    )

    return lines_with_singers, metadata


# ----------------------
# Lyrics merging
# ----------------------
def merge_lyrics_with_singer_info(
    timed_lines: List[Tuple[float, str]],
    genius_lines: List[Tuple[str, str]],
    metadata: SongMetadata,
    romanize: bool = True,
) -> List[Line]:
    if not timed_lines:
        return []

    genius_normalized = [(normalize_text(t), s) for t, s in genius_lines]
    used_genius_indices: set = set()
    lines: List[Line] = []

    for i, (start_time, text) in enumerate(timed_lines):
        line_text = romanize_line(text) if romanize else text
        end_time = timed_lines[i + 1][0] if i + 1 < len(timed_lines) else start_time + 3.0
        if i + 1 < len(timed_lines) and end_time - start_time > 10.0:
            end_time = start_time + 5.0

        # Fuzzy match to Genius line
        best_match_idx: Optional[int] = None
        best_score = 0.0
        text_norm = normalize_text(line_text)

        for j, (genius_norm, singer_name) in enumerate(genius_normalized):
            if j in used_genius_indices:
                continue
            score = SequenceMatcher(None, text_norm, genius_norm).ratio()
            if j not in used_genius_indices:
                score += 0.05
            if score > best_score and score > 0.5:
                best_score = score
                best_match_idx = j

        singer_id = ""
        if best_match_idx is not None:
            used_genius_indices.add(best_match_idx)
            _, singer_name = genius_lines[best_match_idx]
            singer_id = metadata.get_singer_id(singer_name)

        word_texts = line_text.split()
        if not word_texts:
            continue

        line_duration = end_time - start_time
        word_duration = max((line_duration * 0.95) / len(word_texts), 0.01)
        words: List[Word] = []

        for j, word_text in enumerate(word_texts):
            word_start = start_time + j * (line_duration / len(word_texts))
            word_end = word_start + word_duration
            words.append(Word(
                text=word_text,
                start_time=word_start,
                end_time=word_end,
                singer=singer_id,
            ))

        lines.append(Line(words=words, singer=singer_id))

    return lines

