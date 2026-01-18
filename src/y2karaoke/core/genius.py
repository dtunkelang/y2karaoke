"""Genius lyrics fetching with singer annotations."""

import re
import time
import random
import unicodedata
from difflib import SequenceMatcher
from typing import List, Tuple, Optional

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


def _make_request_with_retry(
    url: str,
    headers: Optional[dict] = None,
    timeout: int = 10,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> Optional[requests.Response]:
    """
    Make an HTTP GET request with retry logic and exponential backoff.
    Returns the requests.Response or None if all retries fail.
    """
    headers = headers or {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/143.0.0.0 Safari/537.36"
    }

    delay = 1.0

    for attempt in range(max_retries + 1):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response
        except (requests.exceptions.RequestException, requests.exceptions.Timeout):
            if attempt < max_retries:
                sleep_time = delay + random.uniform(0, 0.5)
                time.sleep(sleep_time)
                delay = min(delay * 2, 30.0)
    return None


def _make_slug(text: str) -> str:
    """Convert text to URL slug for Genius URLs."""
    slug = re.sub(r'[^\w\s-]', '', text.lower())
    slug = re.sub(r'\s+', '-', slug)
    return slug


def _clean_title_for_search(title: str) -> str:
    """Clean a title for Genius search by removing common suffixes and noise."""
    cleaned = re.split(r'\s*[|｜]\s*', title)[0]
    cleaned = re.sub(r'\s*[\(\[]?\s*(ft\.?|feat\.?|featuring).*?[\)\]]?\s*$', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s*[\(\[].*?[\)\]]\s*', '', cleaned).strip()

    # Remove common YouTube suffixes
    for suffix in [' Lyrics', ' Official Video', ' Official Audio', ' Official Music Video', ' Audio', ' Video']:
        if cleaned.endswith(suffix):
            cleaned = cleaned[:-len(suffix)].strip()

    return cleaned


def _is_genius_metadata(line: str) -> bool:
    """Check if a line is Genius page metadata rather than lyrics."""
    # Skip lines with contributor counts
    if re.match(r'^\d+\s*Contributor', line):
        return True
    # Skip lines with translation language lists
    if 'Translations' in line and any(lang in line for lang in ['Türkçe', 'Français', 'Español', 'Deutsch', 'Português']):
        return True
    # Skip description text patterns
    description_patterns = [
        'is the first track',
        'is the second track',
        'is the third track',
        'is a song by',
        'was released as',
        'Read More',
        'studio album',
        'music video featuring',
    ]
    if any(pattern in line for pattern in description_patterns):
        return True
    # Skip if line is extremely long (likely concatenated metadata)
    if len(line) > 300:
        return True
    return False


def fetch_genius_lyrics_with_singers(
    title: str,
    artist: str
) -> Tuple[Optional[List[Tuple[str, str]]], Optional[SongMetadata]]:
    """
    Fetch lyrics from Genius with singer annotations.

    Args:
        title: Song title
        artist: Artist name

    Returns:
        Tuple of (lyrics_with_singers, metadata)
        - lyrics_with_singers: List of (text, singer_name) tuples for each line
        - metadata: SongMetadata with singer info and correct title/artist from Genius
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36',
        'Accept': '*/*'
    }

    cleaned_title = _clean_title_for_search(title)
    artist_slug = _make_slug(artist)
    title_slug = _make_slug(cleaned_title)

    # Try direct URL patterns first
    candidate_urls = [
        f"https://genius.com/{artist_slug}-{title_slug}-lyrics",
        f"https://genius.com/{title_slug}-lyrics",
        f"https://genius.com/Genius-romanizations-{artist_slug}-{title_slug}-romanized-lyrics",
    ]

    song_url = None
    for url in candidate_urls:
        response = _make_request_with_retry(url, headers)
        if response and response.status_code == 200 and "translation" not in url.lower():
            song_url = url
            break

    # Fallback: search API if no URL worked
    if not song_url:
        search_queries = [
            f"{artist} {cleaned_title}",
            f"{cleaned_title} {artist}"
        ]
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

    # Fetch page
    response = _make_request_with_retry(song_url, headers)
    if not response or response.status_code != 200:
        return None, None

    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract title/artist from page
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
    lyrics_started = False  # Track if we've hit actual lyrics content

    for container in lyrics_containers:
        # Remove structural elements that contain metadata, not lyrics
        # These are identified by their class names in the Genius HTML structure
        for elem in container.find_all(['div', 'span', 'a'], class_=lambda x: x and any(
            pattern in ' '.join(x) for pattern in [
                'LyricsHeader',      # Contains contributor count and title
                'SongBioPreview',    # Contains song description/bio
                'ContributorsCredit', # Contributor information
            ]
        )):
            elem.decompose()

        for br in container.find_all('br'):
            br.replace_with('\n')
        text = container.get_text()
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue

            # Check for section marker - this indicates start of actual lyrics
            section_match = section_pattern.match(line)
            if section_match:
                lyrics_started = True
                header = section_match.group(1)
                if ':' in header:
                    singer_part = header.split(':', 1)[1].strip()
                    current_singer = singer_part
                    singers_found.add(singer_part)
                continue

            # Skip everything before the first section marker (bio/description text)
            if not lyrics_started:
                continue

            # Skip Genius metadata lines that might appear within lyrics
            if _is_genius_metadata(line):
                continue

            lines_with_singers.append((line, current_singer))

    if not lines_with_singers:
        return None, None

    # Determine unique singers
    unique_singers: List[str] = []
    for singer in singers_found:
        if '&' in singer or ',' in singer:
            parts = re.split(r'[&,]', singer)
            for part in parts:
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

    return lines_with_singers, metadata


def normalize_text(text: str) -> str:
    """
    Normalize text for fuzzy matching:
    - Lowercase
    - Remove diacritics
    - Remove punctuation
    - Collapse whitespace
    """
    if not text:
        return ""
    text = text.lower()
    text = unicodedata.normalize('NFKD', text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def merge_lyrics_with_singer_info(
    timed_lines: List[Tuple[float, str]],
    genius_lines: List[Tuple[str, str]],
    metadata: SongMetadata,
    romanize: bool = True,
) -> List[Line]:
    """
    Merge timed lyrics with Genius singer annotations using fuzzy matching.

    Args:
        timed_lines: List of (timestamp, text) tuples from synced lyrics
        genius_lines: List of (text, singer_name) tuples from Genius
        metadata: SongMetadata for singer ID mapping
        romanize: Whether to romanize the text

    Returns:
        List of Line objects with timing and singer info
    """
    if not timed_lines:
        return []

    # Normalize Genius lines for matching
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
