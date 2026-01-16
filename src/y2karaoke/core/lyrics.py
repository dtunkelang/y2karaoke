"""Lyrics fetching with forced alignment for accurate word-level timing."""

import re
from dataclasses import dataclass
from typing import Optional, Dict, Any
from pathlib import Path

import json
import os

import syncedlyrics

from ..exceptions import LyricsError
from ..utils.logging import get_logger
from ..utils.retry import retry_with_backoff, DEFAULT_MAX_RETRIES

logger = get_logger(__name__)

# Network exceptions to retry
try:
    import requests
    NETWORK_EXCEPTIONS = (
        requests.exceptions.RequestException,
        requests.exceptions.Timeout,
        requests.exceptions.ConnectionError,
    )
except ImportError:
    NETWORK_EXCEPTIONS = (Exception,)


class LyricsProcessor:
    """Process lyrics with timing and romanization."""
    
    def __init__(self):
        self._setup_romanizers()
    
    def _setup_romanizers(self):
        """Setup optional romanization libraries."""
        pass  # The original code handles this with global imports
    
    def get_lyrics(
        self, 
        title: str, 
        artist: str, 
        vocals_path: str, 
        cache_dir: str,
        force: bool = False
    ) -> Dict[str, Any]:
        """Get lyrics with timing information."""
        
        try:
            lines, metadata = get_lyrics(title, artist, vocals_path, cache_dir)
            return {'lines': lines, 'metadata': metadata}
        except Exception as e:
            raise LyricsError(f"Could not get lyrics: {e}")


# Try to import Korean romanizer
try:
    from korean_romanizer.romanizer import Romanizer
    KOREAN_ROMANIZER_AVAILABLE = True
except ImportError:
    KOREAN_ROMANIZER_AVAILABLE = False

# Try to import Chinese romanizer (pinyin)
try:
    from pypinyin import lazy_pinyin, Style
    CHINESE_ROMANIZER_AVAILABLE = True
except ImportError:
    CHINESE_ROMANIZER_AVAILABLE = False

# Try to import Japanese romanizer (kana/kanji -> romaji)
try:
    from pykakasi import kakasi
    JAPANESE_ROMANIZER_AVAILABLE = True
except ImportError:
    JAPANESE_ROMANIZER_AVAILABLE = False

# Try to import Arabic/Hebrew romanizer
try:
    import pyarabic.araby as araby
    ARABIC_ROMANIZER_AVAILABLE = True
except ImportError:
    ARABIC_ROMANIZER_AVAILABLE = False

HEBREW_ROMANIZER_AVAILABLE = True  # We'll use a simple mapping


def contains_korean(text: str) -> bool:
    """Check if text contains Korean characters (Hangul)."""
    # Korean Unicode ranges: Hangul Syllables (AC00-D7AF), Hangul Jamo, etc.
    for char in text:
        if '\uac00' <= char <= '\ud7af':  # Hangul Syllables
            return True
        if '\u1100' <= char <= '\u11ff':  # Hangul Jamo
            return True
        if '\u3130' <= char <= '\u318f':  # Hangul Compatibility Jamo
            return True
    return False


def romanize_korean(text: str) -> str:
    """
    Romanize Korean text while preserving non-Korean parts.

    Uses the Revised Romanization of Korean standard.
    """
    if not KOREAN_ROMANIZER_AVAILABLE:
        return text

    if not contains_korean(text):
        return text

    try:
        # The romanizer works on the whole string
        romanizer = Romanizer(text)
        return romanizer.romanize()
    except Exception:
        # If romanization fails, return original
        return text


def contains_chinese(text: str) -> bool:
    """Check if text contains Chinese (CJK) characters."""
    # Basic CJK Unified Ideographs block; this covers most common Han characters.
    for char in text:
        if "\u4e00" <= char <= "\u9fff":
            return True
    return False


def romanize_chinese(text: str) -> str:
    """Romanize Chinese text (Han characters) while preserving other scripts.

    Uses pypinyin to generate Pinyin without tone marks for readability.
    Non-Chinese characters are passed through unchanged.
    """
    if not CHINESE_ROMANIZER_AVAILABLE:
        return text

    if not contains_chinese(text):
        return text

    # Build result by chunking consecutive Chinese characters so we can
    # insert spaces between syllables but keep non-Chinese regions intact.
    result_parts: list[str] = []
    chinese_buffer: list[str] = []

    def flush_chinese() -> None:
        if not chinese_buffer:
            return
        segment = "".join(chinese_buffer)
        chinese_buffer.clear()
        try:
            # lazy_pinyin returns a list of syllables; join with spaces
            pinyin_list = lazy_pinyin(segment, style=Style.NORMAL)
            result_parts.append(" ".join(pinyin_list))
        except Exception:
            # On any pypinyin failure, fall back to the original segment
            result_parts.append(segment)

    for ch in text:
        if contains_chinese(ch):
            chinese_buffer.append(ch)
        else:
            flush_chinese()
            result_parts.append(ch)

    flush_chinese()
    return "".join(result_parts)


def contains_japanese(text: str) -> bool:
    """Check if text contains Japanese characters (Hiragana/Katakana)."""
    for ch in text:
        # Hiragana
        if "\u3040" <= ch <= "\u309f":
            return True
        # Katakana (including Katakana Phonetic Extensions)
        if "\u30a0" <= ch <= "\u30ff" or "\u31f0" <= ch <= "\u31ff":
            return True
    return False


_JAPANESE_CONVERTER = None


def romanize_japanese(text: str) -> str:
    """Romanize Japanese text (kana/kanji) while preserving other scripts.

    Uses pykakasi to generate romaji. Non-Japanese characters are passed
    through unchanged.
    """
    global _JAPANESE_CONVERTER

    if not JAPANESE_ROMANIZER_AVAILABLE:
        return text

    if not contains_japanese(text):
        return text

    try:
        if _JAPANESE_CONVERTER is None:
            _JAPANESE_CONVERTER = kakasi()
        
        result = _JAPANESE_CONVERTER.convert(text)
        # Join with spaces to preserve word boundaries
        return " ".join([item["hepburn"] for item in result])
    except Exception:
        # On any pykakasi failure, fall back to the original
        return text


def contains_arabic(text: str) -> bool:
    """Check if text contains Arabic characters."""
    for ch in text:
        # Arabic Unicode range
        if "\u0600" <= ch <= "\u06ff" or "\u0750" <= ch <= "\u077f":
            return True
    return False


def contains_hebrew(text: str) -> bool:
    """Check if text contains Hebrew characters."""
    for ch in text:
        # Hebrew Unicode range
        if "\u0590" <= ch <= "\u05ff":
            return True
    return False


def romanize_arabic(text: str) -> str:
    """Romanize Arabic text to Latin script."""
    if not ARABIC_ROMANIZER_AVAILABLE:
        return text
    
    if not contains_arabic(text):
        return text
    
    try:
        # Simple Arabic to Latin transliteration mapping
        arabic_to_latin = {
            'ا': 'a', 'ب': 'b', 'ت': 't', 'ث': 'th', 'ج': 'j', 'ح': 'h', 'خ': 'kh',
            'د': 'd', 'ذ': 'dh', 'ر': 'r', 'ز': 'z', 'س': 's', 'ش': 'sh', 'ص': 's',
            'ض': 'd', 'ط': 't', 'ظ': 'z', 'ع': 'a', 'غ': 'gh', 'ف': 'f', 'ق': 'q',
            'ك': 'k', 'ل': 'l', 'م': 'm', 'ن': 'n', 'ه': 'h', 'و': 'w', 'ي': 'y',
            'ى': 'a', 'ة': 'h', 'ء': '', 'ئ': '', 'ؤ': 'w', 'إ': 'i', 'أ': 'a',
            'آ': 'aa', 'َ': 'a', 'ُ': 'u', 'ِ': 'i', 'ّ': '', 'ْ': ''
        }
        
        result = []
        for char in text:
            if char in arabic_to_latin:
                result.append(arabic_to_latin[char])
            else:
                result.append(char)
        return ''.join(result)
    except Exception:
        return text


def romanize_hebrew(text: str) -> str:
    """Romanize Hebrew text to Latin script."""
    if not HEBREW_ROMANIZER_AVAILABLE:
        return text
    
    if not contains_hebrew(text):
        return text
    
    try:
        # Simple Hebrew to Latin transliteration mapping
        hebrew_to_latin = {
            'א': 'a', 'ב': 'b', 'ג': 'g', 'ד': 'd', 'ה': 'h', 'ו': 'v', 'ז': 'z',
            'ח': 'ch', 'ט': 't', 'י': 'y', 'כ': 'k', 'ך': 'kh', 'ל': 'l', 'מ': 'm',
            'ם': 'm', 'נ': 'n', 'ן': 'n', 'ס': 's', 'ע': 'a', 'פ': 'p', 'ף': 'f',
            'צ': 'ts', 'ץ': 'ts', 'ק': 'k', 'ר': 'r', 'ש': 'sh', 'ת': 't'
        }
        
        result = []
        for char in text:
            if char in hebrew_to_latin:
                result.append(hebrew_to_latin[char])
            else:
                result.append(char)
        return ''.join(result)
    except Exception:
        return text


def romanize_line(text: str) -> str:
    """Romanize a line of lyrics, handling mixed Korean/Japanese/Chinese/Arabic/Hebrew/English text."""
    # Apply Korean romanization first
    if contains_korean(text):
        text = romanize_korean(text)

    # Then apply Japanese romanization
    if contains_japanese(text):
        text = romanize_japanese(text)

    # Apply Chinese romanization (including any remaining Han characters)
    if contains_chinese(text):
        text = romanize_chinese(text)
    
    # Apply Arabic romanization
    if contains_arabic(text):
        text = romanize_arabic(text)
    
    # Apply Hebrew romanization
    if contains_hebrew(text):
        text = romanize_hebrew(text)

    # Clean up common issues: collapse repeated whitespace
    text = " ".join(text.split())
    return text


@dataclass
class Word:
    """A word with timing information."""
    text: str
    start_time: float  # seconds
    end_time: float    # seconds
    singer: str = ""   # Singer identifier (e.g., "singer1", "singer2", "both")


@dataclass
class Line:
    """A line of lyrics with words."""
    words: list[Word]
    start_time: float
    end_time: float
    singer: str = ""   # Singer for this line (e.g., "singer1", "singer2", "both")

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)


@dataclass
@dataclass
class SongMetadata:
    """Metadata about singers in a song and correct title/artist from Genius."""
    singers: list[str]  # List of singer names in order (e.g., ["Bruno Mars", "Lady Gaga"])
    is_duet: bool = False
    title: Optional[str] = None  # Correct title from Genius
    artist: Optional[str] = None  # Correct artist from Genius

    def get_singer_id(self, singer_name: str) -> str:
        """Convert singer name to singer ID (singer1, singer2, both)."""
        if not singer_name:
            return ""

        name_lower = singer_name.lower()

        # Check for "both" or "&" indicators
        if "&" in name_lower or "both" in name_lower or "," in name_lower:
            return "both"

        # Match against known singers
        for i, known_singer in enumerate(self.singers):
            if known_singer.lower() in name_lower or name_lower in known_singer.lower():
                return f"singer{i + 1}"

        # Default to first singer if no match
        return "singer1" if self.singers else ""


def parse_lrc_timestamp(ts: str) -> float:
    """Parse LRC timestamp [mm:ss.xx] to seconds."""
    match = re.match(r'\[(\d+):(\d+)\.(\d+)\]', ts)
    if match:
        minutes = int(match.group(1))
        seconds = int(match.group(2))
        centiseconds = int(match.group(3))
        return minutes * 60 + seconds + centiseconds / 100
    return 0.0


def extract_lyrics_text(lrc_text: str, title: str = "", artist: str = "") -> list[str]:
    """Extract plain text lines from LRC format (no timing), filtering metadata."""
    lines = []
    for line in lrc_text.strip().split('\n'):
        match = re.match(r'\[\d+:\d+\.\d+\]\s*(.*)', line)
        if match:
            text = match.group(1).strip()
            if text and not _is_metadata_line(text, title, artist):
                lines.append(text)
    return lines


def _is_metadata_line(text: str, title: str = "", artist: str = "") -> bool:
    """Check if a line is metadata rather than actual lyrics."""
    text_lower = text.lower().strip()

    # Skip lines that are obviously metadata labels
    metadata_prefixes = [
        "artist:", "song:", "title:", "album:", "writer:", "composer:",
        "lyricist:", "lyrics by", "written by", "produced by", "music by",
    ]
    for prefix in metadata_prefixes:
        if text_lower.startswith(prefix):
            return True

    # Skip lines that are just the artist or title name (with some flexibility)
    if title:
        title_lower = title.lower()
        # Check if line is just the title (possibly with minor differences)
        if text_lower == title_lower or text_lower.replace(" ", "") == title_lower.replace(" ", ""):
            return True

    if artist:
        artist_lower = artist.lower()
        # Check if line is just the artist name
        if text_lower == artist_lower or text_lower.replace(" ", "") == artist_lower.replace(" ", ""):
            return True

    return False


def parse_lrc_with_timing(lrc_text: str, title: str = "", artist: str = "") -> list[tuple[float, str]]:
    """
    Parse LRC format to extract lines with timestamps.

    Args:
        lrc_text: Raw LRC text content
        title: Song title (used to filter metadata lines)
        artist: Artist name (used to filter metadata lines)

    Returns:
        List of (timestamp_seconds, text) tuples
    """
    lines = []
    for line in lrc_text.strip().split('\n'):
        match = re.match(r'\[(\d+):(\d+)\.(\d+)\]\s*(.*)', line)
        if match:
            minutes = int(match.group(1))
            seconds = int(match.group(2))
            centiseconds = int(match.group(3))
            timestamp = minutes * 60 + seconds + centiseconds / 100
            text = match.group(4).strip()
            if text and not _is_metadata_line(text, title, artist):
                lines.append((timestamp, text))
    return lines


def extract_artists_from_title(title: str, known_artist: str) -> list[str]:
    """
    Extract artist names from a title like "Artist1, Artist2 - Song Name".

    Args:
        title: The full title (e.g., "Lady Gaga, Bruno Mars - Die With A Smile")
        known_artist: The artist we already know (from metadata)

    Returns:
        List of artist names found
    """
    artists = []

    # Check if title has "Artist - Song" format
    if " - " in title:
        artist_part = title.split(" - ")[0].strip()

        # Split by common separators: comma, ampersand, "and", "ft.", "feat."
        parts = re.split(r'[,&]|\b(?:and|ft\.?|feat\.?)\b', artist_part, flags=re.IGNORECASE)
        artists = [p.strip() for p in parts if p.strip()]

    # If we couldn't extract from title, use known_artist
    if not artists:
        artists = [known_artist]

    return artists


def _lines_to_json(lines: list[Line]) -> list[dict]:
    data: list[dict] = []
    for line in lines:
        data.append({
            "start_time": line.start_time,
            "end_time": line.end_time,
            "singer": line.singer,
            "words": [
                {
                    "text": w.text,
                    "start_time": w.start_time,
                    "end_time": w.end_time,
                    "singer": w.singer,
                }
                for w in line.words
            ],
        })
    return data


def _lines_from_json(data: list[dict]) -> list[Line]:
    lines: list[Line] = []
    for item in data:
        words = [
            Word(
                text=w["text"],
                start_time=float(w["start_time"]),
                end_time=float(w["end_time"]),
                singer=w.get("singer", ""),
            )
            for w in item.get("words", [])
        ]
        lines.append(Line(
            words=words,
            start_time=float(item["start_time"]),
            end_time=float(item["end_time"]),
            singer=item.get("singer", ""),
        ))
    return lines


def _metadata_to_json(metadata: Optional[SongMetadata]) -> Optional[dict]:
    if metadata is None:
        return None
    return {
        "singers": metadata.singers,
        "is_duet": metadata.is_duet,
    }


def _metadata_from_json(data: Optional[dict]) -> Optional[SongMetadata]:
    if not data:
        return None
    return SongMetadata(
        singers=list(data.get("singers", [])),
        is_duet=bool(data.get("is_duet", False)),
    )


def _make_request_with_retry(
    url: str,
    headers: dict,
    timeout: int = 10,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> Optional["requests.Response"]:
    """Make an HTTP GET request with retry logic."""
    import requests
    import time

    delay = 1.0
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            return response
        except (requests.exceptions.RequestException, requests.exceptions.Timeout) as e:
            last_error = e
            if attempt < max_retries:
                logger.debug(f"Request retry {attempt + 1}/{max_retries}: {e}")
                time.sleep(delay)
                delay = min(delay * 2, 30.0)

    if last_error:
        logger.warning(f"Request failed after {max_retries} retries: {last_error}")
    return None


def fetch_genius_lyrics_with_singers(title: str, artist: str) -> tuple[Optional[list[tuple[str, str]]], Optional[SongMetadata]]:
    """
    Fetch lyrics from Genius with singer annotations.

    Returns:
        Tuple of (lyrics_with_singers, metadata)
        - lyrics_with_singers: List of (text, singer_name) tuples for each line
        - metadata: SongMetadata with singer info and correct title/artist from Genius
    """
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        logger.warning("requests/beautifulsoup4 not available for Genius scraping")
        return None, None

    # Helper function to create URL slugs
    def make_slug(text: str) -> str:
        slug = re.sub(r'[^\w\s-]', '', text.lower())
        slug = re.sub(r'\s+', '-', slug)
        return slug

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36',
        'Accept': '*/*'
    }

    # Use Genius API to search for the song
    # Format: lowercase, remove punctuation (but convert hyphens to spaces), single spaces
    def clean_for_search(text: str) -> str:
        text = text.lower()
        text = text.replace('-', ' ')  # Convert hyphens to spaces
        text = re.sub(r'[^\w\s]', '', text)  # Remove other punctuation
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize spaces
        return text
    
    # Clean title: remove featured artists and extra info
    def clean_title_for_search(title: str) -> str:
        # Remove everything after pipe character
        title = re.split(r'\s*[|｜]\s*', title)[0]
        # Remove featured artists (ft., feat., featuring)
        title = re.sub(r'\s*[\(\[]?\s*(ft\.?|feat\.?|featuring).*?[\)\]]?\s*$', '', title, flags=re.IGNORECASE)
        title = re.sub(r'\s*[\(\[].*?[\)\]]\s*', '', title)  # Remove any remaining parentheses/brackets
        return title.strip()
    
    cleaned_title = clean_title_for_search(title)
    
    print(f"  Searching Genius: {artist} - {cleaned_title}")
    
    all_urls = []
    genius_title = None
    genius_artist = None
    
    # Try searching with artist + title, then title + artist (in case they're swapped)
    search_queries = [
        f"{clean_for_search(artist)} {clean_for_search(cleaned_title)}",
        f"{clean_for_search(cleaned_title)} {clean_for_search(artist)}"
    ]
    
    for search_query in search_queries:
        api_url = f"https://genius.com/api/search/song?per_page=10&q={search_query.replace(' ', '%20')}"

        response = _make_request_with_retry(api_url, headers)
        if response is None:
            print(f"  Genius search failed after retries")
            continue

        try:
            response.raise_for_status()
            data = response.json()

            if 'response' in data and 'sections' in data['response']:
                for section in data['response']['sections']:
                    if section.get('type') == 'song':
                        for hit in section.get('hits', []):
                            result = hit.get('result', {})
                            url = result.get('url', '')
                            if url and url.endswith('-lyrics') and '/artists/' not in url:
                                all_urls.append(url)
                                # Capture title and artist from first valid result
                                if genius_title is None:
                                    genius_title = result.get('title')
                                    genius_artist = result.get('artist_names')
        except Exception as e:
            print(f"  Genius search failed: {e}")
        
        # If we found results, stop searching
        if all_urls:
            break
    
    # Also try constructing URLs directly
    def make_slug(text: str) -> str:
        slug = re.sub(r'[^\w\s-]', '', text.lower())
        slug = re.sub(r'\s+', '-', slug)
        return slug
    
    artist_slug = make_slug(artist)
    title_slug = make_slug(cleaned_title)
    
    # Add constructed URLs to candidates
    all_urls.extend([
        f"https://genius.com/Genius-romanizations-{artist_slug}-{title_slug}-romanized-lyrics",
        f"https://genius.com/{artist_slug}-{title_slug}-lyrics"
    ])
    
    # Remove duplicates while preserving order
    seen = set()
    unique_urls = []
    for url in all_urls:
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)
    
    # Now prioritize: romanized > original language > skip translations
    song_url = None
    original_url = None
    
    for url in unique_urls:
        # Check if URL exists (with retry)
        response = _make_request_with_retry(url, headers)
        if response is None or response.status_code != 200:
            continue

        # Skip translations
        if 'translation' in url.lower():
            continue
        
        # Verify artist name is in URL (avoid wrong artist matches)
        url_lower = url.lower()
        artist_words = artist.lower().split()
        if not any(word in url_lower for word in artist_words if len(word) > 3):
            continue

        # Prioritize romanized
        if 'romanized' in url.lower():
            song_url = url
            print(f"  Found romanized: {song_url}")
            break

        # Keep track of original language version as fallback
        if not original_url:
            original_url = url
    
    # If no romanized found, use original language version
    if not song_url and original_url:
        song_url = original_url
        print(f"  Found original: {song_url}")
    
    # Print result if found
    if song_url:
        if 'romanized' not in song_url.lower():
            print(f"  Found: {song_url}")
    else:
        print(f"  No Genius results found")
        return None, None

    # Fetch the lyrics page (with retry)
    response = _make_request_with_retry(song_url, headers)
    if response is None or response.status_code != 200:
        print(f"  Error fetching lyrics page")
        return None, None

    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract all text including annotations
    lyrics_containers = soup.find_all('div', {'data-lyrics-container': 'true'})

    if not lyrics_containers:
        return None, None

    # Parse lyrics with singer annotations
    lines_with_singers: list[tuple[str, str]] = []
    current_singer = ""
    singers_found: set[str] = set()

    # Pattern to match section headers like [Verse 1: Bruno Mars] or [Chorus]
    section_pattern = re.compile(r'\[([^\]]+)\]')

    for container in lyrics_containers:
        # Get HTML with line breaks preserved
        for br in container.find_all('br'):
            br.replace_with('\n')

        text = container.get_text()

        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue

            # Check if this is a section header
            section_match = section_pattern.match(line)
            if section_match:
                header = section_match.group(1)
                # Extract singer from header like "Verse 1: Bruno Mars" or "Chorus: Lady Gaga & Bruno Mars"
                if ':' in header:
                    singer_part = header.split(':', 1)[1].strip()
                    current_singer = singer_part
                    singers_found.add(singer_part)
                continue

            # Regular lyrics line
            lines_with_singers.append((line, current_singer))

    if not lines_with_singers:
        return None, None

    # Determine if this is a duet
    # Extract unique singer names (not combined ones)
    unique_singers: list[str] = []
    for singer in singers_found:
        if '&' in singer or ',' in singer:
            # Split combined singers
            parts = re.split(r'[&,]', singer)
            for part in parts:
                name = part.strip()
                if name and name not in unique_singers:
                    unique_singers.append(name)
        elif singer and singer.lower() != 'both':
            if singer not in unique_singers:
                unique_singers.append(singer)

    is_duet = len(unique_singers) >= 2
    
    # Create metadata with title/artist from Genius and singer info
    if is_duet:
        metadata = SongMetadata(
            singers=unique_singers[:2], 
            is_duet=is_duet,
            title=genius_title,
            artist=genius_artist
        )
    elif genius_title or genius_artist:
        # Even if not a duet, include title/artist metadata
        metadata = SongMetadata(
            singers=[],
            is_duet=False,
            title=genius_title,
            artist=genius_artist
        )
    else:
        metadata = None

    print(f"  Found {len(lines_with_singers)} lines with singer annotations")
    if metadata:
        if metadata.is_duet:
            print(f"  Detected duet: {', '.join(metadata.singers)}")
        if metadata.title and metadata.artist:
            print(f"  Genius metadata: {metadata.title} by {metadata.artist}")

    return lines_with_singers, metadata


def merge_lyrics_with_singer_info(
    timed_lines: list[tuple[float, str]],
    genius_lines: list[tuple[str, str]],
    metadata: SongMetadata,
    romanize: bool = True,
) -> list[Line]:
    """
    Merge synced lyrics timing with Genius singer annotations.

    Uses fuzzy matching to align lines from both sources.

    Args:
        timed_lines: List of (timestamp, text) from synced lyrics
        genius_lines: List of (text, singer_name) from Genius
        metadata: SongMetadata with singer info
        romanize: Whether to romanize non-Latin text

    Returns:
        List of Line objects with timing and singer info
    """
    from difflib import SequenceMatcher

    if not timed_lines:
        return []

    # Build a lookup of Genius lines for matching
    # Normalize text for comparison
    genius_normalized = [
        (normalize_text(text), singer)
        for text, singer in genius_lines
    ]

    lines = []
    used_genius_indices: set[int] = set()

    for i, (start_time, text) in enumerate(timed_lines):
        # Romanize if needed
        if romanize:
            text = romanize_line(text)

        # Get end time from next line
        if i + 1 < len(timed_lines):
            end_time = timed_lines[i + 1][0]
            if end_time - start_time > 10.0:
                end_time = start_time + 5.0
        else:
            end_time = start_time + 3.0

        # Try to find matching Genius line for singer info
        text_normalized = normalize_text(text)
        best_match_idx = None
        best_score = 0.0

        for j, (genius_norm, singer) in enumerate(genius_normalized):
            if j in used_genius_indices:
                continue

            score = SequenceMatcher(None, text_normalized, genius_norm).ratio()

            # Prefer sequential matches
            if j not in used_genius_indices:
                score += 0.05

            if score > best_score and score > 0.5:
                best_score = score
                best_match_idx = j

        # Get singer info from match
        singer_id = ""
        if best_match_idx is not None:
            used_genius_indices.add(best_match_idx)
            _, singer_name = genius_lines[best_match_idx]
            singer_id = metadata.get_singer_id(singer_name)

        # Split text into words
        word_texts = text.split()
        if not word_texts:
            continue

        # Distribute timing evenly across words
        line_duration = end_time - start_time
        word_duration = (line_duration * 0.95) / len(word_texts)

        words = []
        for j, word_text in enumerate(word_texts):
            word_start = start_time + j * (line_duration / len(word_texts))
            word_end = word_start + word_duration
            words.append(Word(
                text=word_text,
                start_time=word_start,
                end_time=word_end,
                singer=singer_id,
            ))

        lines.append(Line(
            words=words,
            start_time=start_time,
            end_time=words[-1].end_time if words else end_time,
            singer=singer_id,
        ))

    return lines


def create_lines_from_lrc(
    lrc_text: str,
    romanize: bool = True,
    title: str = "",
    artist: str = "",
) -> list[Line]:
    """
    Create Line objects from LRC format with evenly distributed word timing.

    Uses the LRC timestamps for line timing and distributes words evenly
    within each line's duration.

    Args:
        lrc_text: LRC format lyrics text
        romanize: If True, romanize non-Latin scripts (e.g., Korean to romanized)
        title: Song title for metadata filtering
        artist: Artist name for metadata filtering
    """
    timed_lines = parse_lrc_with_timing(lrc_text, title, artist)

    if not timed_lines:
        return []

    lines = []
    for i, (start_time, text) in enumerate(timed_lines):
        # Romanize Korean text if requested
        if romanize:
            text = romanize_line(text)

        # Get end time from next line or add default duration
        if i + 1 < len(timed_lines):
            end_time = timed_lines[i + 1][0]
            # Cap the line duration at a reasonable max (e.g., 10 seconds)
            if end_time - start_time > 10.0:
                end_time = start_time + 5.0
        else:
            end_time = start_time + 3.0

        # Split text into words
        word_texts = text.split()
        if not word_texts:
            continue

        # Distribute timing evenly across words
        line_duration = end_time - start_time
        # Leave small gaps between words
        word_duration = (line_duration * 0.95) / len(word_texts)

        words = []
        for j, word_text in enumerate(word_texts):
            word_start = start_time + j * (line_duration / len(word_texts))
            word_end = word_start + word_duration
            words.append(Word(
                text=word_text,
                start_time=word_start,
                end_time=word_end,
            ))

        lines.append(Line(
            words=words,
            start_time=start_time,
            end_time=words[-1].end_time if words else end_time,
        ))

    return lines


def normalize_text(text: str) -> str:
    """Normalize text for comparison (lowercase, remove punctuation)."""
    # Lowercase and remove punctuation except apostrophes in contractions
    text = text.lower()
    # Keep apostrophes in words like "don't", "I'm"
    text = re.sub(r"[^\w\s']", "", text)
    # Normalize whitespace
    text = " ".join(text.split())
    return text


def _search_syncedlyrics_with_retry(
    search_term: str,
    synced_only: bool = True,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> Optional[str]:
    """Search syncedlyrics with retry logic for network failures."""
    import time

    delay = 1.0
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            return syncedlyrics.search(search_term, synced_only=synced_only)
        except Exception as e:
            last_error = e
            # Only retry on network-related errors
            error_str = str(e).lower()
            if any(word in error_str for word in ['timeout', 'connection', 'network', 'socket']):
                if attempt < max_retries:
                    logger.debug(f"Syncedlyrics retry {attempt + 1}/{max_retries}: {e}")
                    time.sleep(delay)
                    delay = min(delay * 2, 30.0)
                    continue
            # For non-network errors, don't retry
            break

    if last_error:
        logger.debug(f"Syncedlyrics search failed: {last_error}")
    return None


def fetch_lyrics_multi_source(title: str, artist: str) -> tuple[Optional[str], bool, str]:
    """
    Fetch lyrics from multiple sources and search variations.

    Returns:
        Tuple of (lrc_text, is_synced, source_description)
        - lrc_text: Raw LRC format text (or plain text if not synced)
        - is_synced: True if lyrics have timing info
        - source_description: Description of the source used
    """
    from .downloader import clean_title

    # Clean the title first
    clean = clean_title(title, artist)

    # Try multiple search term variations
    search_variations = [
        f"{artist} {clean}",           # Artist + clean title
        f"{clean} {artist}",           # Clean title + artist
        clean,                          # Just clean title
        f"{artist} {title}",           # Artist + original title
        title,                          # Just original title
    ]

    # Remove duplicates while preserving order
    seen = set()
    unique_searches = []
    for s in search_variations:
        normalized = s.lower().strip()
        if normalized not in seen:
            seen.add(normalized)
            unique_searches.append(s)

    # Try synced lyrics first (has timing info)
    for search_term in unique_searches:
        print(f"  Trying: {search_term}")
        lrc = _search_syncedlyrics_with_retry(search_term, synced_only=True)
        if lrc:
            lines = extract_lyrics_text(lrc, title, artist)
            if lines and len(lines) >= 5:  # Need meaningful content
                print(f"  Found synced lyrics ({len(lines)} lines)")
                return lrc, True, f"synced: {search_term}"

    # Try plain lyrics as fallback (no timing but better for reference)
    for search_term in unique_searches[:3]:  # Only try first few
        lrc = _search_syncedlyrics_with_retry(search_term, synced_only=False)
        if lrc:
            lines = extract_lyrics_text(lrc, title, artist)
            if lines and len(lines) >= 5:
                print(f"  Found plain lyrics ({len(lines)} lines)")
                return lrc, False, f"plain: {search_term}"

    return None, False, "none"


def fetch_synced_lyrics(title: str, artist: str) -> Optional[str]:
    """Fetch synced lyrics using syncedlyrics library."""
    search_term = f"{artist} {title}"
    print(f"Searching for lyrics: {search_term}")

    lrc = _search_syncedlyrics_with_retry(search_term, synced_only=True)
    if lrc:
        logger.info("Found lyrics online!")
        return lrc

    return None


def match_line_to_lyrics(
    transcribed_line: str,
    lyrics_lines: list[str],
    used_indices: set,
    last_idx: Optional[int] = None,
) -> tuple[Optional[str], Optional[int]]:
    """Find the best matching lyrics line for a transcribed line.

    Adds a bias towards keeping the matched lyrics index moving
    forward through the song so that lines are not applied badly
    out of order.

    Returns:
        Tuple of (matched_lyrics_line, lyrics_index) or (None, None) if no good match
    """
    from difflib import SequenceMatcher

    transcribed_norm = normalize_text(transcribed_line)
    if not transcribed_norm:
        return None, None

    best_match = None
    best_score = 0.0
    best_idx = None

    for i, lyrics_line in enumerate(lyrics_lines):
        # Skip already used lines (we don't want to reuse the exact
        # same Genius line for different segments).
        if i in used_indices:
            continue

        lyrics_norm = normalize_text(lyrics_line)
        if not lyrics_norm:
            continue

        # Base similarity
        score = SequenceMatcher(None, transcribed_norm, lyrics_norm).ratio()

        # Encourage monotonic progression through the lyrics: prefer
        # indices at or after the last matched index, slightly penalize
        # going backwards unless the match is extremely strong.
        if last_idx is not None:
            if i >= last_idx:
                score += 0.05
            else:
                score -= 0.15

        if score > best_score:
            best_score = score
            best_match = lyrics_line
            best_idx = i

    # Require decent similarity (0.5 = 50% match)
    if best_score >= 0.5:
        return best_match, best_idx

    return None, None


def _hybrid_alignment(whisper_lines: list["Line"], lyrics_text: list[str], synced_timings: list[tuple[float, str]], norm_token_func) -> list["Line"]:
    """
    Hybrid alignment: use Genius text (absolute) with word-level timing from synced lyrics.
    
    CRITICAL: lyrics_text (Genius) is the absolute source of truth for content and order.
    We match Genius words to synced words for timing, enforcing monotonic time progression.
    """
    from difflib import SequenceMatcher
    
    logger.info(f"_hybrid_alignment called with {len(lyrics_text)} Genius lines, {len(synced_timings)} synced timings")
    
    # Flatten Genius text into words with line boundaries
    genius_words = []
    for line_idx, line_text in enumerate(lyrics_text):
        line_text = line_text.strip()
        if not line_text:
            continue
        for word in line_text.split():
            genius_words.append({'text': word, 'line_idx': line_idx})
    
    # Flatten synced lyrics into words with timing (interpolated within each line)
    synced_words = []
    for sync_time, sync_text in synced_timings:
        words = sync_text.split()
        if not words:
            continue
        # Find next line's start time for duration
        next_time = None
        for t, _ in synced_timings:
            if t > sync_time:
                next_time = t
                break
        if next_time is None:
            next_time = sync_time + 3.0
        
        # Interpolate word timing within this line
        duration = next_time - sync_time
        word_duration = duration / len(words)
        for word_idx, word in enumerate(words):
            word_start = sync_time + word_idx * word_duration
            word_end = word_start + word_duration
            synced_words.append({
                'text': word,
                'norm': norm_token_func(word),
                'start': word_start,
                'end': word_end
            })
    
    logger.info(f"Matching {len(genius_words)} Genius words to {len(synced_words)} synced words")
    
    # Match Genius words to synced words using sequence alignment
    genius_norms = [norm_token_func(w['text']) for w in genius_words]
    synced_norms = [w['norm'] for w in synced_words]
    
    matcher = SequenceMatcher(None, genius_norms, synced_norms)
    matches = matcher.get_opcodes()
    
    # Assign timing to Genius words based on matches
    for tag, g_start, g_end, s_start, s_end in matches:
        if tag == 'equal' or tag == 'replace':
            for i in range(g_end - g_start):
                g_idx = g_start + i
                s_idx = s_start + min(i, s_end - s_start - 1)
                if s_idx < len(synced_words):
                    genius_words[g_idx]['start'] = synced_words[s_idx]['start']
                    genius_words[g_idx]['end'] = synced_words[s_idx]['end']
        elif tag == 'delete':
            for i in range(g_end - g_start):
                g_idx = g_start + i
                base_time = genius_words[g_idx - 1]['end'] if g_idx > 0 and genius_words[g_idx - 1].get('end') else 0.0
                genius_words[g_idx]['start'] = base_time
                genius_words[g_idx]['end'] = base_time + 0.3
    
    # Post-process: enforce monotonic timing at word level
    for i in range(1, len(genius_words)):
        if genius_words[i].get('start', 0) < genius_words[i-1].get('end', 0):
            # Current word starts before previous ends - fix it
            genius_words[i]['start'] = genius_words[i-1]['end']
            genius_words[i]['end'] = genius_words[i]['start'] + 0.3
    
    # Check for large intra-line gaps and fix them
    for line_idx in set(w['line_idx'] for w in genius_words):
        line_words = [w for w in genius_words if w['line_idx'] == line_idx]
        if len(line_words) < 2:
            continue
        
        # Check for gaps larger than 5 seconds between words in same line
        for i in range(1, len(line_words)):
            gap = line_words[i].get('start', 0) - line_words[i-1].get('end', 0)
            if gap > 5.0:
                # Large gap detected - use timing from majority of words (last word's timing)
                logger.info(f"Large gap ({gap:.1f}s) detected in line {line_idx}, using later timing")
                
                # Use the last word's timing as reference and work backwards
                last_word_start = line_words[-1].get('start', 0)
                # Estimate line duration based on word count (0.4s per word)
                estimated_duration = len(line_words) * 0.4
                line_start = max(0, last_word_start - estimated_duration + 0.4)
                line_end = line_words[-1].get('end', line_start + estimated_duration)
                
                duration = line_end - line_start
                word_duration = duration / len(line_words)
                
                # Redistribute evenly
                for j, word in enumerate(line_words):
                    word['start'] = line_start + j * word_duration
                    word['end'] = word['start'] + word_duration
                break
    
    # Group words back into lines
    lines_dict = {}
    for word_data in genius_words:
        line_idx = word_data['line_idx']
        if line_idx not in lines_dict:
            lines_dict[line_idx] = []
        lines_dict[line_idx].append(word_data)
    
    # Build result lines
    result_lines = []
    for line_idx in sorted(lines_dict.keys()):
        words_data = lines_dict[line_idx]
        words = [
            Word(
                text=w['text'],
                start_time=w.get('start', 0.0),
                end_time=w.get('end', 0.3)
            )
            for w in words_data
        ]
        if words:
            result_lines.append(Line(
                words=words,
                start_time=words[0].start_time,
                end_time=words[-1].end_time
            ))
    
    # Check for lines that are too short and extend them
    for i, line in enumerate(result_lines):
        duration = line.end_time - line.start_time
        line_text = ' '.join(w.text for w in line.words)
        
        # If line is very short (< 1.5 seconds) and has repetitive text, extend it
        if duration < 1.5 and ('la-la' in line_text.lower() or 'la-la-la-la' in line_text.lower()):
            logger.info(f"Extending short line {i}: '{line_text}' from {duration:.2f}s to 2.5s")
            target_duration = 2.5
            old_end = line.end_time
            word_duration = target_duration / len(line.words)
            
            # Extend this line's words
            for j, word in enumerate(line.words):
                word.start_time = line.start_time + j * word_duration
                word.end_time = word.start_time + word_duration
            
            line.end_time = line.words[-1].end_time
            new_end = line.end_time
            extension = new_end - old_end
            
            # Only shift following lines that would overlap (within 1 second)
            for j in range(i + 1, len(result_lines)):
                if result_lines[j].start_time < new_end + 0.1:
                    # This line would overlap, shift it
                    result_lines[j].start_time += extension
                    result_lines[j].end_time += extension
                    for word in result_lines[j].words:
                        word.start_time += extension
                        word.end_time += extension
                else:
                    # Lines are far enough apart, stop shifting
                    break
    
    # Final pass: ensure lines are in temporal order
    logger.info(f"Checking temporal order for {len(result_lines)} lines")
    
    fixes_applied = 0
    for i in range(1, len(result_lines)):
        prev_start = result_lines[i-1].start_time
        curr_start = result_lines[i].start_time
        if curr_start < prev_start:
            # This line starts before previous line - shift it forward
            logger.info(f"Fixing line {i}: {curr_start:.2f}s < {prev_start:.2f}s")
            shift = prev_start - curr_start + 0.1
            duration = result_lines[i].end_time - result_lines[i].start_time
            result_lines[i].start_time += shift
            result_lines[i].end_time = result_lines[i].start_time + duration
            # Shift all words in this line
            for word in result_lines[i].words:
                word.start_time += shift
                word.end_time += shift
            fixes_applied += 1
            logger.info(f"  Shifted to {result_lines[i].start_time:.2f}s")
    
    if fixes_applied > 0:
        logger.info(f"Applied {fixes_applied} temporal ordering fixes")
    
    logger.info(f"_hybrid_alignment returning {len(result_lines)} lines")
    return result_lines


def _align_genius_to_whisperx_simple(whisper_lines: list["Line"], genius_text: list[str], norm_token_func) -> list["Line"]:
    """Simple alignment: keep ALL Genius words, use WhisperX for timing hints."""
    from difflib import SequenceMatcher
    
    print(f"DEBUG: Simple alignment with {len(genius_text)} Genius lines")
    
    # Flatten WhisperX words
    whisper_words = []
    for line in whisper_lines:
        for word in line.words:
            whisper_words.append({
                "text": word.text,
                "norm": norm_token_func(word.text),
                "start": word.start_time,
                "end": word.end_time
            })
    
    print(f"DEBUG: {len(whisper_words)} WhisperX words available")
    
    result_lines = []
    whisper_idx = 0
    
    for line_idx, line_text in enumerate(genius_text):
        if not line_text.strip():
            continue
        
        if line_idx == 12:  # Debug the problematic line
            print(f"DEBUG: Processing line 12: {repr(line_text)}")
            
        # Split line into words, preserving punctuation
        import string
        line_words = line_text.split()
        
        if line_idx == 12:
            print(f"DEBUG: Split into {len(line_words)} words: {line_words}")
        
        if not line_words:
            continue
        
        result_words = []
        line_start = None
        line_end = None
        
        for word_text in line_words:
            # Clean word for matching
            word_clean = word_text.strip(string.punctuation)
            word_norm = norm_token_func(word_clean)
            
            # Try to find matching WhisperX word
            best_match = None
            best_score = 0.0
            best_idx = whisper_idx
            
            # Look ahead in WhisperX words (within reason)
            for offset in range(min(10, len(whisper_words) - whisper_idx)):
                idx = whisper_idx + offset
                if idx >= len(whisper_words):
                    break
                    
                w = whisper_words[idx]
                score = SequenceMatcher(None, word_norm, w["norm"]).ratio()
                
                # Prefer earlier matches
                score -= offset * 0.05
                
                if score > best_score and score > 0.6:
                    best_score = score
                    best_match = w
                    best_idx = idx
            
            if best_match:
                # Use WhisperX timing
                result_words.append(Word(
                    text=word_text,
                    start_time=best_match["start"],
                    end_time=best_match["end"]
                ))
                whisper_idx = best_idx + 1
                
                if line_start is None:
                    line_start = best_match["start"]
                line_end = best_match["end"]
            else:
                # No match - interpolate timing
                if result_words:
                    # Place after last word
                    prev_end = result_words[-1].end_time
                    word_start = prev_end
                    word_end = prev_end + 0.3
                else:
                    # First word with no match - use current WhisperX position
                    if whisper_idx < len(whisper_words):
                        word_start = whisper_words[whisper_idx]["start"]
                        word_end = word_start + 0.3
                    else:
                        word_start = line_end + 0.1 if line_end else 0.0
                        word_end = word_start + 0.3
                
                result_words.append(Word(
                    text=word_text,
                    start_time=word_start,
                    end_time=word_end
                ))
                
                if line_start is None:
                    line_start = word_start
                line_end = word_end
        
        if result_words:
            result_lines.append(Line(
                words=result_words,
                start_time=result_words[0].start_time,
                end_time=result_words[-1].end_time
            ))
    
    return result_lines


def correct_transcription_with_lyrics(lines: list["Line"], lyrics_text: list[str], synced_line_timings: Optional[list[tuple[float, str]]] = None) -> list["Line"]:
    """Align Whisper transcription to reference lyrics at the word level.

    This treats the reference lyrics (from Genius/LRC) as the source of
    truth for *text* and WhisperX as the source of *timing*.
    
    If synced_line_timings are provided, use them to anchor line start times
    and only use WhisperX for word-level detail within each line.

    The alignment is done globally over the entire song by flattening
    both sequences into words and running a DP alignment between the
    Whisper words (with timestamps) and the Genius words (without
    timestamps). We then reconstruct per-line timing for Genius lyrics
    based on the aligned word timings.
    """
    if not lyrics_text or not lines:
        return lines

    import re
    from difflib import SequenceMatcher

    def norm_token(t: str) -> str:
        t = t.lower()
        return re.sub(r"[^\w']", "", t)

    # If no synced timings, use simple approach: keep ALL Genius words,
    # match to WhisperX for timing, interpolate for unmatched words
    if synced_line_timings is None:
        return _align_genius_to_whisperx_simple(lines, lyrics_text, norm_token)

    # Use hybrid alignment when synced timings are available
    # This preserves Genius text order while using synced line timing
    if synced_line_timings:
        return _hybrid_alignment(lines, lyrics_text, synced_line_timings, norm_token)

    # Flatten Genius lyrics into tokens with line indices
    ref_lines = [ln.strip() for ln in lyrics_text if ln.strip()]
    if not ref_lines:
        return lines

    genius_tokens: list[dict] = []  # {"raw", "norm", "line_idx"}
    for line_idx, text in enumerate(ref_lines):
        for raw in text.split():
            genius_tokens.append({
                "raw": raw,
                "norm": norm_token(raw),
                "line_idx": line_idx,
            })

    G = len(genius_tokens)
    if G == 0:
        return lines

    # Flatten Whisper output into tokens with timings
    whisper_tokens: list[dict] = []  # {"raw", "norm", "start", "end"}
    for line in lines:
        for w in line.words:
            whisper_tokens.append({
                "raw": w.text,
                "norm": norm_token(w.text),
                "start": w.start_time,
                "end": w.end_time,
            })

    W = len(whisper_tokens)
    if W == 0:
        return lines

    # Global DP alignment between whisper_tokens (i) and genius_tokens (j)
    NEG_INF = -1e9
    dp: list[list[float]] = [[NEG_INF] * (G + 1) for _ in range(W + 1)]
    back: list[list[Optional[tuple[str, int, int, Optional[float]]]]] = [
        [None] * (G + 1) for _ in range(W + 1)
    ]
    dp[0][0] = 0.0

    skip_whisper_penalty = 0.4
    skip_genius_penalty = 0.4

    for i in range(W + 1):
        for j in range(G + 1):
            cur = dp[i][j]
            if cur <= NEG_INF / 2:
                continue

            # Match whisper_tokens[i] with genius_tokens[j]
            if i < W and j < G:
                tn = whisper_tokens[i]["norm"]
                rn = genius_tokens[j]["norm"]
                if tn == rn and tn != "":
                    s = 1.0
                elif tn and rn:
                    s = SequenceMatcher(None, tn, rn).ratio()
                else:
                    s = 0.0

                gain = max(s - 0.3, 0.0)
                new = cur + gain
                if new > dp[i + 1][j + 1]:
                    dp[i + 1][j + 1] = new
                    back[i + 1][j + 1] = ("M", i, j, s)

            # Skip a Whisper word
            if i < W:
                new = cur - skip_whisper_penalty
                if new > dp[i + 1][j]:
                    dp[i + 1][j] = new
                    back[i + 1][j] = ("SW", i, j, None)

            # Skip a Genius word
            if j < G:
                new = cur - skip_genius_penalty
                if new > dp[i][j + 1]:
                    dp[i][j + 1] = new
                    back[i][j + 1] = ("SG", i, j, None)

    # Backtrack to recover word-level matches
    i, j = W, G
    matches: list[tuple[int, int, float]] = []  # (whisper_idx, genius_idx, sim)

    while i > 0 or j > 0:
        info = back[i][j]
        if info is None:
            break
        op, pi, pj, sim = info
        if op == "M" and sim is not None:
            matches.append((pi, pj, sim))
        i, j = pi, pj

    matches.reverse()

    # Aggregate timings for each Genius word
    SIM_THRESHOLD = 0.5
    g_times: list[Optional[tuple[float, float]]] = [None] * G

    for w_idx, g_idx, sim in matches:
        if sim < SIM_THRESHOLD:
            continue
        wt = whisper_tokens[w_idx]
        cur = g_times[g_idx]
        if cur is None:
            g_times[g_idx] = (wt["start"], wt["end"])
        else:
            s0, e0 = cur
            g_times[g_idx] = (min(s0, wt["start"]), max(e0, wt["end"]))

    # If we failed to align anything, fall back to original lines
    if all(t is None for t in g_times):
        return lines

    # Interpolate timings for unmatched Genius words using anchors
    track_start = min(w["start"] for w in whisper_tokens)
    track_end = max(w["end"] for w in whisper_tokens)

    anchors = [idx for idx, t in enumerate(g_times) if t is not None]

    for gi in range(G):
        if g_times[gi] is not None:
            continue

        prev_a = max([a for a in anchors if a < gi], default=None) if anchors else None
        next_a = min([a for a in anchors if a > gi], default=None) if anchors else None

        if prev_a is None and next_a is None:
            # Spread uniformly across the track
            span = max(track_end - track_start, 0.1)
            frac = gi / max(G, 1)
            s = track_start + frac * span
            e = s + span / max(G * 2, 1)
        elif prev_a is None:
            # Before first anchor
            next_s, _ = g_times[next_a]  # type: ignore
            span = max(next_s - track_start, 0.1)
            frac = gi / max(next_a, 1)
            s = track_start + frac * span
            e = s + span / max(next_a + 1, 2)
        elif next_a is None:
            # After last anchor
            _, prev_e = g_times[prev_a]  # type: ignore
            span = max(track_end - prev_e, 0.1)
            frac = (gi - prev_a) / max(G - prev_a, 1)
            s = prev_e + frac * span
            e = s + span / max(G - prev_a + 1, 2)
        else:
            # Between two anchors
            _, prev_e = g_times[prev_a]  # type: ignore
            next_s, _ = g_times[next_a]  # type: ignore
            span = max(next_s - prev_e, 0.1)
            frac = (gi - prev_a) / max(next_a - prev_a, 1)
            s = prev_e + frac * span
            e = s + span / max(next_a - prev_a + 1, 2)

        if e <= s:
            e = s + 0.05
        g_times[gi] = (s, e)

    # Reconstruct Line objects grouped by Genius line indices
    aligned_lines: list[Line] = []
    for line_idx, _ in enumerate(ref_lines):
        token_indices = [
            k for k, tok in enumerate(genius_tokens)
            if tok["line_idx"] == line_idx
        ]
        if not token_indices:
            continue

        words: list[Word] = []
        for k in token_indices:
            tok = genius_tokens[k]
            s, e = g_times[k] or (track_start, track_end)
            if e <= s:
                e = s + 0.05
            words.append(Word(
                text=tok["raw"],
                start_time=s,
                end_time=e,
            ))

        words.sort(key=lambda w: w.start_time)
        aligned_lines.append(Line(
            words=words,
            start_time=words[0].start_time,
            end_time=words[-1].end_time,
        ))

    # Optionally split very long lines for display
    aligned_lines = split_long_lines(aligned_lines)

    return aligned_lines


def transcribe_and_align(vocals_path: str, lyrics_text: Optional[list[str]] = None) -> list[Line]:
    """
    Transcribe audio with word-level alignment.

    Uses whisperx for accurate word-level timestamps via forced alignment.

    Args:
        vocals_path: Path to vocals audio file
        lyrics_text: Optional list of lyrics lines - used to detect language

    Returns:
        List of Line objects with word timing
    """
    
    if lyrics_text:
        print(f"DEBUG transcribe_and_align: received {len(lyrics_text)} lyrics lines")
        if len(lyrics_text) > 12:
            print(f"DEBUG transcribe_and_align: lyrics_text[12] = {repr(lyrics_text[12])}")
    
    import whisperx
    import torch
    import warnings

    # Suppress PyTorch Lightning upgrade warnings
    warnings.filterwarnings("ignore", message="Lightning automatically upgraded")
    
    # Fix for PyTorch 2.6+ weights_only=True compatibility with pyannote
    _original_torch_load = torch.load
    def _patched_torch_load(*args, **kwargs):
        # Force weights_only=False to handle pyannote/omegaconf models
        kwargs['weights_only'] = False
        return _original_torch_load(*args, **kwargs)
    torch.load = _patched_torch_load

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    
    # Set number of threads for CPU inference
    import os
    if device == "cpu":
        # Use all available CPU cores
        num_threads = os.cpu_count() or 4
        torch.set_num_threads(num_threads)
        print(f"Using {num_threads} CPU threads for WhisperX")

    # Detect language from lyrics if available
    language = None
    if lyrics_text:
        sample = " ".join(lyrics_text[:10]).lower()
        
        # Count English words (common words that appear in English)
        english_words = ["the ", "you ", "and ", "are ", "is ", "it ", "to ", "of ", "in ", "that ", "have ", "i ", "for ", "not ", "on ", "with ", "he ", "as ", "do ", "at "]
        english_count = sum(1 for word in english_words if word in sample)
        
        # Check for Japanese romanization patterns (more specific patterns)
        japanese_patterns = [" wa ", " ga ", " wo ", " ni ", " de ", " to ", " no ", " ka ", " ne ", " yo ",
                           "desu", "masu", "tte", "kara", "made", "nai", "tai", "chan", "kun", "san"]
        # Check for Spanish (require multiple matches to avoid false positives)  
        spanish_words = [" el ", " la ", " los ", " las ", " que ", " con ", " por ", " para ", " esta ", " como ",
                        " es ", " en ", " un ", " una ", " del ", " al ", " se ", " te ", " me ", " le "]
        
        japanese_count = sum(1 for pattern in japanese_patterns if pattern in sample)
        spanish_count = sum(1 for word in spanish_words if word in sample)
        
        # Prioritize English if it has strong presence (for mixed-language songs)
        if english_count >= 5:
            language = "en"
            print(f"Detected English from lyrics, using language: {language}")
        elif japanese_count >= 3:
            language = "ja"
            print(f"Detected Japanese from lyrics, using language: {language}")
        elif spanish_count >= 3:
            language = "es"
            print(f"Detected Spanish from lyrics, using language: {language}")
    
    print(f"Loading whisperx model (device: {device}, compute_type: {compute_type})...")
    model = whisperx.load_model(
        "base",  # Use base model instead of medium for 5x speed improvement
        device, 
        compute_type=compute_type, 
        language=None,  # Auto-detect to avoid model download issues
        download_root=None,
        threads=num_threads if device == "cpu" else 0
    )

    logger.info("Transcribing audio (this may take several minutes)...")
    audio = whisperx.load_audio(vocals_path)
    result = model.transcribe(
        audio, 
        batch_size=32,  # Increase batch size for faster processing
        language=language
    )
    
    # Get language (either forced or detected)
    detected_lang = language or result.get("language", "en")
    print(f"Using language: {detected_lang}")

    # Load alignment model for word-level timestamps
    logger.info("Aligning words to audio...")
    model_a, metadata = whisperx.load_align_model(language_code=detected_lang, device=device)
    result = whisperx.align(
        result["segments"], 
        model_a, 
        metadata, 
        audio, 
        device,
        return_char_alignments=False  # Skip character alignments for speed
    )

    # Convert to our Line/Word format
    # Handle words missing timestamps by interpolating from segment timing
    lines = []
    for segment in result.get("segments", []):
        segment_start = segment.get("start", 0.0)
        segment_end = segment.get("end", segment_start + 1.0)
        segment_words = segment.get("words", [])

        if not segment_words:
            continue

        # First pass: collect words and identify which have timestamps
        word_infos = []
        for word_data in segment_words:
            word_text = word_data.get("word", "").strip()
            if not word_text:
                continue
            has_timing = "start" in word_data and "end" in word_data
            word_infos.append({
                "text": word_text,
                "start": word_data.get("start"),
                "end": word_data.get("end"),
                "has_timing": has_timing,
            })

        if not word_infos:
            continue

        # Second pass: interpolate missing timestamps
        # Use segment timing and distribute evenly for words without timestamps
        words = []

        # Find runs of words without timing and interpolate
        i = 0
        while i < len(word_infos):
            wi = word_infos[i]

            if wi["has_timing"]:
                words.append(Word(
                    text=wi["text"],
                    start_time=wi["start"],
                    end_time=wi["end"],
                ))
                i += 1
            else:
                # Find the run of words without timing
                run_start_idx = i
                while i < len(word_infos) and not word_infos[i]["has_timing"]:
                    i += 1
                run_end_idx = i

                # Determine time bounds for this run
                if run_start_idx == 0:
                    # Run starts at beginning - use segment start
                    time_start = segment_start
                else:
                    # Use end time of previous word
                    time_start = words[-1].end_time + 0.05

                if run_end_idx >= len(word_infos):
                    # Run goes to end - use segment end
                    time_end = segment_end
                else:
                    # Use start time of next timed word
                    time_end = word_infos[run_end_idx]["start"] - 0.05

                # Distribute time evenly among words in run
                run_count = run_end_idx - run_start_idx
                if time_end > time_start and run_count > 0:
                    duration_per_word = (time_end - time_start) / run_count
                    for j in range(run_count):
                        word_start = time_start + j * duration_per_word
                        word_end = word_start + duration_per_word - 0.02
                        words.append(Word(
                            text=word_infos[run_start_idx + j]["text"],
                            start_time=word_start,
                            end_time=max(word_end, word_start + 0.1),
                        ))

        if words:
            lines.append(Line(
                words=words,
                start_time=words[0].start_time,
                end_time=words[-1].end_time,
            ))

    # Fix bad word timing (first words, words after long gaps)
    lines = fix_word_timing(lines)

    # Split lines that are too wide for the screen
    lines = split_long_lines(lines)

    return lines


def fix_word_timing(lines: list[Line], max_word_duration: float = 2.0, max_gap: float = 2.0) -> list[Line]:
    """
    Fix unrealistic word timing from whisperx alignment.

    Strategy: Keep original timestamps for well-aligned words, but fix
    words with bad timing by deriving from neighboring words.
    """
    # First pass: collect durations of "good" words to calculate average
    good_durations = []
    for line in lines:
        for word in line.words:
            duration = word.end_time - word.start_time
            # Consider a word "good" if duration is reasonable
            if 0.05 < duration < max_word_duration:
                good_durations.append(duration)

    # Calculate average duration from good words (fallback to 0.4s)
    if good_durations:
        avg_duration = sum(good_durations) / len(good_durations)
    else:
        avg_duration = 0.4

    # Second pass: fix problematic words
    fixed_lines = []

    for line in lines:
        if not line.words or len(line.words) < 2:
            fixed_lines.append(line)
            continue

        fixed_words = []

        for i, word in enumerate(line.words):
            duration = word.end_time - word.start_time
            is_first = (i == 0)
            is_last = (i == len(line.words) - 1)

            # Check if this word needs fixing
            needs_fix = False
            fix_from_next = False
            fix_from_prev = False

            # First word with bad duration or big gap to next word
            if is_first and not is_last:
                next_word = line.words[i + 1]
                gap_to_next = next_word.start_time - word.end_time
                if duration > max_word_duration or gap_to_next > max_gap:
                    needs_fix = True
                    fix_from_next = True

            # Last word with big gap from previous word - derive from PREVIOUS
            if is_last and not is_first:
                prev_word = fixed_words[-1] if fixed_words else line.words[i - 1]
                gap_from_prev = word.start_time - prev_word.end_time
                if gap_from_prev > max_gap:
                    needs_fix = True
                    fix_from_prev = True

            # Middle word after a long gap - derive from next word
            if not is_first and not is_last:
                prev_word = fixed_words[-1] if fixed_words else line.words[i - 1]
                gap_from_prev = word.start_time - prev_word.end_time
                if gap_from_prev > max_gap:
                    needs_fix = True
                    fix_from_next = True

            if needs_fix:
                if fix_from_next:
                    # Derive timing from the NEXT word
                    next_word = line.words[i + 1]
                    new_end = next_word.start_time - 0.05
                    new_start = new_end - avg_duration
                    fixed_words.append(Word(
                        text=word.text,
                        start_time=max(new_start, 0),
                        end_time=new_end,
                    ))
                elif fix_from_prev:
                    # Derive timing from the PREVIOUS word
                    prev_word = fixed_words[-1]
                    new_start = prev_word.end_time + 0.05
                    new_end = new_start + avg_duration
                    fixed_words.append(Word(
                        text=word.text,
                        start_time=new_start,
                        end_time=new_end,
                    ))
                else:
                    fixed_words.append(word)
            else:
                # Keep original timing
                fixed_words.append(word)

        if fixed_words:
            fixed_lines.append(Line(
                words=fixed_words,
                start_time=fixed_words[0].start_time,
                end_time=fixed_words[-1].end_time,
            ))

    return fixed_lines


def split_long_lines(lines: list[Line], max_width_ratio: float = 0.75) -> list[Line]:
    """
    Split lines that are too wide to fit on screen.

    Args:
        lines: List of Line objects
        max_width_ratio: Maximum width as ratio of screen width (0.75 = 75%)

    Returns:
        List of Line objects with long lines split
    """
    from ..utils.fonts import get_font
    from ..config import VIDEO_WIDTH

    # Use the shared font loader to ensure consistency with renderer
    font = get_font()
    max_width = VIDEO_WIDTH * max_width_ratio
    
    split_lines = []

    for line in lines:
        # Measure line width exactly as renderer does - sum of individual word widths with spaces
        total_width = 0
        for word in line.words:
            bbox = font.getbbox(word.text + " ")
            word_width = bbox[2] - bbox[0]
            total_width += word_width

        if total_width <= max_width:
            # Line fits, keep as-is
            split_lines.append(line)
            continue

        # Need to split - find a good break point
        words = line.words
        total_words = len(words)

        if total_words < 2:
            split_lines.append(line)
            continue

        # Find split point by measuring cumulative width
        best_split = total_words // 2
        cumulative_width = 0

        for i, word in enumerate(words):
            word_text = word.text + " "
            word_bbox = font.getbbox(word_text)
            word_width = word_bbox[2] - word_bbox[0]
            cumulative_width += word_width
            
            # Split when we exceed max_width for the first line
            if cumulative_width > max_width:
                best_split = max(1, i)  # Split before this word
                break
        else:
            # If we never exceeded, split at midpoint
            best_split = total_words // 2

        # Ensure we don't create tiny splits
        best_split = max(1, min(best_split, total_words - 1))

        # Create two lines from the split
        first_words = words[:best_split]
        second_words = words[best_split:]

        if first_words:
            first_line = Line(
                words=first_words,
                start_time=first_words[0].start_time,
                end_time=first_words[-1].end_time,
            )
            # Recursively split if still too long
            split_lines.extend(split_long_lines([first_line], max_width_ratio))

        if second_words:
            second_line = Line(
                words=second_words,
                start_time=second_words[0].start_time,
                end_time=second_words[-1].end_time,
            )
            # Recursively split if still too long
            split_lines.extend(split_long_lines([second_line], max_width_ratio))

    return split_lines


def _split_by_char_count(lines: list[Line], max_chars: int = 50) -> list[Line]:
    """Fallback line splitting using character count."""
    split_lines = []
    for line in lines:
        line_text = " ".join(w.text for w in line.words)
        if len(line_text) <= max_chars:
            split_lines.append(line)
            continue
        
        words = line.words
        if len(words) < 2:
            split_lines.append(line)
            continue
        
        # Split at midpoint
        mid = len(words) // 2
        first_words = words[:mid]
        second_words = words[mid:]
        
        if first_words:
            split_lines.append(Line(
                words=first_words,
                start_time=first_words[0].start_time,
                end_time=first_words[-1].end_time,
            ))
        if second_words:
            split_lines.append(Line(
                words=second_words,
                start_time=second_words[0].start_time,
                end_time=second_words[-1].end_time,
            ))
    
    return split_lines


def filter_lines_by_lyrics(transcribed_lines: list[Line], lyrics_text: list[str]) -> list[Line]:
    """
    Filter transcribed lines to only include those that match the provided lyrics.

    This helps remove hallucinated or extra content that Whisper might add.
    """
    import difflib

    # Create a set of normalized lyrics words for matching
    lyrics_words = set()
    for line in lyrics_text:
        for word in line.lower().split():
            # Remove punctuation
            clean = re.sub(r'[^\w]', '', word)
            if clean:
                lyrics_words.add(clean)

    # Filter lines - keep only if most words appear in lyrics
    filtered_lines = []
    for line in transcribed_lines:
        line_words = [re.sub(r'[^\w]', '', w.text.lower()) for w in line.words]
        matches = sum(1 for w in line_words if w in lyrics_words)

        # Keep line if at least 60% of words match lyrics
        if len(line_words) > 0 and matches / len(line_words) >= 0.6:
            filtered_lines.append(line)

    return filtered_lines


def analyze_audio_energy(audio_path: str, hop_length: int = 512, sr: int = 22050) -> dict:
    """
    Analyze audio to detect vocal activity regions.

    Args:
        audio_path: Path to audio file (preferably isolated vocals)
        hop_length: Samples between frames for RMS calculation
        sr: Sample rate for loading audio

    Returns:
        Dict with 'times', 'energy', 'threshold', and 'is_vocal' arrays
    """
    try:
        import librosa
        import numpy as np
    except ImportError:
        logger.warning("librosa not available, skipping audio energy validation")
        return None

    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=sr)

        # Compute RMS energy
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]

        # Convert frame indices to times
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

        # Compute adaptive threshold (median + std gives robust threshold)
        # Use a higher percentile to focus on actual vocal regions
        threshold = np.percentile(rms, 30)  # 30th percentile as baseline for "silence"

        # Smooth the energy to avoid noise
        from scipy.ndimage import uniform_filter1d
        smoothed_rms = uniform_filter1d(rms, size=10)

        # Detect vocal activity
        is_vocal = smoothed_rms > threshold

        return {
            'times': times,
            'energy': smoothed_rms,
            'threshold': threshold,
            'is_vocal': is_vocal,
            'sr': sr,
            'hop_length': hop_length,
        }
    except Exception as e:
        print(f"  Warning: Failed to analyze audio energy: {e}")
        return None


def get_vocal_activity_at_time(audio_analysis: dict, time: float) -> tuple[bool, float]:
    """
    Check if there's vocal activity at a specific time.

    Returns:
        Tuple of (is_vocal, energy_level)
    """
    if audio_analysis is None:
        return True, 1.0  # Assume vocal if no analysis

    import numpy as np
    times = audio_analysis['times']
    idx = np.searchsorted(times, time)
    idx = min(idx, len(times) - 1)

    return bool(audio_analysis['is_vocal'][idx]), float(audio_analysis['energy'][idx])


def validate_and_fix_timing_with_audio(
    lines: list["Line"],
    audio_analysis: dict,
    min_vocal_ratio: float = 0.3,
) -> list["Line"]:
    """
    Validate lyrics timing against audio energy and fix obvious mismatches.

    This function:
    1. Checks if lyrics lines fall within vocal activity regions
    2. Detects lines placed during silence (likely timing errors)
    3. Attempts to shift misaligned lines to nearby vocal regions
    4. Validates that gaps marked as instrumental actually have low vocal energy

    Args:
        lines: List of Line objects with timing
        audio_analysis: Output from analyze_audio_energy()
        min_vocal_ratio: Minimum ratio of line duration that should have vocal activity

    Returns:
        List of Line objects with validated/corrected timing
    """
    if audio_analysis is None or not lines:
        return lines

    import numpy as np

    times = audio_analysis['times']
    is_vocal = audio_analysis['is_vocal']
    energy = audio_analysis['energy']

    corrected_lines = []
    corrections_made = 0

    for line in lines:
        # Find the time range for this line
        start_time = line.start_time
        end_time = line.end_time

        # Get indices for this time range
        start_idx = np.searchsorted(times, start_time)
        end_idx = np.searchsorted(times, end_time)
        start_idx = min(start_idx, len(times) - 1)
        end_idx = min(end_idx, len(times) - 1)

        if start_idx >= end_idx:
            corrected_lines.append(line)
            continue

        # Check vocal activity during this line
        line_vocal = is_vocal[start_idx:end_idx]
        vocal_ratio = np.mean(line_vocal) if len(line_vocal) > 0 else 0

        if vocal_ratio >= min_vocal_ratio:
            # Line timing seems correct - vocals present
            corrected_lines.append(line)
        else:
            # Line placed during silence - try to find nearby vocal region
            # Search within ±3 seconds for vocal activity
            search_window = 3.0
            search_start = max(0, start_time - search_window)
            search_end = min(times[-1], end_time + search_window)

            search_start_idx = np.searchsorted(times, search_start)
            search_end_idx = np.searchsorted(times, search_end)

            # Find the strongest vocal region in the search window
            search_energy = energy[search_start_idx:search_end_idx]
            search_vocal = is_vocal[search_start_idx:search_end_idx]

            if len(search_energy) > 0 and np.any(search_vocal):
                # Find the peak energy within vocal regions
                vocal_energy = np.where(search_vocal, search_energy, 0)
                peak_idx = np.argmax(vocal_energy)
                peak_time = times[search_start_idx + peak_idx]

                # Calculate offset to shift the line
                line_duration = end_time - start_time
                offset = peak_time - (start_time + line_duration / 2)

                # Only apply correction if offset is significant but not too large
                if 0.5 < abs(offset) < search_window:
                    # Shift the line
                    new_start = start_time + offset
                    new_end = end_time + offset

                    # Shift all words proportionally
                    new_words = []
                    for word in line.words:
                        new_word = Word(
                            text=word.text,
                            start_time=word.start_time + offset,
                            end_time=word.end_time + offset,
                            singer=word.singer,
                        )
                        new_words.append(new_word)

                    corrected_line = Line(
                        words=new_words,
                        start_time=new_start,
                        end_time=new_end,
                        singer=line.singer,
                    )
                    corrected_lines.append(corrected_line)
                    corrections_made += 1
                    continue

            # Couldn't find better placement, keep original
            corrected_lines.append(line)

    if corrections_made > 0:
        print(f"  Audio validation: corrected timing for {corrections_made} lines")

    return corrected_lines


def validate_instrumental_breaks(
    lines: list["Line"],
    audio_analysis: dict,
    break_threshold: float = 8.0,
    max_vocal_ratio: float = 0.2,
) -> list[tuple[float, float, bool]]:
    """
    Validate that gaps between lines actually correspond to instrumental sections.

    Returns:
        List of (gap_start, gap_end, is_valid_break) tuples
    """
    if audio_analysis is None or len(lines) < 2:
        return []

    import numpy as np

    times = audio_analysis['times']
    is_vocal = audio_analysis['is_vocal']
    results = []

    for i in range(len(lines) - 1):
        gap_start = lines[i].end_time
        gap_end = lines[i + 1].start_time
        gap_duration = gap_end - gap_start

        if gap_duration >= break_threshold:
            # Check vocal activity during the gap
            start_idx = np.searchsorted(times, gap_start)
            end_idx = np.searchsorted(times, gap_end)
            start_idx = min(start_idx, len(times) - 1)
            end_idx = min(end_idx, len(times) - 1)

            if start_idx < end_idx:
                gap_vocal = is_vocal[start_idx:end_idx]
                vocal_ratio = np.mean(gap_vocal)

                # Gap is valid if mostly non-vocal
                is_valid = vocal_ratio <= max_vocal_ratio
                results.append((gap_start, gap_end, is_valid))

                if not is_valid:
                    print(f"  Warning: Gap at {gap_start:.1f}s-{gap_end:.1f}s has {vocal_ratio:.0%} vocal activity")

    return results


def get_lyrics(
    title: str,
    artist: str,
    vocals_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> tuple[list[Line], Optional[SongMetadata]]:
    """
    Get lyrics with accurate word-level timing and optional singer info.

    Strategy:
    1. Try to fetch Genius lyrics with singer annotations (for duets)
    2. Fetch synced lyrics from online sources (preferred - has timing + correct text)
    3. If both found, merge timing with singer info
    4. If no synced lyrics, fall back to whisperx transcription

    Returns:
        Tuple of (lines, metadata)
        - lines: List of Line objects with word timing (and singer info if duet)
        - metadata: SongMetadata with singer info, or None if not a duet
    """
    # Try to fetch Genius lyrics with singer annotations, with simple
    # on-disk caching when a cache_dir is provided.
    logger.info("Checking for singer annotations (Genius)...")
    genius_lines: Optional[list[tuple[str, str]]] = None
    metadata: Optional[SongMetadata] = None

    genius_cache_path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        # Include artist and title in cache filename to handle overrides
        import hashlib
        cache_key = hashlib.md5(f"{artist}:{title}".encode()).hexdigest()[:8]
        genius_cache_path = os.path.join(cache_dir, f"genius_cache_{cache_key}.json")
        if os.path.exists(genius_cache_path):
            try:
                with open(genius_cache_path, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                cached_lines = cached.get("lines", [])
                genius_lines = [(item["text"], item.get("singer", "")) for item in cached_lines]
                metadata = _metadata_from_json(cached.get("metadata"))
                logger.info("Using cached Genius lyrics")
            except Exception:
                genius_lines = None
                metadata = None

    if genius_lines is None:
        genius_lines, metadata = fetch_genius_lyrics_with_singers(title, artist)
        if genius_cache_path and genius_lines is not None:
            try:
                payload = {
                    "lines": [
                        {"text": text, "singer": singer}
                        for (text, singer) in genius_lines
                    ],
                    "metadata": _metadata_to_json(metadata),
                }
                with open(genius_cache_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

    # Extract plain Genius lyrics text (without singer info) for later use.
    # Filter out obvious non-lyric lines like contributor headers and
    # page titles (e.g., "3 Contributors... Lyrics[Verse]").
    if genius_lines:
        def _is_probable_lyric(text: str) -> bool:
            t = text.strip()
            if not t:
                return False
            lower = t.lower()
            if "contributors" in lower:
                return False
            if " lyrics" in lower or lower.endswith(" lyrics"):
                return False
            if "lyrics[" in lower:
                return False
            # Drop pure section headers like [Verse], [Chorus]
            if t.startswith("[") and t.endswith("]"):
                return False
            return True

        genius_lyrics_text = [text for text, _ in genius_lines if _is_probable_lyric(text)] or None
    else:
        genius_lyrics_text = None

    # Try to fetch lyrics from multiple sources
    logger.info("Fetching lyrics from online sources...")
    lrc_text, is_synced, source = fetch_lyrics_multi_source(title, artist)

    # If we found synced lyrics, but we also have Genius lyrics, make
    # sure they look like the same song before trusting the LRC file.
    if lrc_text and is_synced and genius_lyrics_text:
        from difflib import SequenceMatcher

        lrc_plain = extract_lyrics_text(lrc_text, title, artist)
        # Compare up to the first N non-empty lines to avoid being
        # tricked by completely wrong matches from providers.
        max_lines = 10
        lrc_sample = [ln for ln in lrc_plain if ln.strip()][:max_lines]
        genius_sample = [ln for ln in genius_lyrics_text if ln.strip()][:max_lines]

        def best_match_score(src: str, candidates: list[str]) -> float:
            if not candidates:
                return 0.0
            src_norm = normalize_text(src)
            if not src_norm:
                return 0.0
            best = 0.0
            for cand in candidates:
                cand_norm = normalize_text(cand)
                if not cand_norm:
                    continue
                s = SequenceMatcher(None, src_norm, cand_norm).ratio()
                if s > best:
                    best = s
            return best

        if lrc_sample and genius_sample:
            # Try romanizing synced lyrics if they contain non-Latin scripts
            logger.info("Checking if synced lyrics need romanization...")
            lrc_sample_romanized = [romanize_line(ln) if any(ord(c) > 127 for c in ln) else ln 
                                   for ln in lrc_sample]
            
            scores = [best_match_score(ln, genius_sample) for ln in lrc_sample_romanized]
            avg_score = sum(scores) / len(scores) if scores else 0.0
            print(f"Match score after romanization: {avg_score:.2f}")
            
            # If romanization improved the match, romanize all synced lyrics
            if avg_score >= 0.4 and lrc_sample_romanized != lrc_sample:
                print(f"Romanizing synced lyrics for better match (score: {avg_score:.2f})")
                lrc_lines = lrc_text.split('\n')
                romanized_lines = []
                for i, line in enumerate(lrc_lines):
                    if i % 10 == 0:
                        print(f"  Romanizing line {i}/{len(lrc_lines)}...")
                    # Preserve timing tags, romanize only the text
                    if line.strip().startswith('[') and ']' in line:
                        tag_end = line.index(']') + 1
                        timing = line[:tag_end]
                        text = line[tag_end:]
                        romanized_lines.append(timing + romanize_line(text))
                    else:
                        romanized_lines.append(romanize_line(line) if line.strip() else line)
                lrc_text = '\n'.join(romanized_lines)
                logger.info("✓ Romanization complete")
        else:
            avg_score = 0.0

        if avg_score < 0.75:
            print(f"Synced lyrics match score too low ({avg_score:.2f}); using WhisperX with Genius lyrics instead.")
            lrc_text = None
            is_synced = False

    if lrc_text and is_synced:
        # We have synced lyrics - use them for timing
        print(f"Found synced lyrics from {source}")
        
        # Prefer Genius lyrics text (more complete) with synced timing
        if genius_lyrics_text and avg_score >= 0.75:
            print(f"Using Genius lyrics text with synced timing (match score: {avg_score:.2f})")
            print(f"DEBUG: genius_lyrics_text has {len(genius_lyrics_text)} lines")
            if len(genius_lyrics_text) > 12:
                print(f"DEBUG: genius_lyrics_text[12] = {repr(genius_lyrics_text[12])}")
            lyrics_text = genius_lyrics_text
        else:
            # Extract lyrics text from synced source
            lyrics_text = extract_lyrics_text(lrc_text, title, artist)
        
        # If we have vocals, use WhisperX for accurate word-level timing
        if vocals_path:
            logger.info("Using WhisperX for accurate word-level timing with synced lyrics as reference...")
            # Fall through to WhisperX transcription below
        else:
            # No vocals available, use synced lyrics timing directly
            print(f"Using synced lyrics timing from {source}")
            if genius_lines and metadata and metadata.is_duet:
                print(f"Merging with singer info from Genius")
                timed_lines = parse_lrc_with_timing(lrc_text, title, artist)
                lines = merge_lyrics_with_singer_info(timed_lines, genius_lines, metadata)
            else:
                lines = create_lines_from_lrc(lrc_text, title=title, artist=artist)

            if lines:
                lines = split_long_lines(lines)
                print(f"Got {len(lines)} lines of lyrics")
                # Cache the final result
                if cache_dir:
                    try:
                        payload = {
                            "lines": _lines_to_json(lines),
                            "metadata": _metadata_to_json(metadata) if metadata else None
                        }
                        with open(os.path.join(cache_dir, "lyrics_final.json"), "w", encoding="utf-8") as f:
                            json.dump(payload, f, ensure_ascii=False, indent=2)
                        print(f"✓ Cached final lyrics result")
                    except Exception as e:
                        print(f"  Warning: Failed to cache final lyrics: {e}")
                return lines, metadata

    # Fall back to whisperx transcription if no synced lyrics
    if not vocals_path:
        raise RuntimeError("Could not get lyrics: no synced lyrics found and no vocals path provided")

    # Initialize lyrics_text
    lyrics_text = None
    
    # If we only have unsynced/plain lyrics from providers and ALSO have
    # Genius lyrics, prefer Genius as the reference text and ignore the
    # plain provider lyrics.
    if lrc_text and (not is_synced) and genius_lyrics_text:
        logger.info("Using whisperx with lyrics reference from Genius (ignoring unsynced lyrics from providers)")
        lyrics_text = [line for line in genius_lyrics_text if line.strip()]
    elif lrc_text and not lyrics_text:
        # Have plain lyrics (no timing) - use whisperx but try to match text
        # Only use if lyrics_text wasn't already set above
        print(f"Using whisperx with lyrics reference from {source}")
        lyrics_text = extract_lyrics_text(lrc_text, title, artist)
    elif genius_lyrics_text and not lyrics_text:
        # No LRC from providers, but Genius lyrics are available
        # Only use if lyrics_text wasn't already set above
        logger.info("Using whisperx with lyrics reference from Genius")
        # Filter out any empty lines just in case
        lyrics_text = [line for line in genius_lyrics_text if line.strip()]
    elif not lyrics_text:
        logger.info("No lyrics found online, will transcribe from audio only")
        lyrics_text = None

    # Transcribe and align with whisperx, with optional caching of the
    # pre-alignment transcription so we don't have to rerun WhisperX
    # on subsequent runs.
    
    # First check if we have a complete cached result
    final_cache_path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        final_cache_path = os.path.join(cache_dir, "lyrics_final.json")
        
        if os.path.exists(final_cache_path):
            try:
                with open(final_cache_path, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                lines = _lines_from_json(cached.get("lines", []))
                metadata = _metadata_from_json(cached.get("metadata"))
                print(f"✓ Using cached final lyrics ({len(lines)} lines)")
                # Split long lines even when loading from cache
                lines = split_long_lines(lines)
                # Romanize any non-Latin text (for mixed-language songs)
                for line in lines:
                    for word in line.words:
                        if any(ord(c) > 127 for c in word.text):
                            word.text = romanize_line(word.text)
                return lines, metadata
            except Exception as e:
                print(f"  Cache read failed: {e}, will regenerate")
    
    # No final cache, proceed with transcription
    # Detect language from lyrics first for cache key
    detected_language = None
    if lyrics_text:
        sample = " ".join(lyrics_text[:5]).lower()
        if any(word in sample for word in ["el", "la", "los", "las", "que", "de", "con", "por", "para"]):
            detected_language = "es"
    
    if cache_dir:
        # Include language in cache filename to avoid conflicts
        lang_suffix = f"_{detected_language}" if detected_language else ""
        transcript_path = os.path.join(cache_dir, f"whisper_transcript{lang_suffix}.json")
    else:
        transcript_path = None

    if transcript_path and os.path.exists(transcript_path):
        try:
            with open(transcript_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            
            # Check if cached transcript was made with same lyrics
            cached_lyrics_hash = raw.get("lyrics_hash")
            current_lyrics_hash = hash(tuple(lyrics_text)) if lyrics_text else None
            
            if cached_lyrics_hash == current_lyrics_hash:
                lines = _lines_from_json(raw.get("lines", []))
                print(f"Loaded cached Whisper transcription with {len(lines)} lines")
            else:
                print(f"Cached transcript has different lyrics, regenerating...")
                lines = transcribe_and_align(vocals_path, lyrics_text)
        except Exception:
            lines = transcribe_and_align(vocals_path, lyrics_text)
    else:
        lines = transcribe_and_align(vocals_path, lyrics_text)
        if transcript_path:
            try:
                payload = {
                    "lines": _lines_to_json(lines),
                    "language": detected_language,
                    "lyrics_hash": hash(tuple(lyrics_text)) if lyrics_text else None
                }
                with open(transcript_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False)
                print(f"✓ Cached Whisper transcription")
            except Exception:
                pass

    # Cross-check and correct transcription using known lyrics (if we have them)
    # If we have synced lyrics with line timing AND Genius lyrics, use synced timing
    # with Genius text (which is more complete)
    synced_timings = None
    if lrc_text and is_synced:
        synced_timings = parse_lrc_with_timing(lrc_text, title, artist)
        if genius_lyrics_text and avg_score >= 0.75:
            # CRITICAL: Override lyrics_text with Genius text for absolute order
            lyrics_text = [line for line in genius_lyrics_text if line.strip()]
            print(f"Using hybrid alignment: Genius text (absolute order) + synced timing hints")
            print(f"  Genius lines: {len(lyrics_text)}, Synced timings: {len(synced_timings)}")
        else:
            print(f"Using hybrid alignment: synced line timing + WhisperX word timing")
    
    if lyrics_text:
        original_count = len(lines)
        lines = correct_transcription_with_lyrics(lines, lyrics_text, synced_timings)
        print(f"Corrected transcription: {original_count} lines processed, {len(lines)} lines returned")
        
        # Debug: check word order after correction
        if genius_lyrics_text and avg_score >= 0.75:
            all_words = [w.text for line in lines for w in line.words]
            print(f"  After correction: {len(all_words)} words, first 5: {all_words[:5]}")
    
    # Romanize any non-Latin text before caching (for mixed-language songs)
    for line in lines:
        for word in line.words:
            if any(ord(c) > 127 for c in word.text):
                word.text = romanize_line(word.text)
    
    # Split long lines for display
    lines = split_long_lines(lines)
    
    # Debug: check word order after splitting
    if genius_lyrics_text and avg_score >= 0.75:
        all_words = [w.text for line in lines for w in line.words]
        print(f"  After splitting: {len(all_words)} words, first 5: {all_words[:5]}")
        print(f"  Lines 14-17 after split: {[(i, lines[i].start_time, ' '.join([w.text for w in lines[i].words])[:30]) for i in range(14, min(17, len(lines)))]}")

    # Sort lines by start time ONLY if not using Genius text as absolute source
    # When using Genius text, the order is already correct and must be preserved
    if not (genius_lyrics_text and avg_score >= 0.75 and lrc_text and is_synced):
        lines.sort(key=lambda l: l.start_time)
    else:
        print(f"  Skipping sort to preserve Genius order")

    # Fix overlapping lines by capping end_time at next line's start_time
    for i in range(len(lines) - 1):
        if lines[i].end_time > lines[i + 1].start_time:
            # Cap end_time with a small gap for visual separation
            lines[i].end_time = lines[i + 1].start_time - 0.1
            # Also update the last word's end_time
            if lines[i].words:
                lines[i].words[-1].end_time = lines[i].end_time

    # Validate and fix timing using audio energy analysis
    if vocals_path:
        logger.info("Validating timing against audio energy...")
        audio_analysis = analyze_audio_energy(vocals_path)
        if audio_analysis is not None:
            lines = validate_and_fix_timing_with_audio(lines, audio_analysis)
            # Check for problematic instrumental breaks
            break_issues = validate_instrumental_breaks(lines, audio_analysis)
            if break_issues:
                invalid_breaks = [(s, e) for s, e, valid in break_issues if not valid]
                if invalid_breaks:
                    print(f"  Found {len(invalid_breaks)} gaps with unexpected vocal activity")
            # Re-sort after any corrections ONLY if not using Genius text
            if not (genius_lyrics_text and avg_score >= 0.75 and lrc_text and is_synced):
                lines.sort(key=lambda l: l.start_time)
            else:
                # When using Genius text, fix any temporal ordering issues without sorting
                for i in range(1, len(lines)):
                    if lines[i].start_time < lines[i-1].end_time:
                        # Shift this line to start after previous line ends
                        shift = lines[i-1].end_time - lines[i].start_time + 0.1
                        lines[i].start_time += shift
                        lines[i].end_time += shift
                        for word in lines[i].words:
                            word.start_time += shift
                            word.end_time += shift

    # Cache the final result
    if final_cache_path:
        try:
            payload = {
                "lines": _lines_to_json(lines),
                "metadata": _metadata_to_json(metadata) if metadata else None
            }
            with open(final_cache_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            print(f"✓ Cached final lyrics result")
        except Exception as e:
            print(f"  Warning: Failed to cache final lyrics: {e}")

    return lines, metadata


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        logger.info("Usage: python lyrics.py <title> <artist> [vocals_path]")
        sys.exit(1)

    title = sys.argv[1]
    artist = sys.argv[2]
    vocals_path = sys.argv[3] if len(sys.argv) > 3 else None

    lines, metadata = get_lyrics(title, artist, vocals_path)

    if metadata and metadata.is_duet:
        print(f"\nDuet detected: {', '.join(metadata.singers)}")

    print(f"\nFound {len(lines)} lines:")
    for line in lines[:10]:  # Show first 10 lines
        singer_info = f" [{line.singer}]" if line.singer else ""
        print(f"[{line.start_time:.2f}s]{singer_info} {line.text}")
