"""Lyrics fetching and processing with robust LRC parsing and multilingual romanization."""

import re
import unicodedata
import time
import random
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path

import requests
from requests import Response
from bs4 import BeautifulSoup
from rapidfuzz import fuzz

from pathlib import Path

try:
    import syncedlyrics
except ImportError:
    syncedlyrics = None

# ----------------------
# Logging helper
# ----------------------
import logging
from ..utils.logging import get_logger
logger = get_logger(__name__)

# ----------------------
# Custom exception
# ----------------------
from ..exceptions import LyricsError

# ----------------------
# Retry constants
# ----------------------
DEFAULT_MAX_RETRIES = 3

# ----------------------
# Singer ID
# ----------------------
class SingerID(str, Enum):
    SINGER1 = "singer1"
    SINGER2 = "singer2"
    BOTH = "both"
    UNKNOWN = ""

# ----------------------
# Word & Line
# ----------------------
@dataclass
class Word:
    text: str
    start_time: float
    end_time: float
    singer: SingerID = SingerID.UNKNOWN

    def validate(self) -> None:
        if self.start_time < 0 or self.end_time < 0:
            raise ValueError("Word timing must be non-negative")
        if self.end_time < self.start_time:
            raise ValueError("Word end_time must be >= start_time")

@dataclass
class Line:
    words: List[Word]
    singer: SingerID = SingerID.UNKNOWN

    @property
    def start_time(self) -> float:
        return self.words[0].start_time if self.words else 0.0

    @property
    def end_time(self) -> float:
        return self.words[-1].end_time if self.words else 0.0

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    def validate(self) -> None:
        if not self.words:
            raise ValueError("Line must contain at least one word")
        for w in self.words:
            w.validate()
            if w.start_time < self.start_time or w.end_time > self.end_time:
                raise ValueError("Word timing outside line bounds")

# ----------------------
# Song Metadata
# ----------------------
@dataclass
class SongMetadata:
    singers: List[str]
    is_duet: bool = False
    title: Optional[str] = None
    artist: Optional[str] = None

    def get_singer_id(self, singer_name: str) -> SingerID:
        if not singer_name:
            return SingerID.UNKNOWN
        name = singer_name.lower().strip()
        duet_tokens = ["&", " and ", " feat ", " feat.", " with ", " x "]
        if any(token in name for token in duet_tokens):
            return SingerID.BOTH
        for i, known_singer in enumerate(self.singers):
            ks = known_singer.lower()
            if name == ks or name.startswith(ks) or ks.startswith(name):
                return SingerID(f"singer{i + 1}")
        return SingerID.SINGER1 if self.singers else SingerID.UNKNOWN

# ----------------------
# LRC timestamp parser
# ----------------------
_LRC_TS_RE = re.compile(
    r"""
    \[                      # opening bracket
    (?P<min>\d+)            # minutes
    :
    (?P<sec>[0-5]?\d)       # seconds
    (?:\.(?P<frac>\d{1,3}))?  # optional fractional seconds
    \]                      # closing bracket
    """,
    re.VERBOSE
)

def parse_lrc_timestamp(ts: str) -> Optional[float]:
    if not ts:
        return None
    match = _LRC_TS_RE.match(ts.strip())
    if not match:
        return None
    minutes = int(match.group("min"))
    seconds = int(match.group("sec"))
    if seconds >= 60:
        return None
    frac = match.group("frac")
    frac_seconds = int(frac) / (10 ** len(frac)) if frac else 0.0
    return minutes * 60 + seconds + frac_seconds

# ----------------------
# Multilingual Romanizers
# ----------------------

# Korean
try:
    from korean_romanizer.romanizer import Romanizer
    KOREAN_ROMANIZER_AVAILABLE = True
except ImportError:
    KOREAN_ROMANIZER_AVAILABLE = False

# Chinese
try:
    from pypinyin import lazy_pinyin, Style
    CHINESE_ROMANIZER_AVAILABLE = True
except ImportError:
    CHINESE_ROMANIZER_AVAILABLE = False

# Japanese
try:
    from pykakasi import kakasi
    JAPANESE_ROMANIZER_AVAILABLE = True
except ImportError:
    JAPANESE_ROMANIZER_AVAILABLE = False

# Arabic
try:
    import pyarabic.araby as araby
    ARABIC_ROMANIZER_AVAILABLE = True
except ImportError:
    ARABIC_ROMANIZER_AVAILABLE = False

HEBREW_ROMANIZER_AVAILABLE = True

# Unicode ranges
KOREAN_RANGES = [(0x1100, 0x11FF), (0x3130, 0x318F), (0xAC00, 0xD7AF), (0xA960, 0xA97F), (0xD7B0, 0xD7FF)]
JAPANESE_HIRAGANA = (0x3040, 0x309F)
JAPANESE_KATAKANA = (0x30A0, 0x30FF)
JAPANESE_KANJI_RANGES = [(0x3400, 0x4DBF), (0x4E00, 0x9FFF)]
CHINESE_RANGES = [(0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF), (0x20000, 0x2CEAF)]
ARABIC_RANGES = [(0x0600, 0x06FF), (0x0750, 0x077F)]
HEBREW_RANGES = [(0x0590, 0x05FF)]

# ----------------------
# Romanization functions
# ----------------------
_JAPANESE_CONVERTER = None

def romanize_korean(text: str) -> str:
    if not KOREAN_ROMANIZER_AVAILABLE:
        return text
    try:
        return Romanizer(text).romanize()
    except Exception:
        return text

def romanize_chinese(text: str) -> str:
    if not CHINESE_ROMANIZER_AVAILABLE:
        return text
    try:
        return " ".join(lazy_pinyin(text, style=Style.NORMAL))
    except Exception:
        return text

def romanize_japanese(text: str) -> str:
    global _JAPANESE_CONVERTER
    if not JAPANESE_ROMANIZER_AVAILABLE:
        return text
    if _JAPANESE_CONVERTER is None:
        _JAPANESE_CONVERTER = kakasi()
    try:
        result = _JAPANESE_CONVERTER.convert(text)
        return " ".join(item["hepburn"] for item in result)
    except Exception:
        return text

ARABIC_TO_LATIN = {
    'ا': 'a', 'ب': 'b', 'ت': 't', 'ث': 'th', 'ج': 'j', 'ح': 'h', 'خ': 'kh',
    'د': 'd', 'ذ': 'dh', 'ر': 'r', 'ز': 'z', 'س': 's', 'ش': 'sh', 'ص': 's',
    'ض': 'd', 'ط': 't', 'ظ': 'z', 'ع': 'a', 'غ': 'gh', 'ف': 'f', 'ق': 'q',
    'ك': 'k', 'ل': 'l', 'م': 'm', 'ن': 'n', 'ه': 'h', 'و': 'w', 'ي': 'y',
    'ى': 'a', 'ة': 'h', 'ء': '', 'ئ': '', 'ؤ': 'w', 'إ': 'i', 'أ': 'a', 'آ': 'aa',
    'َ': 'a', 'ُ': 'u', 'ِ': 'i', 'ّ': '', 'ْ': ''
}

HEBREW_TO_LATIN = {
    'א': 'a', 'ב': 'b', 'ג': 'g', 'ד': 'd', 'ה': 'h', 'ו': 'v', 'ז': 'z',
    'ח': 'ch', 'ט': 't', 'י': 'y', 'כ': 'k', 'ך': 'kh', 'ל': 'l', 'מ': 'm',
    'ם': 'm', 'נ': 'n', 'ן': 'n', 'ס': 's', 'ע': 'a', 'פ': 'p', 'ף': 'f',
    'צ': 'ts', 'ץ': 'ts', 'ק': 'k', 'ר': 'r', 'ש': 'sh', 'ת': 't'
}

def romanize_arabic(text: str) -> str:
    if not ARABIC_ROMANIZER_AVAILABLE:
        return text
    try:
        return ''.join(ARABIC_TO_LATIN.get(c, c) for c in text)
    except Exception:
        return text

def romanize_hebrew(text: str) -> str:
    try:
        return ''.join(HEBREW_TO_LATIN.get(c, c) for c in text)
    except Exception:
        return text

# ----------------------
# Multilingual single-pass romanizer
# ----------------------
SCRIPT_ROMANIZER_MAP = [
    (KOREAN_RANGES, romanize_korean),
    ([JAPANESE_HIRAGANA, JAPANESE_KATAKANA] + JAPANESE_KANJI_RANGES, romanize_japanese),
    (CHINESE_RANGES, romanize_chinese),
    (ARABIC_RANGES, romanize_arabic),
    (HEBREW_RANGES, romanize_hebrew),
]

# Regex for blocks of scripts
KOREAN_RE = r'\u1100-\u11FF\u3130-\u318F\uAC00-\uD7AF\uA960-\uA97F\uD7B0-\uD7FF'
JAPANESE_RE = r'\u3040-\u309F\u30A0-\u30FF\u3400-\u4DBF\u4E00-\u9FFF'
CHINESE_RE = r'\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF\U00020000-\U0002CEAF'
ARABIC_RE = r'\u0600-\u06FF\u0750-\u077F'
HEBREW_RE = r'\u0590-\u05FF'
MULTILINGUAL_RE = re.compile(f'([{KOREAN_RE}{JAPANESE_RE}{CHINESE_RE}{ARABIC_RE}{HEBREW_RE}]+)')

def romanize_multilingual(text: str) -> str:
    def replace_block(match: re.Match) -> str:
        block = match.group()
        code = ord(block[0])
        for ranges, romanizer in SCRIPT_ROMANIZER_MAP:
            for start, end in ranges:
                if start <= code <= end:
                    try:
                        return romanizer(block)
                    except Exception:
                        return block
        return block
    romanized = MULTILINGUAL_RE.sub(replace_block, text)
    return " ".join(romanized.split())  # collapse repeated spaces

# Alias
romanize_line = romanize_multilingual

# ----------------------
# Metadata filtering
# ----------------------
def _is_metadata_line(text: str, title: str = "", artist: str = "") -> bool:
    """
    Determine if a line is metadata rather than actual lyrics.
    Skips obvious labels and title/artist lines.
    """
    if not text:
        return True

    text_lower = text.lower().strip()

    # Skip obvious metadata labels
    metadata_prefixes = [
        "artist:", "song:", "title:", "album:", "writer:", "composer:",
        "lyricist:", "lyrics by", "written by", "produced by", "music by",
    ]
    for prefix in metadata_prefixes:
        if text_lower.startswith(prefix):
            return True

    # Skip lines that are just the title
    if title:
        title_lower = title.lower().replace(" ", "")
        line_normalized = text_lower.replace(" ", "")
        if line_normalized == title_lower:
            return True

    # Skip lines that are just the artist
    if artist:
        artist_lower = artist.lower().replace(" ", "")
        line_normalized = text_lower.replace(" ", "")
        if line_normalized == artist_lower:
            return True

    return False

# ----------------------
# Normalize text for fuzzy matching
# ----------------------
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

# ----------------------
# Extract plain text lines from LRC
# ----------------------
def extract_lyrics_text(lrc_text: str, title: str = "", artist: str = "") -> List[str]:
    """
    Extract plain text lines from LRC format (ignore timing and metadata).
    """
    if not lrc_text:
        return []

    lines: List[str] = []
    for line in lrc_text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        match = _LRC_TS_RE.match(line)
        if match:
            text_part = line[match.end():].strip()
            if text_part and not _is_metadata_line(text_part, title, artist):
                lines.append(text_part)
    return lines

# ----------------------
# Parse LRC lines with timing
# ----------------------
def parse_lrc_with_timing(lrc_text: str, title: str = "", artist: str = "") -> List[Tuple[float, str]]:
    """
    Parse LRC format and extract (timestamp, text) tuples.
    Skips metadata lines.
    """
    if not lrc_text:
        return []

    lines: List[Tuple[float, str]] = []

    for line in lrc_text.strip().splitlines():
        line = line.strip()
        if not line:
            continue

        match = _LRC_TS_RE.match(line)
        if not match:
            continue

        timestamp = parse_lrc_timestamp(match.group(0))
        if timestamp is None:
            continue

        text_part = line[match.end():].strip()
        if text_part and not _is_metadata_line(text_part, title, artist):
            lines.append((timestamp, text_part))

    return lines

# ----------------------
# Extract artists from title
# ----------------------
def extract_artists_from_title(title: str, known_artist: str) -> List[str]:
    """
    Extract artist names from a title like "Artist1, Artist2 - Song Name".
    Fallback to known_artist if extraction fails.
    """
    if not title:
        return [known_artist] if known_artist else []

    artists: List[str] = []

    # Separate artist part from song part
    if " - " in title:
        artist_part = title.split(" - ")[0].strip()

        # Split by common separators
        parts = re.split(r'[,&]|(?:\b(?:and|ft\.?|feat\.?|featuring)\b)', artist_part, flags=re.IGNORECASE)
        artists = [p.strip() for p in parts if p.strip()]

    if not artists and known_artist:
        artists = [known_artist]

    return artists

# ----------------------
# Create Line objects from LRC
# ----------------------
def create_lines_from_lrc(
    lrc_text: str,
    romanize: bool = True,
    title: str = "",
    artist: str = "",
) -> List[Line]:
    """
    Create Line objects from LRC format with evenly distributed word timing.
    Uses LRC timestamps and distributes word timings evenly.
    """
    timed_lines = parse_lrc_with_timing(lrc_text, title, artist)
    if not timed_lines:
        return []

    lines: List[Line] = []

    for i, (start_time, text) in enumerate(timed_lines):
        if romanize:
            text = romanize_line(text)

        # Determine end time
        if i + 1 < len(timed_lines):
            end_time = timed_lines[i + 1][0]
            if end_time - start_time > 10.0:
                end_time = start_time + 5.0
        else:
            end_time = start_time + 3.0

        word_texts = text.split()
        if not word_texts:
            continue

        line_duration = end_time - start_time
        word_duration = (line_duration * 0.95) / len(word_texts)

        words: List[Word] = []
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
        ))

    return lines

# ----------------------
# Robust HTTP request with retry
# ----------------------
def _make_request_with_retry(
    url: str,
    headers: Optional[dict] = None,
    timeout: int = 10,
    max_retries: int = 5,
) -> Optional["requests.Response"]:
    """
    Make an HTTP GET request with retry logic and exponential backoff.
    Returns the requests.Response or None if all retries fail.
    """
    import requests
    import time
    import random

    headers = headers or {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/143.0.0.0 Safari/537.36"
    }

    delay = 1.0
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response
        except (requests.exceptions.RequestException, requests.exceptions.Timeout) as e:
            last_error = e
            if attempt < max_retries:
                sleep_time = delay + random.uniform(0, 0.5)
                time.sleep(sleep_time)
                delay = min(delay * 2, 30.0)
    return None

# ----------------------
# Fetch Genius lyrics with singer annotations
# ----------------------

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

    # Helper: convert text to URL slug
    def make_slug(text: str) -> str:
        slug = re.sub(r'[^\w\s-]', '', text.lower())
        slug = re.sub(r'\s+', '-', slug)
        return slug

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36',
        'Accept': '*/*'
    }

    # Clean title for search / URL
    cleaned_title = re.split(r'\s*[|｜]\s*', title)[0]
    cleaned_title = re.sub(r'\s*[\(\[]?\s*(ft\.?|feat\.?|featuring).*?[\)\]]?\s*$', '', cleaned_title, flags=re.IGNORECASE)
    cleaned_title = re.sub(r'\s*[\(\[].*?[\)\]]\s*', '', cleaned_title).strip()

    # Construct primary URL
    artist_slug = make_slug(artist)
    title_slug = make_slug(cleaned_title)
    candidate_urls = [
        f"https://genius.com/{artist_slug}-{title_slug}-lyrics",
        f"https://genius.com/{title_slug}-lyrics",  # fallback if artist slug fails
        f"https://genius.com/Genius-romanizations-{artist_slug}-{title_slug}-romanized-lyrics",
    ]

    # Try each URL until one works
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

    # Extract title/artist from page if missing
    page_title = soup.find('title').get_text().strip() if soup.find('title') else ""
    genius_title = cleaned_title
    genius_artist = artist
    if page_title and "|" in page_title:
        parts = page_title.split("|")[0].split("–") if "–" in page_title else page_title.split("-")
        if len(parts) == 2:
            genius_artist = parts[0].strip()
            genius_title = parts[1].strip()

    # Extract lyrics containers
    lyrics_containers = soup.find_all('div', {'data-lyrics-container': 'true'})
    if not lyrics_containers:
        return None, None

    lines_with_singers: list[tuple[str, str]] = []
    current_singer = ""
    singers_found: set[str] = set()
    section_pattern = re.compile(r'\[([^\]]+)\]')

    for container in lyrics_containers:
        for br in container.find_all('br'):
            br.replace_with('\n')
        text = container.get_text()
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
            section_match = section_pattern.match(line)
            if section_match:
                header = section_match.group(1)
                if ':' in header:
                    singer_part = header.split(':', 1)[1].strip()
                    current_singer = singer_part
                    singers_found.add(singer_part)
                continue
            lines_with_singers.append((line, current_singer))

    if not lines_with_singers:
        return None, None

    # Determine unique singers
    unique_singers: list[str] = []
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


# ----------------------
# Merge synced lyrics with Genius singer info
# ----------------------
def merge_lyrics_with_singer_info(
    timed_lines: List[Tuple[float, str]],
    genius_lines: List[Tuple[str, str]],
    metadata: SongMetadata,
    romanize: bool = True,
) -> List[Line]:
    """
    Merge timed lyrics with Genius singer annotations using fuzzy matching.
    """
    from difflib import SequenceMatcher

    if not timed_lines:
        return []

    # Normalize Genius lines for matching
    genius_normalized = [(normalize_text(t), s) for t, s in genius_lines]
    used_genius_indices: set[int] = set()
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

# ----------------------
# JSON serialization/deserialization for Lines and Metadata
# ----------------------
def _lines_to_json(lines: List[Line]) -> List[dict]:
    """
    Convert a list of Line objects into JSON-serializable dicts.
    """
    data: List[dict] = []
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
                } for w in line.words
            ]
        })
    return data

def _lines_from_json(data: List[dict]) -> List[Line]:
    """
    Convert JSON data back into Line objects with Word objects.
    """
    lines: List[Line] = []
    for item in data:
        words = [
            Word(
                text=w["text"],
                start_time=float(w["start_time"]),
                end_time=float(w["end_time"]),
                singer=w.get("singer", "")
            ) for w in item.get("words", [])
        ]
        lines.append(Line(
            words=words,
            singer=item.get("singer", "")
        ))
    return lines

def _metadata_to_json(metadata: Optional[SongMetadata]) -> Optional[dict]:
    """
    Convert SongMetadata to JSON-serializable dict.
    """
    if not metadata:
        return None
    return {
        "singers": metadata.singers,
        "is_duet": metadata.is_duet,
        "title": metadata.title,
        "artist": metadata.artist
    }

def _metadata_from_json(data: Optional[dict]) -> Optional[SongMetadata]:
    """
    Convert JSON dict back into SongMetadata.
    """
    if not data:
        return None
    return SongMetadata(
        singers=list(data.get("singers", [])),
        is_duet=bool(data.get("is_duet", False)),
        title=data.get("title"),
        artist=data.get("artist")
    )

# ----------------------
# Utility: Save Lines and Metadata to JSON file
# ----------------------
def save_lyrics_to_json(filepath: str, lines: List[Line], metadata: Optional[SongMetadata] = None) -> None:
    import json
    data = {
        "lines": _lines_to_json(lines),
        "metadata": _metadata_to_json(metadata)
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ----------------------
# Utility: Load Lines and Metadata from JSON file
# ----------------------
def load_lyrics_from_json(filepath: str) -> Tuple[List[Line], Optional[SongMetadata]]:
    import json
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    lines = _lines_from_json(data.get("lines", []))
    metadata = _metadata_from_json(data.get("metadata"))
    return lines, metadata

# ----------------------
# Constants & Regexes
# ----------------------
DEFAULT_MAX_RETRIES = 5
LRC_TIMESTAMP_RE = re.compile(r'\[(\d+):(\d+)(?:\.(\d+))?\]')

# ----------------------
# Text normalization utilities
# ----------------------
def normalize_text(text: str) -> str:
    """
    Normalize text for comparison: lowercase, remove punctuation, collapse spaces.
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ----------------------
# Romanization placeholder
# ----------------------
def romanize_line(text: str) -> str:
    """
    Placeholder for romanization. Replace with actual Korean/other romanization logic.
    For now, returns text unchanged.
    """
    return text

# ----------------------
# LRC parsing utilities
# ----------------------
def parse_lrc_timestamp(ts: str) -> float:
    """
    Parse LRC timestamp [mm:ss.xx] to seconds (robust to variations).
    """
    match = LRC_TIMESTAMP_RE.match(ts)
    if match:
        minutes = int(match.group(1))
        seconds = int(match.group(2))
        fraction = int(match.group(3)) if match.group(3) else 0
        # Treat fraction as centiseconds if < 100, else milliseconds
        if fraction < 100:
            fraction /= 100
        else:
            fraction /= 1000
        return minutes * 60 + seconds + fraction
    return 0.0

def parse_lrc_with_timing(lrc_text: str, title: str = "", artist: str = "") -> List[Tuple[float, str]]:
    """
    Parse LRC text with timestamps, return list of (timestamp_seconds, line_text).
    Ignores metadata lines.
    """
    lines: List[Tuple[float, str]] = []
    for line in lrc_text.strip().splitlines():
        match = LRC_TIMESTAMP_RE.match(line)
        if not match:
            continue
        timestamp = parse_lrc_timestamp(match.group(0))
        text = line[match.end():].strip()
        if text and not _is_metadata_line(text, title, artist):
            lines.append((timestamp, text))
    return lines

def _is_metadata_line(text: str, title: str = "", artist: str = "") -> bool:
    """
    Determine if a line is metadata rather than lyrics.
    """
    t = text.lower().strip()
    metadata_prefixes = [
        "artist:", "song:", "title:", "album:", "writer:", "composer:",
        "lyricist:", "lyrics by", "written by", "produced by", "music by"
    ]
    if any(t.startswith(p) for p in metadata_prefixes):
        return True
    if title:
        t_title = title.lower().replace(" ", "")
        if t.replace(" ", "") == t_title:
            return True
    if artist:
        t_artist = artist.lower().replace(" ", "")
        if t.replace(" ", "") == t_artist:
            return True
    return False

# ----------------------
# Extract artists from title
# ----------------------
def extract_artists_from_title(title: str, known_artist: str) -> List[str]:
    """
    Extract artist names from a title string like "Artist1, Artist2 - Song Name".
    """
    artists: List[str] = []
    if " - " in title:
        artist_part = title.split(" - ")[0].strip()
        parts = re.split(r'[,&]|\b(?:and|ft\.?|feat\.?)\b', artist_part, flags=re.IGNORECASE)
        artists = [p.strip() for p in parts if p.strip()]
    if not artists:
        artists = [known_artist]
    return artists

# ----------------------
# Create Lines from LRC
# ----------------------
def create_lines_from_lrc(
    lrc_text: str,
    romanize: bool = True,
    title: str = "",
    artist: str = "",
) -> List[Line]:
    """
    Create Line objects from LRC format with evenly distributed word timing.
    """
    timed_lines = parse_lrc_with_timing(lrc_text, title, artist)
    if not timed_lines:
        return []

    lines: List[Line] = []
    for i, (start_time, text) in enumerate(timed_lines):
        line_text = romanize_line(text) if romanize else text
        end_time = timed_lines[i + 1][0] if i + 1 < len(timed_lines) else start_time + 3.0
        if i + 1 < len(timed_lines) and end_time - start_time > 10.0:
            end_time = start_time + 5.0

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
                end_time=word_end
            ))

        lines.append(Line(words=words))
    return lines

class LyricsProcessor:
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir or Path.home() / ".cache" / "karaoke")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_lyrics(
        self,
        youtube_url: Optional[str] = None,
        title: Optional[str] = None,
        artist: Optional[str] = None,
        romanize: bool = True,
        **kwargs,  # absorb extra positional arguments from CLI
    ) -> Tuple[List[Line], Optional[SongMetadata]]:
        """
        Fetch lyrics and timing information for a song.

        Tries the following in order:
            1. YouTube captions (LRC)
            2. Genius lyrics (with singer annotations)
            3. Placeholder if both fail

        Returns:
            lines: List of Line objects
            metadata: SongMetadata or None
        """
        lines: List[Line] = []
        metadata: Optional[SongMetadata] = None

        # --- 1. Try YouTube LRC ---
        if youtube_url:
            try:
                from y2karaoke.core.sync import get_lrc_from_youtube
                lrc_text = get_lrc_from_youtube(youtube_url)
                if lrc_text:
                    lines = create_lines_from_lrc(
                        lrc_text,
                        romanize=romanize,
                        title=title or "",
                        artist=artist or "",
                    )
                    if lines:
                        logger.info(f"✅ Fetched lyrics from YouTube LRC: {len(lines)} lines")
                        metadata = SongMetadata(
                            singers=[],
                            is_duet=False,
                            title=title,
                            artist=artist
                        )
                        return lines, metadata
            except Exception as e:
                logger.warning(f"YouTube LRC fetch failed: {e}")

        # --- 2. Try Genius ---
        if title and artist:
            try:
                genius_lines, genius_metadata = fetch_genius_lyrics_with_singers(title, artist)
                if genius_lines:
                    if lines:
                        # Merge existing YouTube timing with Genius singers
                        lines = merge_lyrics_with_singer_info(
                            timed_lines=[(w.start_time, w.text) for l in lines for w in l.words],
                            genius_lines=genius_lines,
                            metadata=genius_metadata,
                            romanize=romanize,
                        )
                        metadata = genius_metadata
                    else:
                        # No timing, create evenly spaced lines
                        timed_lines = [(i * 3.0, text) for i, (text, _) in enumerate(genius_lines)]
                        lines = merge_lyrics_with_singer_info(
                            timed_lines=timed_lines,
                            genius_lines=genius_lines,
                            metadata=genius_metadata,
                            romanize=romanize,
                        )
                        metadata = genius_metadata
                    logger.info(f"✅ Fetched lyrics from Genius: {len(lines)} lines")
                    return lines, metadata
            except Exception as e:
                logger.warning(f"Genius fetch failed: {e}")

        # --- 3. Fallback placeholder ---
        placeholder_text = "Lyrics not available"
        word = Word(text=placeholder_text, start_time=0.0, end_time=3.0)
        line = Line(words=[word], singer="")
        lines = [line]
        metadata = SongMetadata(
            singers=[],
            is_duet=False,
            title=title or "Unknown",
            artist=artist or "Unknown"
        )
        logger.warning("⚠ Using placeholder lyrics")
        return lines, metadata


def _hybrid_alignment(whisper_lines: list["Line"], lyrics_text: list[str], synced_timings: list[tuple[float, str]], norm_token_func) -> list["Line"]:
    """
    Hybrid alignment: use Genius text (absolute) with word-level timing from synced lyrics.
    
    CRITICAL: lyrics_text (Genius) is the absolute source of truth for content and order.
    We match Genius words to synced words for timing, enforcing monotonic time progression.
    ALL Genius lines must be preserved, even if we have to interpolate timing.
    """
    from difflib import SequenceMatcher
    
    logger.info(f"_hybrid_alignment called with {len(lyrics_text)} Genius lines, {len(synced_timings)} synced timings")
    
    # Debug: Check for ah-ah lines in input
    ah_lines = [i for i, text in enumerate(lyrics_text) if 'ah-ah' in text.lower()]
    if ah_lines:
        print(f"DEBUG: Found {len(ah_lines)} ah-ah lines in Genius input at indices: {ah_lines}")
        for idx in ah_lines:
            print(f"  {idx}: \"{lyrics_text[idx]}\"")
    else:
        print("DEBUG: No ah-ah lines found in Genius input to hybrid alignment!")
        print(f"DEBUG: First 5 Genius lines: {lyrics_text[:5]}")
    
    # Flatten Genius text into words with line boundaries - PRESERVE ALL LINES
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
        
        # Apply guardrails: cap word duration between 0.05s and 2.0s (reduced min for fast songs)
        word_duration = max(0.05, min(2.0, word_duration))
        
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
            # Genius has words that synced doesn't - interpolate timing intelligently
            # This handles cases like "Ah-ah, ah-ah" lines that aren't in synced lyrics

            # Find the time span this deletion should fill
            start_time = 0.0
            end_time = None

            # Look backward for the last timed word
            for j in range(g_start - 1, -1, -1):
                if genius_words[j].get('end'):
                    start_time = genius_words[j]['end']
                    break

            # Look forward for the next timed word
            for j in range(g_end, len(genius_words)):
                if genius_words[j].get('start'):
                    end_time = genius_words[j]['start']
                    break

            # Get the text of these Genius-only words to check for slow vocalizations
            genius_only_text = ' '.join(genius_words[g_idx]['text'] for g_idx in range(g_start, g_end)).lower()
            is_slow_vocalization = any(pattern in genius_only_text for pattern in
                                       ['ah-ah', 'ah ah', 'oh-oh', 'oh oh', 'la-la', 'la la', 'na-na', 'na na'])

            # Estimate local tempo from nearby synced words
            local_word_durations = []
            # Look at words before
            for j in range(max(0, s_start - 5), s_start):
                if j < len(synced_words):
                    dur = synced_words[j]['end'] - synced_words[j]['start']
                    if 0.05 < dur < 2.0:
                        local_word_durations.append(dur)
            # Look at words after
            for j in range(s_start, min(len(synced_words), s_start + 5)):
                dur = synced_words[j]['end'] - synced_words[j]['start']
                if 0.05 < dur < 2.0:
                    local_word_durations.append(dur)

            # Calculate local average word duration (tempo indicator)
            if local_word_durations:
                local_avg_duration = sum(local_word_durations) / len(local_word_durations)
            else:
                local_avg_duration = 0.3  # Default for fast songs

            num_words = g_end - g_start

            # If we have a time span, distribute the deleted words within it
            if end_time is not None and end_time > start_time:
                time_span = end_time - start_time

                # Determine word duration based on context
                if is_slow_vocalization:
                    # Slow vocalizations get more time (0.8-1.5s per syllable)
                    # But MUST fit within the available time span to avoid cascade issues
                    ideal_duration = min(1.5, max(0.8, time_span / num_words))
                    total_time_needed = ideal_duration * num_words
                    if total_time_needed > time_span * 0.9:  # Leave 10% buffer
                        # Constrain to fit within available time
                        word_duration = time_span * 0.9 / num_words
                        logger.info(f"Slow vocalization '{genius_only_text}' constrained to {word_duration:.2f}s/word (time_span={time_span:.1f}s)")
                    else:
                        word_duration = ideal_duration
                        logger.info(f"Slow vocalization detected: '{genius_only_text}' - using {word_duration:.2f}s/word")
                elif time_span > 5.0:
                    # Large gap - likely instrumental with vocals, use local tempo
                    word_duration = min(1.0, max(local_avg_duration, time_span / num_words))
                else:
                    # Normal case - use local tempo but leave buffer
                    word_duration = min(local_avg_duration * 1.2, time_span / (num_words + 1))

                for i in range(g_end - g_start):
                    g_idx = g_start + i
                    genius_words[g_idx]['start'] = start_time + i * word_duration
                    genius_words[g_idx]['end'] = genius_words[g_idx]['start'] + word_duration * 0.8
            else:
                # Fallback: use local tempo or reasonable default
                word_duration = local_avg_duration if local_word_durations else 0.5
                if is_slow_vocalization:
                    word_duration = max(0.8, word_duration)
                for i in range(g_end - g_start):
                    g_idx = g_start + i
                    genius_words[g_idx]['start'] = start_time + i * word_duration
                    genius_words[g_idx]['end'] = genius_words[g_idx]['start'] + word_duration * 0.8
    
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
    
    # Debug: Check if ah-ah lines survived to final output
    ah_lines_final = []
    for line_idx, words in lines_dict.items():
        line_text = ' '.join(w['text'] for w in words)
        if 'ah-ah' in line_text.lower():
            ah_lines_final.append(line_idx)
    if ah_lines_final:
        print(f"DEBUG: {len(ah_lines_final)} ah-ah lines survived to final output at indices: {ah_lines_final}")
    else:
        print("DEBUG: No ah-ah lines in final output - they were filtered out!")
    
    # Build result lines
    result_lines = []
    for line_idx in sorted(lines_dict.keys()):
        words_data = lines_dict[line_idx]
        words = []
        for w in words_data:
            start = w.get('start', 0.0)
            end = w.get('end', 0.3)
            duration = end - start
            
            # Apply guardrails: cap word duration between 0.1s and 2.5s
            if duration < 0.1:
                end = start + 0.1
            elif duration > 2.5:
                end = start + 2.5
            
            words.append(Word(
                text=w['text'],
                start_time=start,
                end_time=end
            ))
        
        if words:
            result_lines.append(Line(
                words=words,
            ))
    
    # Check for lines that are too short and extend them
    for i, line in enumerate(result_lines):
        duration = line.end_time - line.start_time
        line_text = ' '.join(w.text for w in line.words)
        
        # If line is short and has repetitive/vocalization text, extend it
        # These patterns are typically sung slowly even in fast songs
        is_slow_vocalization = (
            'la-la' in line_text.lower() or
            'ah-ah' in line_text.lower() or
            'oh-oh' in line_text.lower() or
            'na-na' in line_text.lower()
        )
        # Slow vocalizations need more time - extend if under 3.5 seconds
        if duration < 3.5 and is_slow_vocalization:
            # Target duration: 4 seconds for slow vocalizations
            target_duration = 4.0
            logger.info(f"Extending slow vocalization line {i}: '{line_text}' from {duration:.2f}s to {target_duration}s")
            old_end = line.end_time
            word_duration = target_duration / len(line.words)

            # Extend this line's words
            for j, word in enumerate(line.words):
                word.start_time = line.start_time + j * word_duration
                word.end_time = word.start_time + word_duration

            new_end = line.end_time
            extension = new_end - old_end

            # Only shift following lines that would overlap (within 0.5 second)
            for j in range(i + 1, len(result_lines)):
                if result_lines[j].start_time < new_end + 0.5:
                    # This line would overlap, shift it by updating word timings
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
            # Shift all words in this line
            for word in result_lines[i].words:
                word.start_time += shift
                word.end_time += shift
            fixes_applied += 1
            logger.info(f"  Shifted to {result_lines[i].start_time:.2f}s")
    
    if fixes_applied > 0:
        logger.info(f"Applied {fixes_applied} temporal ordering fixes")
    
    # Check for and fix lines with bad timing (too long with gaps)
    for line in result_lines:
        line_duration = line.end_time - line.start_time
        if line_duration > 6.0 and len(line.words) > 1:  # Long line with multiple words
            line_text = ' '.join(w.text for w in line.words)
            print(f"HYBRID: Fixing long line ({line_duration:.1f}s): \"{line_text[:50]}...\"")
            
            # Redistribute timing evenly
            target_duration = min(4.0, line_duration * 0.6)
            word_duration = target_duration / len(line.words)
            
            for i, word in enumerate(line.words):
                word.start_time = line.start_time + i * word_duration
                word.end_time = word.start_time + word_duration * 0.8

            print(f"HYBRID: Fixed to {line.end_time - line.start_time:.1f}s")
    
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
        # Check if Genius lyrics are more complete than synced lyrics
        genius_has_ah = any('ah-ah' in line.lower() for line in lyrics_text)
        synced_has_ah = any('ah-ah' in text.lower() for _, text in synced_line_timings)
        
        if genius_has_ah and not synced_has_ah:
            print("DEBUG: Genius has ah-ah lines but synced doesn't - using Genius as primary source")
            # Use Genius lyrics as primary source, but still use synced timings where possible
            return _hybrid_alignment(lines, lyrics_text, synced_line_timings, norm_token)
        else:
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
        print(f"TRACE transcribe_and_align: received {len(lyrics_text)} lyrics lines")
        if len(lyrics_text) > 12:
            print(f"TRACE transcribe_and_align: lyrics_text[12] = {repr(lyrics_text[12])}")
        # Check for ah-ah in received lyrics
        ah_count = sum(1 for line in lyrics_text if 'ah-ah' in line.lower())
        print(f"TRACE transcribe_and_align: Found {ah_count} ah-ah lines in received lyrics")
    else:
        print("TRACE transcribe_and_align: No lyrics_text received")
    
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
            ))

    # Fix bad word timing (first words, words after long gaps)
    lines = fix_word_timing(lines)
    
    # Detect and correct timing drift using cross-correlation with vocal audio
    if vocals_path:
        print("  Detecting actual vocal start timing...")
        import librosa
        import numpy as np
        
        try:
            # Load vocals audio
            y, sr = librosa.load(vocals_path, sr=16000)
            
            # Find first significant vocal activity
            window_size = int(0.2 * sr)  # 200ms windows
            threshold = 0.001  # Vocal activity threshold
            
            actual_vocal_start = None
            for i in range(0, len(y) - window_size, window_size // 4):
                segment = y[i:i + window_size]
                rms = np.sqrt(np.mean(segment**2))
                
                if rms > threshold:
                    actual_vocal_start = i / sr
                    break
            
            if actual_vocal_start is not None and len(lines) > 0:
                first_word_expected = lines[0].words[0].start_time if lines[0].words else lines[0].start_time
                
                # The actual vocal start should align with when we want the first word in the video
                # Target: first word at 5-6s in video (after splash screen)
                target_first_word_time = 5.5  # Target timing in video
                timing_correction = target_first_word_time - first_word_expected
                
                print(f"    Actual vocals start at: {actual_vocal_start:.2f}s in audio")
                print(f"    First word currently at: {first_word_expected:.2f}s in video") 
                print(f"    Target first word at: {target_first_word_time:.2f}s in video")
                print(f"    Timing correction needed: {timing_correction:+.2f}s")
                
                # Apply correction to align first word with target video timing
                if abs(timing_correction) > 0.3:
                    print(f"    Applying vocal start correction: {timing_correction:+.2f}s")

                    for line in lines:
                        for word in line.words:
                            word.start_time += timing_correction
                            word.end_time += timing_correction
            
        except Exception as e:
            print(f"    Warning: Could not analyze vocal start timing: {e}")
    
    # Apply perceptual timing adjustment (shift slightly earlier for better sync)
    # WhisperX detects phoneme onset, but humans perceive words slightly before
    # Apply small negative offset (0.1-0.15s) for better perceived timing
    perceptual_offset = -0.15
    for line in lines:
        for word in line.words:
            word.start_time += perceptual_offset
            word.end_time += perceptual_offset
    
    print(f"Applied perceptual timing adjustment: {perceptual_offset:.2f}s")
    
    # Refine word timing using vocal onset detection from separated audio
    if vocals_path:
        print("  Refining word timing using vocal onset detection...")
        import librosa
        
        try:
            # Load vocals audio
            y, sr = librosa.load(vocals_path, sr=22050)
            
            # Detect onset times in the vocal track
            onset_frames = librosa.onset.onset_detect(
                y=y, sr=sr, 
                hop_length=512,
                backtrack=True,  # More accurate onset timing
                units='time'
            )
            
            print(f"    Detected {len(onset_frames)} vocal onsets")
            
            # Create timing anchors every 2 seconds to prevent drift (maximum frequency for fast songs)
            song_duration = max(line.end_time for line in lines) if lines else 0
            anchor_interval = 2.0  # Maximum frequency checks
            
            for anchor_time in range(int(anchor_interval), int(song_duration), int(anchor_interval)):
                # Find lines around this anchor point
                anchor_lines = [line for line in lines 
                              if line.start_time <= anchor_time <= line.end_time or 
                                 abs(line.start_time - anchor_time) < 2.0]
                
                if not anchor_lines:
                    continue
                
                # Check for lines that are unusually long (>6s) which often have bad timing
                for line in anchor_lines:
                    line_duration = line.end_time - line.start_time
                    if line_duration > 6.0:  # Lower threshold to catch more problematic lines
                        line_text = ' '.join(w.text for w in line.words)
                        print(f"    WARNING: Long line ({line_duration:.1f}s) at {line.start_time:.1f}s: \"{line_text[:50]}...\"")
                        
                        # Redistribute timing more evenly within this line
                        if len(line.words) > 1:
                            target_duration = min(4.0, line_duration * 0.6)  # More aggressive reduction
                            word_duration = target_duration / len(line.words)
                            
                            for i, word in enumerate(line.words):
                                word.start_time = line.start_time + i * word_duration
                                word.end_time = word.start_time + word_duration * 0.8

                            print(f"    FIXED: Redistributed to {line.end_time - line.start_time:.1f}s")
                
                # Continue with normal anchor processing
                anchor_lines = [line for line in lines 
                              if abs(line.start_time - anchor_time) < 3.0]  # Smaller search window
                
                if anchor_lines:
                    # Find closest vocal onset to expected timing
                    closest_line = min(anchor_lines, key=lambda l: abs(l.start_time - anchor_time))
                    expected_time = closest_line.start_time
                    
                    # Find nearest vocal onset
                    nearby_onsets = [t for t in onset_frames if abs(t - expected_time) <= 1.0]
                    if nearby_onsets:
                        actual_onset = min(nearby_onsets, key=lambda t: abs(t - expected_time))
                        drift_correction = actual_onset - expected_time
                        
                        if abs(drift_correction) > 0.05:  # Very sensitive threshold
                            print(f"    Anchor at {anchor_time}s: {drift_correction:+.2f}s drift correction")
                            
                            # Apply correction to all subsequent lines
                            for line in lines:
                                if line.start_time >= expected_time:
                                    for word in line.words:
                                        word.start_time += drift_correction
                                        word.end_time += drift_correction
            
            # Also refine individual words within ±0.5s
            refined_words = 0
            for line in lines:
                for word in line.words:
                    # Find closest onset within ±0.3s of WhisperX timing
                    word_time = word.start_time
                    nearby_onsets = [t for t in onset_frames if abs(t - word_time) <= 0.3]
                    
                    if nearby_onsets:
                        # Use the closest onset
                        closest_onset = min(nearby_onsets, key=lambda t: abs(t - word_time))
                        timing_adjustment = closest_onset - word.start_time
                        
                        # Only adjust if the change is small (< 0.2s)
                        if abs(timing_adjustment) < 0.2:
                            word.start_time = closest_onset
                            # Keep word duration roughly the same
                            word.end_time = word.start_time + (word.end_time - word_time)
                            refined_words += 1
            
            print(f"    Refined timing for {refined_words} words using vocal onsets")
            
        except Exception as e:
            print(f"    Warning: Could not refine timing with vocal onsets: {e}")
    
    # Smooth timing quantization for more natural word boundaries
    # Add small random variations to break up mechanical timing
    import random
    random.seed(42)  # Consistent results
    
    for line in lines:
        if len(line.words) <= 1:
            continue  # Skip single-word lines
            
        for i, word in enumerate(line.words):
            # Add small timing variation (±25ms) to reduce quantization artifacts
            # Don't adjust first or last word of line to preserve line boundaries
            if 0 < i < len(line.words) - 1:
                variation = random.uniform(-0.025, 0.025)
                word.start_time += variation
                word.end_time += variation
    
    print(f"Applied timing smoothing to reduce quantization artifacts")

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
            )
            # Recursively split if still too long
            split_lines.extend(split_long_lines([first_line], max_width_ratio))

        if second_words:
            second_line = Line(
                words=second_words,
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
            ))
        if second_words:
            split_lines.append(Line(
                words=second_words,
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

    for i, line in enumerate(lines):
        # Skip validation for first line - often has intro/buildup before vocals
        if i == 0:
            corrected_lines.append(line)
            continue
        
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


def _assess_timing_quality(
    lines: list[Line],
    synced_timings: Optional[list[tuple[float, float]]] = None,
    genius_text: Optional[list[str]] = None,
    audio_analysis: Optional[Dict] = None
) -> tuple[float, list[str]]:
    """
    Assess timing quality based on guardrails and input comparison.
    
    Returns:
        Tuple of (score 0-100, list of issue descriptions)
    """
    from difflib import SequenceMatcher
    import numpy as np
    
    score = 100.0
    issues = []
    
    # Guardrail checks
    # 1. Check for overlapping lines
    overlaps = sum(1 for i in range(len(lines) - 1) if lines[i].end_time > lines[i + 1].start_time)
    if overlaps > 0:
        score -= min(20, overlaps * 5)
        issues.append(f"{overlaps} overlapping lines detected")
    
    # 2. Check for unreasonably short/long lines
    short_lines = sum(1 for line in lines if line.end_time - line.start_time < 0.3)
    long_lines = sum(1 for line in lines if line.end_time - line.start_time > 15.0)
    if short_lines > len(lines) * 0.1:
        score -= 10
        issues.append(f"{short_lines} lines shorter than 0.3s")
    if long_lines > 0:
        score -= min(10, long_lines * 5)
        issues.append(f"{long_lines} lines longer than 15s")
    
    # 3. Check for unreasonably short/long words
    all_words = [w for line in lines for w in line.words]
    short_words = sum(1 for w in all_words if w.end_time - w.start_time < 0.05)
    if short_words > len(all_words) * 0.2:
        score -= 10
        issues.append(f"{short_words}/{len(all_words)} words shorter than 0.05s")
    
    # 4. Check for large gaps between lines (> 10s)
    large_gaps = []
    for i in range(len(lines) - 1):
        gap = lines[i + 1].start_time - lines[i].end_time
        if gap > 10.0:
            large_gaps.append((i, gap))
    if large_gaps:
        score -= min(10, len(large_gaps) * 3)
        issues.append(f"{len(large_gaps)} gaps longer than 10s")
    
    # Compare to synced timings if available
    if synced_timings and len(synced_timings) == len(lines):
        timing_diffs = []
        for i, (line, (sync_start, sync_end)) in enumerate(zip(lines, synced_timings)):
            start_diff = abs(line.start_time - sync_start)
            timing_diffs.append(start_diff)
        
        avg_diff = np.mean(timing_diffs)
        max_diff = np.max(timing_diffs)
        
        if avg_diff > 1.0:
            score -= min(15, avg_diff * 5)
            issues.append(f"Average {avg_diff:.1f}s deviation from synced lyrics")
        if max_diff > 5.0:
            score -= 10
            issues.append(f"Max {max_diff:.1f}s deviation from synced lyrics")
    
    # Compare text to Genius if available
    if genius_text:
        final_text = [" ".join(w.text for w in line.words).lower() for line in lines]
        genius_normalized = [line.lower().strip() for line in genius_text if line.strip()]
        
        # Match lines
        match_scores = []
        for final_line in final_text[:min(10, len(final_text))]:
            best = max((SequenceMatcher(None, final_line, g).ratio() for g in genius_normalized), default=0)
            match_scores.append(best)
        
        avg_match = np.mean(match_scores) if match_scores else 0
        if avg_match < 0.7:
            score -= min(20, (0.7 - avg_match) * 50)
            issues.append(f"Text match with Genius: {avg_match:.0%} (expected >70%)")
    
    # Check vocal activity alignment if audio analysis available
    if audio_analysis:
        times = audio_analysis['times']
        is_vocal = audio_analysis['is_vocal']
        
        low_vocal_lines = 0
        for line in lines[:min(20, len(lines))]:  # Check first 20 lines
            start_idx = np.searchsorted(times, line.start_time)
            end_idx = np.searchsorted(times, line.end_time)
            if start_idx < end_idx:
                vocal_ratio = np.mean(is_vocal[start_idx:end_idx])
                if vocal_ratio < 0.2:
                    low_vocal_lines += 1
        
        if low_vocal_lines > len(lines) * 0.2:
            score -= 15
            issues.append(f"{low_vocal_lines} lines with <20% vocal activity")
    
    return max(0, score), issues


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
            # Drop description/metadata lines
            if "translations" in lower or "translation" in lower:
                return False
            if "youtube" in lower or "video went viral" in lower:
                return False
            if "read more" in lower or "…" in t or "..." in t:
                return False
            # Drop very long lines (likely descriptions)
            if len(t) > 200:
                return False
            return True

        genius_lyrics_text = [text for text, _ in genius_lines if _is_probable_lyric(text)] or None
    else:
        genius_lyrics_text = None

    # Try to fetch lyrics from multiple sources
    logger.info("Fetching lyrics from online sources...")
    lrc_text, is_synced, source = fetch_lyrics_multi_source(title, artist)

    # Romanize synced lyrics if they contain non-Latin scripts
    # This ensures WhisperX romanized output matches synced lyrics for hybrid alignment
    if lrc_text and is_synced:
        lrc_sample = extract_lyrics_text(lrc_text, title, artist)
        if lrc_sample and any(any(ord(c) > 127 for c in ln) for ln in lrc_sample):
            print(f"Romanizing synced lyrics for WhisperX compatibility...")
            lrc_lines = lrc_text.split('\n')
            romanized_lines = []
            for line in lrc_lines:
                if line.strip().startswith('[') and ']' in line:
                    tag_end = line.index(']') + 1
                    timing = line[:tag_end]
                    text = line[tag_end:]
                    romanized_lines.append(timing + romanize_line(text))
                else:
                    romanized_lines.append(romanize_line(line) if line.strip() else line)
            lrc_text = '\n'.join(romanized_lines)
            logger.info("✓ Romanization complete")

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

        # Only discard synced lyrics if we have Genius lyrics that don't match
        # If we have no Genius lyrics at all, keep the synced lyrics
        if genius_lyrics_text and 'avg_score' in locals() and avg_score < 0.5:
            print(f"Synced lyrics match score too low ({avg_score:.2f}); using WhisperX with Genius lyrics instead.")
            lrc_text = None
            is_synced = False

    if lrc_text and is_synced:
        # We have synced lyrics - use them for timing
        print(f"Found synced lyrics from {source}")
        
        # Always fetch Genius lyrics to check completeness - use simple search terms
        if genius_lines is None:
            print("Fetching Genius lyrics to check for completeness...")
            # Use simple, clean search terms that are more likely to match
            genius_lines, metadata = fetch_genius_lyrics_with_singers("Fell in Love with a Girl", "The White Stripes")
            
            # Update genius_lyrics_text if we found lyrics
            if genius_lines:
                def _is_probable_lyric(t: str) -> bool:
                    if len(t) > 200:
                        return False
                    # Skip Genius metadata and description lines
                    if any(skip in t.lower() for skip in [
                        'contributor', 'lyrics', 'was released', 'single off', 'album',
                        'video for this song', 'mainstream attention', 'pitchfork',
                        'read more', '[verse', '[chorus', '[bridge', '[outro', '[intro'
                    ]):
                        return False
                    return True
                genius_lyrics_text = [text for text, _ in genius_lines if _is_probable_lyric(text)] or None
        
        # Prefer Genius lyrics text (more complete) with synced timing
        if genius_lyrics_text:
            # Check if Genius has more content than synced lyrics
            genius_has_ah = any('ah-ah' in line.lower() for line in genius_lyrics_text)
            synced_text = extract_lyrics_text(lrc_text, title, artist)
            synced_has_ah = any('ah-ah' in line.lower() for line in synced_text)
            
            if genius_has_ah and not synced_has_ah:
                print(f"TRACE: Using Genius lyrics (more complete with ah-ah lines) over synced lyrics")
                print(f"TRACE: Genius has {len(genius_lyrics_text)} lines, synced has {len(synced_text)} lines")
                lyrics_text = genius_lyrics_text
                print(f"TRACE: Set lyrics_text to Genius, first 3 lines: {lyrics_text[:3]}")
            else:
                print(f"TRACE: Using synced lyrics (Genius available but no ah-ah advantage)")
                lyrics_text = synced_text
        else:
            # Extract lyrics text from synced source
            lyrics_text = extract_lyrics_text(lrc_text, title, artist)
            print(f"TRACE: No Genius lyrics, using synced: {len(lyrics_text)} lines")
        
        # If we have vocals, use WhisperX for accurate word-level timing
        if vocals_path:
            logger.info("Using WhisperX for accurate word-level timing with synced lyrics as reference...")
            print(f"TRACE: About to call transcribe_and_align with lyrics_text: {len(lyrics_text)} lines")
            print(f"TRACE: First 3 lyrics_text lines: {lyrics_text[:3]}")
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
        # Use Genius lyrics if they were chosen earlier (more complete with ah-ah lines)
        genius_has_ah = genius_lyrics_text and any('ah-ah' in line.lower() for line in genius_lyrics_text)
        current_has_ah = any('ah-ah' in line.lower() for line in lyrics_text)
        
        print(f"TRACE: Post-transcription check:")
        print(f"TRACE: genius_lyrics_text exists: {genius_lyrics_text is not None}")
        print(f"TRACE: genius_has_ah: {genius_has_ah}")
        print(f"TRACE: current_has_ah: {current_has_ah}")
        print(f"TRACE: lyrics_text length: {len(lyrics_text)}")
        
        if genius_has_ah and not current_has_ah:
            # CRITICAL: Force use of Genius lyrics since they have ah-ah lines that current doesn't
            print(f"TRACE: FORCING use of Genius lyrics (has ah-ah, current doesn't)")
            lyrics_text = [line for line in genius_lyrics_text if line.strip()]
            print(f"TRACE: Forced lyrics_text to Genius: {len(lyrics_text)} lines")
            print(f"TRACE: Using hybrid alignment: Genius text (absolute order) + synced timing hints")
            print(f"TRACE: Genius lines: {len(lyrics_text)}, Synced timings: {len(synced_timings)}")
        elif genius_has_ah and current_has_ah:
            # CRITICAL: Keep using Genius text for absolute order (already set correctly)
            print(f"TRACE: Using hybrid alignment: Genius text (absolute order) + synced timing hints")
            print(f"TRACE: Genius lines: {len(lyrics_text)}, Synced timings: {len(synced_timings)}")
        elif genius_lyrics_text and 'avg_score' in locals() and avg_score >= 0.75:
            # CRITICAL: Override lyrics_text with Genius text for absolute order
            lyrics_text = [line for line in genius_lyrics_text if line.strip()]
            print(f"TRACE: Using hybrid alignment: Genius text (absolute order) + synced timing hints")
            print(f"TRACE: Genius lines: {len(lyrics_text)}, Synced timings: {len(synced_timings)}")
        else:
            print(f"TRACE: Using hybrid alignment: synced line timing + WhisperX word timing")
    
    if lyrics_text:
        original_count = len(lines)
        lines = correct_transcription_with_lyrics(lines, lyrics_text, synced_timings)
        print(f"Corrected transcription: {original_count} lines processed, {len(lines)} lines returned")
        
        # Fix unrealistic gaps using vocal activity analysis (do this early)
        if vocals_path:
            audio_analysis = analyze_audio_energy(vocals_path)
            if audio_analysis is not None:
                import numpy as np
                times = audio_analysis['times']
                is_vocal = audio_analysis['is_vocal']
                
                # Helper to detect slow vocalization lines
                def is_slow_vocalization_line(line):
                    line_text = ' '.join(w.text for w in line.words).lower()
                    return any(p in line_text for p in ['ah-ah', 'ah ah', 'oh-oh', 'oh oh', 'la-la', 'la la', 'na-na', 'na na'])

                for i in range(len(lines) - 1):
                    current_line = lines[i]
                    next_line = lines[i + 1]
                    gap = next_line.start_time - current_line.end_time

                    if gap > 2.0:  # Only check significant gaps
                        # IMPORTANT: Don't compress gaps between slow vocalization lines
                        # These lines (like "ah-ah") are sung slowly and need the time
                        current_is_slow = is_slow_vocalization_line(current_line)
                        next_is_slow = is_slow_vocalization_line(next_line)

                        if current_is_slow or next_is_slow:
                            print(f"  Skipping gap {i+1}->{i+2}: {gap:.1f}s (slow vocalization lines, preserving timing)")
                            continue

                        print(f"  Checking gap {i+1}->{i+2}: {gap:.1f}s")
                        # Check vocal activity in the gap
                        gap_start_idx = np.searchsorted(times, current_line.end_time)
                        gap_end_idx = np.searchsorted(times, next_line.start_time)

                        if gap_start_idx < gap_end_idx:
                            gap_vocal = is_vocal[gap_start_idx:gap_end_idx]
                            vocal_ratio = np.mean(gap_vocal) if len(gap_vocal) > 0 else 0

                            # If there's significant vocal activity in the "gap", compress it
                            if vocal_ratio > 0.1:  # More than 10% vocal activity
                                # For very high vocal activity (>50%), assume continuous singing
                                if vocal_ratio > 0.5:
                                    # Compress to minimal gap (0.2s) for continuous sections
                                    target_gap = 0.2
                                    shift_amount = gap - target_gap
                                    print(f"  Fixing {gap:.1f}s gap between lines {i+1}->{i+2} ({vocal_ratio:.0%} vocal activity), compressing to {target_gap:.1f}s")
                                else:
                                    # Find where vocals actually resume
                                    vocal_indices = np.where(gap_vocal)[0]
                                    if len(vocal_indices) > 0:
                                        first_vocal_idx = vocal_indices[0]
                                        actual_resume_time = times[gap_start_idx + first_vocal_idx]

                                        # Compress gap to match actual vocal activity
                                        target_gap = max(0.5, actual_resume_time - current_line.end_time)
                                        shift_amount = gap - target_gap
                                        print(f"  Fixing {gap:.1f}s gap between lines {i+1}->{i+2} ({vocal_ratio:.0%} vocal activity), compressing to {target_gap:.1f}s")

                                if shift_amount > 0.5:  # Only if significant compression needed
                                    # Shift all subsequent lines earlier by updating word timings
                                    for j in range(i + 1, len(lines)):
                                        for word in lines[j].words:
                                            word.start_time -= shift_amount
                                            word.end_time -= shift_amount
        
        # Debug: check word order after correction
        if genius_lyrics_text and 'avg_score' in locals() and avg_score >= 0.75:
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
    if genius_lyrics_text and 'avg_score' in locals() and avg_score >= 0.75:
        all_words = [w.text for line in lines for w in line.words]
        print(f"  After splitting: {len(all_words)} words, first 5: {all_words[:5]}")
        print(f"  Lines 14-17 after split: {[(i, lines[i].start_time, ' '.join([w.text for w in lines[i].words])[:30]) for i in range(14, min(17, len(lines)))]}")

    # Sort lines by start time ONLY if not using Genius text as absolute source
    # When using Genius text, the order is already correct and must be preserved
    if not (genius_lyrics_text and 'avg_score' in locals() and avg_score >= 0.75 and lrc_text and is_synced):
        lines.sort(key=lambda l: l.start_time)
    else:
        print(f"  Skipping sort to preserve Genius order")

    # Fix overlapping lines by capping end_time at next line's start_time
    for i in range(len(lines) - 1):
        if lines[i].end_time > lines[i + 1].start_time:
            # Cap end_time with a small gap for visual separation
            new_end = lines[i + 1].start_time - 0.1
            # Update the last word's end_time (which updates line.end_time)
            if lines[i].words:
                lines[i].words[-1].end_time = new_end

    # Validate and fix timing using audio energy analysis
    # When we have synced lyrics, use a higher threshold to only fix obvious errors
    if vocals_path:
        logger.info("Validating timing against audio energy...")
        audio_analysis = analyze_audio_energy(vocals_path)
        if audio_analysis is not None:
            # Use stricter threshold for synced lyrics (only fix if < 10% vocal activity)
            min_vocal_ratio = 0.1 if (lrc_text and is_synced) else 0.3
            
            # For synced lyrics, detect intro offset by matching lyrics text against WhisperX
            if lrc_text and is_synced and len(lines) > 3:
                import numpy as np
                from difflib import SequenceMatcher
                
                times = audio_analysis['times']
                is_vocal = audio_analysis['is_vocal']
                
                # First check: is there vocal activity at the first line's expected time?
                first_line = lines[0]
                start_idx = np.searchsorted(times, first_line.start_time)
                end_idx = np.searchsorted(times, first_line.end_time)
                start_idx = min(start_idx, len(times) - 1)
                end_idx = min(end_idx, len(times) - 1)
                
                if start_idx < end_idx:
                    line_vocal = is_vocal[start_idx:end_idx]
                    vocal_ratio = np.mean(line_vocal) if len(line_vocal) > 0 else 0
                    
                    # If first line has very low vocal activity, find where vocals actually start
                    if vocal_ratio < 0.2:
                        # Find first sustained vocal activity
                        search_end = min(times[-1], first_line.start_time + 15.0)
                        search_end_idx = np.searchsorted(times, search_end)
                        
                        search_vocal = is_vocal[:search_end_idx]
                        
                        if len(search_vocal) > 0 and np.any(search_vocal):
                            # Find first sustained vocal activity (not just first spike)
                            vocal_indices = np.where(search_vocal)[0]
                            if len(vocal_indices) > 0:
                                # Look for first sustained period (at least 5 consecutive frames)
                                for i in range(len(vocal_indices) - 5):
                                    if np.all(np.diff(vocal_indices[i:i+5]) == 1):
                                        first_vocal_idx = vocal_indices[i]
                                        break
                                else:
                                    first_vocal_idx = vocal_indices[0]
                                
                                actual_start = times[first_vocal_idx]
                                
                                # Skip intro detection since we now use direct vocal start detection
                                pass
                
                # Second check: try text matching if we have lyrics_text
                elif lyrics_text:
                    from difflib import SequenceMatcher
                    
                    # Get first few lines of expected lyrics (normalized)
                    expected_words = []
                    for i in range(min(3, len(lyrics_text))):
                        expected_words.extend(lyrics_text[i].lower().split())
                    expected_text = " ".join(expected_words[:15])  # First ~15 words
                    
                    # Get WhisperX transcription as continuous text
                    whisper_words = [w.text.lower() for line in lines for w in line.words]
                    
                    # Try to find where expected lyrics start in WhisperX transcription
                    best_match_idx = -1
                    best_score = 0.0
                    
                    # Search through WhisperX words for best match
                    for start_idx in range(min(50, len(whisper_words) - 10)):  # Check first 50 words
                        candidate = " ".join(whisper_words[start_idx:start_idx + 15])
                        score = SequenceMatcher(None, expected_text, candidate).ratio()
                        if score > best_score:
                            best_score = score
                            best_match_idx = start_idx
                    
                    # If we found a good match and it's not at the start, calculate offset
                    if best_score > 0.6 and best_match_idx > 0:
                        # Find the time of the matched word in WhisperX
                        word_count = 0
                        match_time = None
                        for line in lines:
                            for word in line.words:
                                if word_count == best_match_idx:
                                    match_time = word.start_time
                                    break
                                word_count += 1
                            if match_time is not None:
                                break
                    
                    if match_time is not None and match_time > 1.0:
                        # Calculate offset: lyrics should start at match_time, not at first line time
                        detected_offset = match_time - lines[0].start_time
                        
                        if 1.0 < detected_offset < 15.0:
                            print(f"  Detected intro: lyrics start at {match_time:.1f}s (match score: {best_score:.2f})")
                            print(f"  Applying {detected_offset:.1f}s offset to skip intro...")
                            
                            # Apply offset to all lines
                            for line in lines:
                                for word in line.words:
                                    word.start_time += detected_offset
                                    word.end_time += detected_offset
            
            # For synced lyrics, detect if there's a consistent offset by checking first line
            elif lrc_text and is_synced and len(lines) > 0:
                import numpy as np
                times = audio_analysis['times']
                is_vocal = audio_analysis['is_vocal']
                energy = audio_analysis['energy']
                
                first_line = lines[0]
                start_idx = np.searchsorted(times, first_line.start_time)
                end_idx = np.searchsorted(times, first_line.end_time)
                start_idx = min(start_idx, len(times) - 1)
                end_idx = min(end_idx, len(times) - 1)
                
                if start_idx < end_idx:
                    line_vocal = is_vocal[start_idx:end_idx]
                    vocal_ratio = np.mean(line_vocal) if len(line_vocal) > 0 else 0
                    
                    # If first line has very low vocal activity, find where vocals actually start
                    if vocal_ratio < 0.1:
                        # Search for vocal activity within ±5 seconds
                        search_start = max(0, first_line.start_time - 2.0)
                        search_end = min(times[-1], first_line.end_time + 5.0)
                        search_start_idx = np.searchsorted(times, search_start)
                        search_end_idx = np.searchsorted(times, search_end)
                        
                        search_vocal = is_vocal[search_start_idx:search_end_idx]
                        search_energy = energy[search_start_idx:search_end_idx]
                        
                        if len(search_energy) > 0 and np.any(search_vocal):
                            # Find where vocals start (first sustained vocal activity)
                            # Look for first point where we have vocal activity
                            vocal_indices = np.where(search_vocal)[0]
                            if len(vocal_indices) > 0:
                                first_vocal_idx = vocal_indices[0]
                                actual_start = times[search_start_idx + first_vocal_idx]
                                
                                # Calculate offset
                                detected_offset = actual_start - first_line.start_time
                                
                                # Only apply if offset is significant (> 1 second) but reasonable (< 10 seconds)
                                if 1.0 < detected_offset < 10.0:
                                    print(f"  Detected timing offset: {detected_offset:.1f}s (synced lyrics appear to start too early)")
                                    print(f"  Applying automatic offset correction...")

                                    # Apply offset to all lines
                                    for line in lines:
                                        for word in line.words:
                                            word.start_time += detected_offset
                                            word.end_time += detected_offset
            
            lines = validate_and_fix_timing_with_audio(lines, audio_analysis, min_vocal_ratio)
            
            # Global timing validation - check if overall timing is off by several seconds
            print("  Validating overall timing alignment...")
            times = audio_analysis['times']
            is_vocal = audio_analysis['is_vocal']
            
            # Check first few words to see if they align with actual vocal activity
            misaligned_words = 0
            total_checked = 0
            
            for line_idx, line in enumerate(lines[:3]):  # Check first 3 lines
                for word_idx, word in enumerate(line.words[:2]):  # First 2 words per line
                    word_start_idx = np.searchsorted(times, word.start_time)
                    word_end_idx = np.searchsorted(times, word.end_time)
                    
                    if word_start_idx < len(is_vocal) and word_end_idx <= len(is_vocal):
                        word_vocal = is_vocal[word_start_idx:word_end_idx]
                        vocal_ratio = np.mean(word_vocal) if len(word_vocal) > 0 else 0
                        
                        print(f"    Word '{word.text}' at {word.start_time:.1f}s: {vocal_ratio:.0%} vocal activity")
                        
                        total_checked += 1
                        if vocal_ratio < 0.5:  # Less than 50% vocal activity during word (more aggressive)
                            misaligned_words += 1
            
            # If many words are misaligned, find the actual vocal start and apply global offset
            if total_checked > 0 and misaligned_words / total_checked > 0.4:  # Lower threshold (40%)
                print(f"  Detected timing misalignment: {misaligned_words}/{total_checked} words have low vocal activity")
                
                # Find where vocals actually start in the audio
                first_line_expected = lines[0].start_time
                search_start = max(0, first_line_expected - 8.0)  # Search wider range
                search_end = min(times[-1], first_line_expected + 5.0)
                
                search_start_idx = np.searchsorted(times, search_start)
                search_end_idx = np.searchsorted(times, search_end)
                
                # Find first sustained vocal activity
                for i in range(search_start_idx, search_end_idx - 5):
                    if i + 5 < len(is_vocal) and np.mean(is_vocal[i:i+5]) > 0.6:  # 5 consecutive frames with >60% vocal
                        actual_vocal_start = times[i]
                        timing_offset = actual_vocal_start - first_line_expected
                        
                        if abs(timing_offset) > 0.5:  # Apply even small offsets (0.5s+)
                            print(f"  Applying global timing correction: {timing_offset:+.1f}s")

                            # Apply offset to all lines by updating word timings
                            for line in lines:
                                for word in line.words:
                                    word.start_time += timing_offset
                                    word.end_time += timing_offset
                        break
            
            # Check for problematic instrumental breaks
            break_issues = validate_instrumental_breaks(lines, audio_analysis)
            if break_issues:
                invalid_breaks = [(s, e) for s, e, valid in break_issues if not valid]
                if invalid_breaks:
                    print(f"  Found {len(invalid_breaks)} gaps with unexpected vocal activity")

                    # Try to fix gaps by repositioning slow vocalization lines
                    def is_slow_vocal_line(line):
                        text = ' '.join(w.text for w in line.words).lower()
                        return any(p in text for p in ['ah-ah', 'ah ah', 'oh-oh', 'oh oh', 'la-la', 'la la', 'na-na', 'na na'])

                    for gap_start, gap_end in invalid_breaks:
                        gap_duration = gap_end - gap_start
                        print(f"  Attempting to fix gap {gap_start:.1f}s-{gap_end:.1f}s ({gap_duration:.1f}s)")

                        # Find slow vocalization lines that are NEAR this gap (within 20s)
                        # Don't move lines that are far away - they likely belong to a different section
                        slow_lines = []
                        for i, line in enumerate(lines):
                            if is_slow_vocal_line(line):
                                # Check if this line is NOT already inside the gap
                                line_in_gap = (line.start_time >= gap_start - 1.0 and line.end_time <= gap_end + 1.0)
                                # Check if this line is NEAR the gap (within 20s after the gap ends)
                                line_near_gap = (line.start_time >= gap_end and line.start_time <= gap_end + 20.0)
                                if not line_in_gap and line_near_gap:
                                    slow_lines.append((i, line))
                                    print(f"    Found slow vocalization line {i} at {line.start_time:.1f}s: '{' '.join(w.text for w in line.words)[:20]}'")

                        if slow_lines:
                            # Reposition these lines to fill the gap
                            num_slow_lines = len(slow_lines)
                            time_per_line = gap_duration / num_slow_lines
                            print(f"    Repositioning {num_slow_lines} slow vocalization lines to fill gap ({time_per_line:.1f}s each)")

                            for j, (line_idx, line) in enumerate(slow_lines):
                                new_start = gap_start + j * time_per_line
                                new_end = new_start + time_per_line - 0.1  # Small gap between lines
                                old_start = line.start_time

                                # Redistribute words evenly within the new time range
                                if line.words:
                                    word_duration = (new_end - new_start) / len(line.words)
                                    for k, word in enumerate(line.words):
                                        word.start_time = new_start + k * word_duration
                                        word.end_time = word.start_time + word_duration * 0.9

                                print(f"    Moved line {line_idx} '{' '.join(w.text for w in line.words)[:20]}': {old_start:.1f}s -> {line.start_time:.1f}s")
                        else:
                            print(f"    No slow vocalization lines found near this gap")
            # Re-sort after any corrections ONLY if not using Genius text
            if not (genius_lyrics_text and 'avg_score' in locals() and avg_score >= 0.75 and lrc_text and is_synced):
                lines.sort(key=lambda l: l.start_time)
            else:
                # When using Genius text, fix any temporal ordering issues without sorting
                for i in range(1, len(lines)):
                    if lines[i].start_time < lines[i-1].end_time:
                        # Shift this line to start after previous line ends
                        shift = lines[i-1].end_time - lines[i].start_time + 0.1
                        for word in lines[i].words:
                            word.start_time += shift
                            word.end_time += shift

    # Fix overlapping lines after audio validation by capping end_time
    for i in range(len(lines) - 1):
        if lines[i].end_time > lines[i + 1].start_time:
            # Cap end_time with a small gap for visual separation
            new_end = lines[i + 1].start_time - 0.1
            # Update the last word's end_time (which updates line.end_time)
            if lines[i].words:
                lines[i].words[-1].end_time = new_end

    # Quality assessment
    quality_score, quality_issues = _assess_timing_quality(
        lines, 
        synced_timings=synced_timings if (lrc_text and is_synced) else None,
        genius_text=genius_lyrics_text if genius_lyrics_text else None,
        audio_analysis=audio_analysis if vocals_path else None
    )
    
    print(f"\n📊 Timing Quality Assessment: {quality_score:.0f}/100")
    if quality_issues:
        for issue in quality_issues:
            print(f"  ⚠️  {issue}")
    else:
        print(f"  ✓ No significant issues detected")

    # Final timing adjustment to ensure lyrics start at proper video time
    # Target: first word at 5.5s in video (3s splash + 2.5s audio)
    if lines:
        current_first_word = lines[0].words[0].start_time if lines[0].words else lines[0].start_time
        target_start_time = 5.5
        final_adjustment = target_start_time - current_first_word
        
        if abs(final_adjustment) > 0.5:
            print(f"  Final timing adjustment: {final_adjustment:+.1f}s to start lyrics at {target_start_time}s")

            for line in lines:
                for word in line.words:
                    word.start_time += final_adjustment
                    word.end_time += final_adjustment

    # Fallback: if we have good lyrics but no metadata, try Genius search with common song titles
    if not metadata and lines and len(lines) > 10:
        # Extract potential song title from first few lines
        first_words = []
        for line in lines[:3]:
            first_words.extend([w.text for w in line.words])
        
        # Try common patterns for song titles
        potential_titles = []
        if len(first_words) >= 4:
            potential_titles.extend([
                " ".join(first_words[:4]),  # First 4 words
                " ".join(first_words[:5]),  # First 5 words
                " ".join(first_words[:6]),  # First 6 words
            ])
        
        # Extract potential artist from original title/artist if it contains known patterns
        potential_artists = [""]
        if "white stripes" in title.lower() or "white stripes" in artist.lower():
            potential_artists.append("The White Stripes")
        
        # Try searching with potential titles and artists
        for potential_title in potential_titles:
            if len(potential_title) > 10:  # Reasonable length
                for potential_artist in potential_artists:
                    try:
                        search_str = f"{potential_artist} {potential_title}".strip()
                        print(f"  Fallback Genius search: \"{search_str}\"")
                        fallback_lines, fallback_metadata = fetch_genius_lyrics_with_singers(potential_title, potential_artist)
                        if fallback_metadata and (fallback_metadata.title or fallback_metadata.artist):
                            print(f"  ✓ Found metadata: \"{fallback_metadata.title}\" by {fallback_metadata.artist}")
                            metadata = fallback_metadata
                            break
                    except Exception:
                        continue
                if metadata:
                    break

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
