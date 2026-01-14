"""Lyrics fetching with forced alignment for accurate word-level timing."""

import re
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from pathlib import Path

import json
import os

import syncedlyrics

from ..exceptions import LyricsError
from ..utils.logging import get_logger

logger = get_logger(__name__)

@dataclass
class Word:
    """A word with timing information."""
    text: str
    start_time: float
    end_time: float

@dataclass
class Line:
    """A line of lyrics with words."""
    words: List[Word]
    start_time: float
    end_time: float

@dataclass
class SongMetadata:
    """Metadata about the song."""
    is_duet: bool = False
    singers: List[str] = None
    
    def __post_init__(self):
        if self.singers is None:
            self.singers = []

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


def romanize_line(text: str) -> str:
    """Romanize a line of lyrics, handling mixed Korean/Japanese/Chinese/English text."""
    # Apply Korean romanization first
    if contains_korean(text):
        text = romanize_korean(text)

    # Then apply Japanese romanization
    if contains_japanese(text):
        text = romanize_japanese(text)

    # Finally apply Chinese romanization (including any remaining Han characters)
    if contains_chinese(text):
        text = romanize_chinese(text)

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
class SongMetadata:
    """Metadata about singers in a song."""
    singers: list[str]  # List of singer names in order (e.g., ["Bruno Mars", "Lady Gaga"])
    is_duet: bool = False

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


def extract_lyrics_text(lrc_text: str) -> list[str]:
    """Extract plain text lines from LRC format (no timing)."""
    lines = []
    for line in lrc_text.strip().split('\n'):
        match = re.match(r'\[\d+:\d+\.\d+\]\s*(.*)', line)
        if match:
            text = match.group(1).strip()
            if text:
                lines.append(text)
    return lines


def parse_lrc_with_timing(lrc_text: str) -> list[tuple[float, str]]:
    """
    Parse LRC format to extract lines with timestamps.

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
            if text:
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


def fetch_genius_lyrics_with_singers(title: str, artist: str) -> tuple[Optional[list[tuple[str, str]]], Optional[SongMetadata]]:
    """
    Fetch lyrics from Genius with singer annotations.

    Returns:
        Tuple of (lyrics_with_singers, metadata)
        - lyrics_with_singers: List of (text, singer_name) tuples for each line
        - metadata: SongMetadata with singer info, or None if not a duet
    """
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        print("  requests/beautifulsoup4 not available for Genius scraping")
        return None, None

    from downloader import clean_title

    # Extract artists from title (for collaborations/duets)
    artists_from_title = extract_artists_from_title(title, artist)

    # Clean title and create URL slug
    clean = clean_title(title, artist)

    # Try to construct Genius URL
    # Format: artist-song-title-lyrics (lowercase, spaces to dashes)
    def make_slug(text: str) -> str:
        # Remove special characters, convert spaces to dashes
        slug = re.sub(r'[^\w\s-]', '', text.lower())
        slug = re.sub(r'\s+', '-', slug)
        return slug

    from urllib.parse import urlparse

    def add_romanized_urls(url_patterns):
        new_urls = []
        for url in url_patterns:
            if not url.endswith("-lyrics"):
                continue  # defensive: skip unexpected formats
            base = url.replace("https://genius.com/", "")
            base = base[:-len("-lyrics")]
            romanized_url = (
                "https://genius.com/"
                "Genius-romanizations-"
                f"{base}-romanized-lyrics"
            )
            new_urls.append(romanized_url)
            return url_patterns + new_urls

    # Try different URL patterns
    artist_slug = make_slug(artist)
    title_slug = make_slug(clean)

    # Handle multiple artists (from title or artist field)
    # Genius uses "and" between artists in URLs
    artist_parts = re.split(r'[,&]', artist)
    artist_parts = [p.strip() for p in artist_parts if p.strip()]

    # If only one artist in the artist field, try using artists from title
    if len(artist_parts) < 2 and len(artists_from_title) >= 2:
        artist_parts = artists_from_title

    url_patterns = []

    # Try full artist slug with "and" for duets/collaborations
    if len(artist_parts) >= 2:
        combined_slug = "-and-".join(make_slug(p) for p in artist_parts)
        url_patterns.append(f"https://genius.com/{combined_slug}-{title_slug}-lyrics")
        # Also try reversed order
        combined_slug_reversed = "-and-".join(make_slug(p) for p in reversed(artist_parts))
        url_patterns.append(f"https://genius.com/{combined_slug_reversed}-{title_slug}-lyrics")

    # Try standard patterns
    url_patterns.extend([
        f"https://genius.com/{artist_slug}-{title_slug}-lyrics",
        f"https://genius.com/{artist_slug.split('-')[0]}-{title_slug}-lyrics",
    ])

    # Also try with just the first artist
    if artist_parts:
        first_artist_slug = make_slug(artist_parts[0])
        url_patterns.append(f"https://genius.com/{first_artist_slug}-{title_slug}-lyrics")

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36'
    }

    # Use Genius API to search for the song
    # Format: lowercase, remove punctuation (but convert hyphens to spaces), single spaces
    def clean_for_search(text: str) -> str:
        text = text.lower()
        text = text.replace('-', ' ')  # Convert hyphens to spaces
        text = re.sub(r'[^\w\s]', '', text)  # Remove other punctuation
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize spaces
        return text
    
    search_query = f"{clean_for_search(artist)} {clean_for_search(title)}"
    api_url = f"https://genius.com/api/search/multi?q={search_query.replace(' ', '%20')}"
    
    song_url = None
    
    try:
        print(f"  Searching Genius: {artist} - {title}")
        response = requests.get(api_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Look through search results for song matches
        if 'response' in data and 'sections' in data['response']:
            for section in data['response']['sections']:
                if section.get('type') == 'song':
                    for hit in section.get('hits', []):
                        result = hit.get('result', {})
                        url = result.get('url', '')
                        result_title = result.get('title', '').lower()
                        result_artist = result.get('primary_artist', {}).get('name', '').lower()
                        
                        # Prefer romanized versions
                        if 'romanized' in url.lower():
                            song_url = url
                            print(f"  Found romanized: {song_url}")
                            break
                        elif not song_url:
                            song_url = url
                    
                    if song_url and 'romanized' in song_url.lower():
                        break
        
        if song_url and 'romanized' not in song_url.lower():
            print(f"  Found: {song_url}")
        
        if not song_url:
            print(f"  No Genius results found")
            return None, None
        
    except Exception as e:
        print(f"  Genius search failed: {e}")
        return None, None

    # Fetch the lyrics page
    soup = None
    try:
        response = requests.get(song_url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
    except Exception as e:
        print(f"  Error fetching lyrics: {e}")
        return None, None

    if not soup:
        return None, None

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
    metadata = SongMetadata(singers=unique_singers[:2], is_duet=is_duet) if is_duet else None

    print(f"  Found {len(lines_with_singers)} lines with singer annotations")
    if metadata:
        print(f"  Detected duet: {', '.join(metadata.singers)}")

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


def create_lines_from_lrc(lrc_text: str, romanize: bool = True) -> list[Line]:
    """
    Create Line objects from LRC format with evenly distributed word timing.

    Uses the LRC timestamps for line timing and distributes words evenly
    within each line's duration.

    Args:
        lrc_text: LRC format lyrics text
        romanize: If True, romanize non-Latin scripts (e.g., Korean to romanized)
    """
    timed_lines = parse_lrc_with_timing(lrc_text)

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


def fetch_lyrics_multi_source(title: str, artist: str) -> tuple[Optional[str], bool, str]:
    """
    Fetch lyrics from multiple sources and search variations.

    Returns:
        Tuple of (lrc_text, is_synced, source_description)
        - lrc_text: Raw LRC format text (or plain text if not synced)
        - is_synced: True if lyrics have timing info
        - source_description: Description of the source used
    """
    from downloader import clean_title

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
        try:
            lrc = syncedlyrics.search(search_term, synced_only=True)
            if lrc:
                lines = extract_lyrics_text(lrc)
                if lines and len(lines) >= 5:  # Need meaningful content
                    print(f"  Found synced lyrics ({len(lines)} lines)")
                    return lrc, True, f"synced: {search_term}"
        except Exception as e:
            print(f"  Error: {e}")

    # Try plain lyrics as fallback (no timing but better for reference)
    for search_term in unique_searches[:3]:  # Only try first few
        try:
            lrc = syncedlyrics.search(search_term, synced_only=False)
            if lrc:
                lines = extract_lyrics_text(lrc)
                if lines and len(lines) >= 5:
                    print(f"  Found plain lyrics ({len(lines)} lines)")
                    return lrc, False, f"plain: {search_term}"
        except Exception:
            pass

    return None, False, "none"


def fetch_synced_lyrics(title: str, artist: str) -> Optional[str]:
    """Fetch synced lyrics using syncedlyrics library."""
    search_term = f"{artist} {title}"
    print(f"Searching for lyrics: {search_term}")

    try:
        lrc = syncedlyrics.search(search_term, synced_only=True)
        if lrc:
            print("Found lyrics online!")
            return lrc
    except Exception as e:
        print(f"Error fetching lyrics: {e}")

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
    Hybrid alignment: use synced lyrics for line timing, WhisperX for word timing.
    
    For each synced line:
    1. Find matching WhisperX words within the line's time window
    2. Use WhisperX word timings if they look reasonable
    3. Fall back to interpolation if WhisperX timing is missing or unreasonable
    4. Ensure words are always in correct order
    """
    from difflib import SequenceMatcher
    
    # Build a flat list of all WhisperX words with timing
    whisper_words = []
    for line in whisper_lines:
        for word in line.words:
            whisper_words.append({
                "text": word.text,
                "norm": norm_token_func(word.text),
                "start": word.start_time,
                "end": word.end_time
            })
    
    # Sort WhisperX words by start time to ensure temporal order
    whisper_words.sort(key=lambda w: w["start"])
    
    result_lines = []
    
    for line_start, line_text in synced_timings:
        # Find the next line's start time (or use a default duration)
        idx = synced_timings.index((line_start, line_text))
        if idx + 1 < len(synced_timings):
            line_end = synced_timings[idx + 1][0]
        else:
            line_end = line_start + 3.0  # Default 3 second duration for last line
        
        # Get words from this line
        line_words_text = line_text.split()
        if not line_words_text:
            continue
        
        # Find WhisperX words that fall within this time window
        candidates = [w for w in whisper_words if line_start <= w["start"] < line_end]
        
        # Try to match line words to WhisperX words in order
        result_words = []
        used_indices = set()
        
        for word_idx, word_text in enumerate(line_words_text):
            word_norm = norm_token_func(word_text)
            
            # Find best matching WhisperX word that hasn't been used yet
            # and comes after the last used word (to maintain order)
            best_match = None
            best_score = 0.0
            best_cand_idx = -1
            
            for cand_idx, cand in enumerate(candidates):
                if cand_idx in used_indices:
                    continue
                    
                # Ensure temporal order: this candidate should come after previous matches
                if result_words and cand["start"] < result_words[-1].start_time:
                    continue
                
                score = SequenceMatcher(None, word_norm, cand["norm"]).ratio()
                if score > best_score:
                    best_score = score
                    best_match = cand
                    best_cand_idx = cand_idx
            
            # Use WhisperX timing if match is good, otherwise interpolate
            if best_match and best_score > 0.6:
                result_words.append(Word(
                    text=word_text,
                    start_time=best_match["start"],
                    end_time=best_match["end"]
                ))
                used_indices.add(best_cand_idx)
            else:
                # Interpolate timing within the line
                num_words = len(line_words_text)
                word_duration = (line_end - line_start) / num_words
                word_start = line_start + word_idx * word_duration
                word_end = word_start + word_duration
                
                # Ensure this word doesn't start before the previous word ends
                if result_words and word_start < result_words[-1].end_time:
                    word_start = result_words[-1].end_time
                    word_end = word_start + word_duration
                
                result_words.append(Word(
                    text=word_text,
                    start_time=word_start,
                    end_time=word_end
                ))
        
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

    # If we have synced line timings, use hybrid approach:
    # - Use synced lyrics for line start/end times
    # - Use WhisperX for word-level timing within each line
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
        # Check for Japanese romanization patterns (common particles and endings)
        japanese_patterns = ["wa ", "ga ", "wo ", "ni ", "de ", "to ", "no ", "ka ", "ne ", "yo ", 
                           "desu", "masu", "tte", "kara", "made", "nai", "tai", "tte"]
        # Check for Spanish (require multiple matches to avoid false positives)
        spanish_words = ["el ", "la ", "los ", "las ", "que ", "con ", "por ", "para ", "esta ", "como "]
        
        japanese_count = sum(1 for pattern in japanese_patterns if pattern in sample)
        spanish_count = sum(1 for word in spanish_words if word in sample)
        
        if japanese_count >= 3:
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

    print("Transcribing audio (this may take several minutes)...")
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
    print("Aligning words to audio...")
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
    from PIL import ImageFont
    
    # Use the same font size as the renderer (72)
    FONT_SIZE = 72
    
    # Load font to measure width
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", FONT_SIZE)
    except:
        try:
            # Try alternative font path
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", FONT_SIZE)
        except:
            # Fallback - use character count approximation
            return _split_by_char_count(lines, max_chars=50)
    
    VIDEO_WIDTH = 1920
    max_width = VIDEO_WIDTH * max_width_ratio
    
    split_lines = []

    for line in lines:
        # Measure line width
        line_text = " ".join(w.text for w in line.words)
        bbox = font.getbbox(line_text)
        line_width = bbox[2] - bbox[0]

        if line_width <= max_width:
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
        target_width = max_width / 2  # Split at half of max width, not half of current width

        for i, word in enumerate(words):
            word_text = word.text + " "
            word_bbox = font.getbbox(word_text)
            word_width = word_bbox[2] - word_bbox[0]
            cumulative_width += word_width
            
            if cumulative_width >= target_width:
                best_split = i + 1
                break

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
    print("Checking for singer annotations (Genius)...")
    genius_lines: Optional[list[tuple[str, str]]] = None
    metadata: Optional[SongMetadata] = None

    genius_cache_path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        genius_cache_path = os.path.join(cache_dir, "genius_cache.json")
        if os.path.exists(genius_cache_path):
            try:
                with open(genius_cache_path, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                cached_lines = cached.get("lines", [])
                genius_lines = [(item["text"], item.get("singer", "")) for item in cached_lines]
                metadata = _metadata_from_json(cached.get("metadata"))
                print("  Using cached Genius lyrics")
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
    print("Fetching lyrics from online sources...")
    lrc_text, is_synced, source = fetch_lyrics_multi_source(title, artist)

    # If we found synced lyrics, but we also have Genius lyrics, make
    # sure they look like the same song before trusting the LRC file.
    if lrc_text and is_synced and genius_lyrics_text:
        from difflib import SequenceMatcher

        lrc_plain = extract_lyrics_text(lrc_text)
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
            print("Checking if synced lyrics need romanization...")
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
                print("✓ Romanization complete")
        else:
            avg_score = 0.0

        if avg_score < 0.4:
            print("Synced lyrics appear to be for a different song; ignoring and falling back to Genius/audio.")
            lrc_text = None
            is_synced = False

    if lrc_text and is_synced:
        # We have synced lyrics - use them as reference text for WhisperX
        print(f"Found synced lyrics from {source}")
        
        # Extract lyrics text for WhisperX alignment
        lyrics_text = extract_lyrics_text(lrc_text)
        
        # If we have vocals, use WhisperX for accurate word-level timing
        if vocals_path:
            print("Using WhisperX for accurate word-level timing with synced lyrics as reference...")
            # Fall through to WhisperX transcription below
        else:
            # No vocals available, use synced lyrics timing directly
            print(f"Using synced lyrics timing from {source}")
            if genius_lines and metadata and metadata.is_duet:
                print(f"Merging with singer info from Genius")
                timed_lines = parse_lrc_with_timing(lrc_text)
                lines = merge_lyrics_with_singer_info(timed_lines, genius_lines, metadata)
            else:
                lines = create_lines_from_lrc(lrc_text)

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

    # If we only have unsynced/plain lyrics from providers and ALSO have
    # Genius lyrics, prefer Genius as the reference text and ignore the
    # plain provider lyrics.
    if lrc_text and (not is_synced) and genius_lyrics_text:
        print("Using whisperx with lyrics reference from Genius (ignoring unsynced lyrics from providers)")
        lyrics_text = [line for line in genius_lyrics_text if line.strip()]
    elif lrc_text:
        # Have plain lyrics (no timing) - use whisperx but try to match text
        print(f"Using whisperx with lyrics reference from {source}")
        lyrics_text = extract_lyrics_text(lrc_text)
    elif genius_lyrics_text:
        # No LRC from providers, but Genius lyrics are available
        print("Using whisperx with lyrics reference from Genius")
        # Filter out any empty lines just in case
        lyrics_text = [line for line in genius_lyrics_text if line.strip()]
    else:
        print("No lyrics found online, will transcribe from audio only")
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
            lines = _lines_from_json(raw.get("lines", []))
            print(f"Loaded cached Whisper transcription with {len(lines)} lines")
        except Exception:
            lines = transcribe_and_align(vocals_path, lyrics_text)
    else:
        lines = transcribe_and_align(vocals_path, lyrics_text)
        if transcript_path:
            try:
                payload = {
                    "lines": _lines_to_json(lines),
                    "language": detected_language
                }
                with open(transcript_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False)
                print(f"✓ Cached Whisper transcription")
            except Exception:
                pass

    # Cross-check and correct transcription using known lyrics (if we have them)
    # If we have synced lyrics with line timing, pass them for hybrid alignment
    synced_timings = None
    if lrc_text and is_synced:
        synced_timings = parse_lrc_with_timing(lrc_text)
        print(f"Using hybrid alignment: synced line timing + WhisperX word timing")
    
    if lyrics_text:
        original_count = len(lines)
        lines = correct_transcription_with_lyrics(lines, lyrics_text, synced_timings)
        print(f"Corrected transcription: {original_count} lines processed")
    
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
        print("Usage: python lyrics.py <title> <artist> [vocals_path]")
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
