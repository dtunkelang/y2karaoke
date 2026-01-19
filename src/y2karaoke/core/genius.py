"""Genius lyrics fetching with singer annotations."""

from typing import List, Tuple, Optional
from ..utils.logging import get_logger
from .models import SingerID, Word, Line, SongMetadata
from .romanization import romanize_line
from .fetch import fetch_html, fetch_json
from .text_utils import normalize_text, make_slug, clean_title_for_search
from .genius_utils import (
    _is_genius_metadata,
    strip_leading_artist_from_line,
    filter_singer_only_lines,
    resolve_genius_url
)
from .lyrics_merge import merge_lyrics_with_singer_info

logger = get_logger(__name__)

# ----------------------
# HTML parsing / singer extraction
# ----------------------
def parse_genius_html(html: str, artist: str) -> Tuple[Optional[List[Tuple[str, str]]], Optional[SongMetadata]]:
    """
    Parse Genius lyrics HTML and extract lines with singer annotations.

    Returns:
        lines_with_singers: List of (line_text, singer_name)
        metadata: SongMetadata object
    """
    from bs4 import BeautifulSoup
    import re

    soup = BeautifulSoup(html, 'html.parser')

    # Extract Genius title/artist from page
    page_title = soup.find('title').get_text().strip() if soup.find('title') else ""
    genius_title = artist  # fallback
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

    # Resolve song URL using Genius-specific helper
    song_url = resolve_genius_url(title, artist)
    if not song_url:
        return None, None

    # Fetch lyrics page
    html = fetch_html(song_url, headers=headers)
    if not html:
        return None, None

    # Parse HTML and extract lines + metadata
    lines_with_singers, metadata = parse_genius_html(html, artist)
    return lines_with_singers, metadata
