"""Genius lyrics fetching with singer annotations."""

import re
from typing import List, Tuple, Optional

from ..utils.logging import get_logger
from .models import SongMetadata
from .fetch import fetch_html, fetch_json
from .text_utils import (
    make_slug,
    clean_title_for_search,
    strip_leading_artist_from_line,
    filter_singer_only_lines,
)

logger = get_logger(__name__)

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
# HTML parsing / singer extraction
# ----------------------
def _extract_genius_title_artist(soup, artist: str) -> Tuple[str, str]:
    """Extract title and artist from Genius page title."""
    page_title = soup.find("title").get_text().strip() if soup.find("title") else ""
    genius_title = artist  # fallback
    genius_artist = artist

    if page_title and "|" in page_title:
        parts = (
            page_title.split("|")[0].split("–")
            if "–" in page_title
            else page_title.split("-")
        )
        if len(parts) == 2:
            genius_artist = parts[0].strip()
            genius_title = parts[1].strip()
            if genius_title.endswith(" Lyrics"):
                genius_title = genius_title[:-7].strip()

    return genius_title, genius_artist


def _extract_lines_from_containers(
    lyrics_containers,
) -> Tuple[List[Tuple[str, str]], set]:
    """Extract lyrics lines with singer annotations from containers."""
    lines_with_singers: List[Tuple[str, str]] = []
    current_singer = ""
    singers_found: set = set()
    section_pattern = re.compile(r"\[([^\]]+)\]")

    for container in lyrics_containers:
        # Remove structural metadata
        for elem in container.find_all(
            ["div", "span", "a"],
            class_=lambda x: x
            and any(
                pattern in " ".join(x)
                for pattern in ["LyricsHeader", "SongBioPreview", "ContributorsCredit"]
            ),
        ):
            elem.decompose()

        for br in container.find_all("br"):
            br.replace_with("\n")

        text = container.get_text()
        for line in text.split("\n"):
            line = line.strip()
            if not line or re.match(r"^\d+\s*Contributor", line) or len(line) > 300:
                continue

            section_match = section_pattern.match(line)
            if section_match and ":" in section_match.group(1):
                singer_part = section_match.group(1).split(":", 1)[1].strip()
                current_singer = singer_part
                singers_found.add(singer_part)
                continue

            lines_with_singers.append((line, current_singer))

    return lines_with_singers, singers_found


def _collect_unique_singers(
    lines_with_singers: List[Tuple[str, str]], singers_found: set
) -> List[str]:
    """Collect unique singer names from lines and section headers."""
    all_singers: set = set()

    for line_text, singer in lines_with_singers:
        for part in re.split(r"\s*[,&/]\s*", singer):
            name = part.strip()
            if name and name.lower() != "both":
                all_singers.add(name)

    for header in singers_found:
        for part in re.split(r"\s*[,&/]\s*", header):
            name = part.strip()
            if name and name.lower() != "both":
                all_singers.add(name)

    return list(all_singers)


def parse_genius_html(
    html: str, artist: str
) -> Tuple[Optional[List[Tuple[str, str]]], Optional[SongMetadata]]:
    """Parse Genius lyrics HTML and extract lines with singer annotations."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")

    genius_title, genius_artist = _extract_genius_title_artist(soup, artist)

    lyrics_containers = soup.find_all("div", {"data-lyrics-container": "true"})
    if not lyrics_containers:
        return None, None

    lines_with_singers, singers_found = _extract_lines_from_containers(
        lyrics_containers
    )

    if not lines_with_singers:
        return None, None

    unique_singers = _collect_unique_singers(lines_with_singers, singers_found)
    is_duet = len(unique_singers) >= 2

    metadata = SongMetadata(
        singers=unique_singers[:2] if is_duet else [],
        is_duet=is_duet,
        title=genius_title,
        artist=genius_artist,
    )

    lines_with_singers = [
        (strip_leading_artist_from_line(text, artist), singer)
        for text, singer in lines_with_singers
    ]

    lines_with_singers = filter_singer_only_lines(
        lines_with_singers, known_singers=metadata.singers if metadata else [artist]
    )

    return lines_with_singers, metadata


# ----------------------
# Main lyrics fetching
# ----------------------
def fetch_genius_lyrics_with_singers(
    title: str, artist: str
) -> Tuple[Optional[List[Tuple[str, str]]], Optional[SongMetadata]]:
    """
    Fetch lyrics from Genius with singer annotations.

    Captures all lines, including those before any section markers.
    Strips artist prefixes and filters singer-only lines.
    Fast-fail safe for tests.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36",
        "Accept": "*/*",
    }

    cleaned_title = clean_title_for_search(
        title, TITLE_CLEANUP_PATTERNS, YOUTUBE_SUFFIXES
    )
    artist_slug = make_slug(artist)
    title_slug = make_slug(cleaned_title)

    candidate_urls = [
        f"https://genius.com/{artist_slug}-{title_slug}-lyrics",
        f"https://genius.com/{title_slug}-lyrics",
        f"https://genius.com/Genius-romanizations-{artist_slug}-{title_slug}-romanized-lyrics",
    ]

    # Try candidate URLs first (short timeout, skip translations)
    song_url = None
    for url in candidate_urls:
        html = fetch_html(url, headers=headers, timeout=5)  # 5s per request
        if html and "translation" not in url.lower():
            song_url = url
            break

    # Fallback: search API
    if not song_url:
        search_queries = [f"{artist} {cleaned_title}", f"{cleaned_title} {artist}"]
        for query in search_queries:
            api_url = f"https://genius.com/api/search/song?per_page=2&q={query.replace(' ', '%20')}"
            data = fetch_json(api_url, headers=headers, timeout=5)
            if not data:
                continue

            sections = data.get("response", {}).get("sections", [])
            for section in sections:
                if section.get("type") != "song":
                    continue
                for hit in section.get("hits", []):
                    result = hit.get("result", {})
                    url = result.get("url")
                    if (
                        url
                        and url.endswith("-lyrics")
                        and "/artists/" not in url
                        and "translation" not in url.lower()
                    ):
                        song_url = url
                        break
                if song_url:
                    break
            if song_url:
                break

    if not song_url:
        logger.warning(f"Failed to resolve Genius URL for {title} {artist}")
        return None, None

    # Fetch and parse lyrics
    html = fetch_html(song_url, headers=headers, timeout=5)
    if not html:
        logger.warning(f"Failed to fetch Genius page for {song_url}")
        return None, None

    lines_with_singers, metadata = parse_genius_html(html, artist)
    return lines_with_singers, metadata
