"""YouTube metadata extraction helpers."""

import re
import hashlib
from typing import Tuple, Dict


def sanitize_filename(name: str) -> str:
    """Make a filename safe for the filesystem."""
    return re.sub(r'[\\/*?:"<>|]', "", name)


def validate_youtube_url(url: str) -> str:
    """Basic validation of YouTube URL."""
    if not url or "youtube.com" not in url and "youtu.be" not in url:
        raise ValueError(f"Invalid YouTube URL: {url}")
    return url


def extract_video_id(url: str) -> str:
    """Extract YouTube video ID from URL or fallback to hash."""
    patterns = [
        r"(?:v=|/v/|youtu\.be/)([a-zA-Z0-9_-]{11})",
        r"(?:embed/)([a-zA-Z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return hashlib.md5(url.encode()).hexdigest()[:11]


def _parse_artist_title_from_video_title(
    video_title: str, uploader: str = ""
) -> Tuple[str, str]:
    """Parse artist and song title from YouTube video title."""
    if not video_title:
        return "", ""

    cleaned = video_title
    patterns_to_remove = [
        r"\s*\(Official\s*(Music\s*)?Video\)",
        r"\s*\(Official\s*Audio\)",
        r"\s*\(Official\s*Lyric\s*Video\)",
        r"\s*\(Lyric\s*Video\)",
        r"\s*\(Lyrics?\)",
        r"\s*\(Audio\)",
        r"\s*\(Visualizer\)",
        r"\s*\(Remaster(ed)?\s*\d*\)",
        r"\s*\(Live\)",
        r"\s*\[Official\s*(Music\s*)?Video\]",
        r"\s*\[Official\s*Audio\]",
        r"\s*\[4K\]",
        r"\s*\[HD\]",
        r"\s*\[HQ\]",
        r"\s*\[\d+K\]",
        r"\s*\(4K\)",
        r"\s*\(HD\)",
        r"\s*\(HQ\)",
        r"\s*\(\d+K\)",
        r"\s*【[^】]*】",
        r"\s*M/?V\s*$",
        r"\s*\(M/?V\)",
    ]
    for pattern in patterns_to_remove:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip()

    if " - " in cleaned:
        parts = cleaned.split(" - ", 1)
        if len(parts) == 2:
            return parts[0].strip(), parts[1].strip()

    by_match = re.search(r"^(.+?)\s+by\s+([^()\[\]]+)$", cleaned, re.IGNORECASE)
    if by_match:
        return by_match.group(2).strip(), by_match.group(1).strip()

    return "", ""


def _parse_metadata_from_description(
    description: str, video_title: str
) -> Tuple[str, str]:
    """Extract artist and title from video description (YouTube Music format)."""
    if not description:
        return "", ""
    lines = description.split("\n")
    for line in lines[:10]:
        line = line.strip()
        if not line or line.startswith("http") or "@" in line:
            continue
        if "·" in line:
            parts = [p.strip() for p in line.split("·")]
            if len(parts) >= 2:
                return parts[1], parts[0]
    return "", ""


def _clean_uploader_name(uploader: str) -> str:
    """Clean uploader name to use as fallback artist."""
    if not uploader:
        return ""
    artist = uploader
    suffixes = [
        "Official",
        "VEVO",
        "Records",
        "Music",
        "Channel",
        "- Topic",
        " - Topic",
        "TV",
        "Band",
        "Entertainment",
    ]
    for suffix in suffixes:
        if artist.endswith(suffix):
            artist = artist[: -len(suffix)].strip()
    prefixes = ["Official"]
    for prefix in prefixes:
        if artist.startswith(prefix):
            artist = artist[len(prefix):].strip()
    return artist


def clean_title(title: str, artist: str = "") -> str:
    """Clean video title for display."""
    cleaned = title
    if " - " in cleaned and artist:
        parts = cleaned.split(" - ", 1)
        if artist.lower() in parts[0].lower():
            cleaned = parts[1]
        elif artist.lower() in parts[1].lower():
            cleaned = parts[0]
    return cleaned.strip()


def extract_metadata_from_youtube(url: str) -> Dict[str, str]:
    """Unified metadata extraction (artist/title) for YouTube URL."""
    import yt_dlp

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "extract_flat": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        video_title = info.get("title", "")
        uploader = info.get("uploader", "")
        description = info.get("description", "")

        yt_artist = info.get("artist") or info.get("creator") or ""
        yt_track = info.get("track") or ""
        if yt_artist and yt_track:
            return {
                "artist": yt_artist,
                "title": yt_track,
                "video_id": info.get("id", extract_video_id(url)),
            }

        artist, title = _parse_artist_title_from_video_title(video_title, uploader)
        if artist and title:
            return {
                "artist": artist,
                "title": title,
                "video_id": info.get("id", extract_video_id(url)),
            }

        artist, title = _parse_metadata_from_description(description, video_title)
        if artist and title:
            return {
                "artist": artist,
                "title": title,
                "video_id": info.get("id", extract_video_id(url)),
            }

        artist = _clean_uploader_name(uploader)
        title = clean_title(video_title, artist)
        return {
            "artist": artist or "Unknown",
            "title": title or video_title,
            "video_id": info.get("id", extract_video_id(url)),
        }
