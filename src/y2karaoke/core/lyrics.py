"""Lyrics public API.

This module provides the main interface for lyrics fetching and processing:
- Fetches lyrics from Genius (canonical text + singer info)
- Gets LRC timing from syncedlyrics
- Aligns text to audio for word-level timing
- Applies romanization for non-Latin scripts
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

from .models import Word, Line, SongMetadata
from .romanization import romanize_line
from .lrc import (
    parse_lrc_timestamp,
    parse_lrc_with_timing,
    create_lines_from_lrc,
    create_lines_from_lrc_timings,
    split_long_lines,
)

logger = logging.getLogger(__name__)

__all__ = [
    # Models
    "Word",
    "Line",
    "SongMetadata",
    # Utilities
    "split_long_lines",
    "parse_lrc_timestamp",
    "parse_lrc_with_timing",
    "romanize_line",
    # API
    "LyricsProcessor",
    "get_lyrics",
    "get_lyrics_simple",
]


def _fetch_lrc_text_and_timings(title: str, artist: str) -> Tuple[Optional[str], Optional[List[Tuple[float, str]]]]:
    """Fetch raw LRC text and parsed timings from available sources."""
    try:
        from .sync import fetch_lyrics_multi_source
        lrc_text, is_synced, source = fetch_lyrics_multi_source(title, artist)
        if lrc_text and is_synced:
            lines = parse_lrc_with_timing(lrc_text, title, artist)
            logger.info(f"Got {len(lines)} LRC lines from {source}")
            return lrc_text, lines
        else:
            logger.info(f"No synced LRC available from {source}")
            return None, None
    except Exception as e:
        logger.warning(f"LRC fetch failed: {e}")
        return None, None


def get_lyrics_simple(
    title: str,
    artist: str,
    vocals_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    lyrics_offset: Optional[float] = None,
    romanize: bool = True,
) -> Tuple[List[Line], Optional[SongMetadata]]:
    """Simplified lyrics pipeline favoring LRC over Genius.

    Pipeline:
    1. Try to fetch LRC lyrics with timing (preferred source)
    2. If no LRC, fall back to Genius lyrics
    3. Detect vocal offset and align timing
    4. Create Line objects with word-level timing
    5. Refine timing using audio onset detection
    6. Apply romanization if needed

    Args:
        title: Song title
        artist: Artist name
        vocals_path: Path to vocals audio (for timing refinement)
        cache_dir: Cache directory (unused, for API compatibility)
        lyrics_offset: Manual timing offset in seconds (auto-detected if None)
        romanize: Whether to romanize non-Latin scripts

    Returns:
        Tuple of (lines, metadata)
    """
    from .genius import fetch_genius_lyrics_with_singers
    from .refine import refine_word_timing
    from .alignment import detect_song_start

    # 1. Try LRC first (preferred source)
    logger.info("Fetching LRC lyrics...")
    lrc_text, line_timings = _fetch_lrc_text_and_timings(title, artist)

    # 2. Fetch Genius as fallback or for singer info
    genius_lines, metadata = None, None
    if not line_timings:
        logger.info("No LRC found, fetching lyrics from Genius...")
        genius_lines, metadata = fetch_genius_lyrics_with_singers(title, artist)
        if not genius_lines:
            logger.warning("No lyrics found from any source, using placeholder")
            placeholder_word = Word(text="Lyrics not available", start_time=0.0, end_time=3.0)
            return [Line(words=[placeholder_word])], SongMetadata(
                singers=[], is_duet=False, title=title, artist=artist
            )
    else:
        # Still fetch Genius for singer/duet metadata only
        genius_lines, metadata = fetch_genius_lyrics_with_singers(title, artist)

    # 3. Apply vocal offset if available
    offset = 0.0
    if vocals_path and line_timings:
        detected_vocal_start = detect_song_start(vocals_path)
        logger.info(f"Detected vocal start in audio: {detected_vocal_start:.2f}s")
        if lyrics_offset is not None:
            offset = lyrics_offset
        else:
            first_lrc_time = line_timings[0][0]
            delta = detected_vocal_start - first_lrc_time
            if 0.0 < delta <= 5.0:
                offset = delta

        if offset != 0.0:
            line_timings = [(ts + offset, text) for ts, text in line_timings]

    # 4. Create Line objects
    if line_timings:
        # Use LRC text directly (preferred)
        lines = create_lines_from_lrc(lrc_text, romanize=False, title=title, artist=artist)
        # Apply timing from parsed line_timings
        for i, line in enumerate(lines):
            if i < len(line_timings):
                line_start = line_timings[i][0]
                line_end = line_timings[i + 1][0] if i + 1 < len(line_timings) else line_start + 3.0
                if line_end - line_start > 10.0:
                    line_end = line_start + 5.0
                word_count = len(line.words)
                if word_count > 0:
                    line_duration = line_end - line_start
                    word_duration = (line_duration * 0.95) / word_count
                    for j, word in enumerate(line.words):
                        word.start_time = line_start + j * (line_duration / word_count)
                        word.end_time = word.start_time + word_duration
        # 5. Refine word timing using audio
        if vocals_path and len(line_timings) > 1:
            lines = refine_word_timing(lines, vocals_path)
            logger.info("Word-level timing refined using vocals")
    else:
        # Fallback: use Genius text with evenly spaced lines
        text_lines = [text for text, _ in genius_lines if text.strip()]
        lrc_text = "\n".join(text_lines)
        lines = create_lines_from_lrc(lrc_text, romanize=romanize, title=title, artist=artist)

    # 6. Romanize
    if romanize:
        for line in lines:
            for word in line.words:
                if any(ord(c) > 127 for c in word.text):
                    word.text = romanize_line(word.text)

    # Apply singer info for duets (from Genius metadata)
    if metadata and metadata.is_duet and genius_lines:
        for i, line in enumerate(lines):
            if i < len(genius_lines):
                _, singer_name = genius_lines[i]
                singer_id = metadata.get_singer_id(singer_name)
                line.singer = singer_id
                for word in line.words:
                    word.singer = singer_id

    logger.info(f"Returning {len(lines)} lines")
    return lines, metadata


class LyricsProcessor:
    """High-level lyrics processor with caching support."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir or Path.home() / ".cache" / "karaoke")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_lyrics(
        self,
        title: Optional[str] = None,
        artist: Optional[str] = None,
        romanize: bool = True,
        **kwargs,
    ) -> Tuple[List[Line], Optional[SongMetadata]]:
        """Get lyrics for a song.

        Args:
            title: Song title
            artist: Artist name
            romanize: Whether to romanize non-Latin scripts
            **kwargs: Additional options (vocals_path, lyrics_offset)

        Returns:
            Tuple of (lines, metadata)
        """
        if not title or not artist:
            placeholder_line = Line(words=[])
            placeholder_metadata = SongMetadata(
                singers=[],
                is_duet=False,
                title=title or "Unknown",
                artist=artist or "Unknown",
            )
            return [placeholder_line], placeholder_metadata

        lines, metadata = get_lyrics_simple(
            title=title,
            artist=artist,
            vocals_path=kwargs.get("vocals_path"),
            cache_dir=str(self.cache_dir),
            lyrics_offset=kwargs.get("lyrics_offset"),
            romanize=romanize,
        )
        return lines, metadata


def get_lyrics(
    title: str,
    artist: str,
    vocals_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> Tuple[List[Line], Optional[SongMetadata]]:
    """Get lyrics for a song (convenience function).

    Args:
        title: Song title
        artist: Artist name
        vocals_path: Path to vocals audio (optional)
        cache_dir: Cache directory (optional)

    Returns:
        Tuple of (lines, metadata)
    """
    return get_lyrics_simple(
        title=title,
        artist=artist,
        vocals_path=vocals_path,
        cache_dir=cache_dir,
    )
