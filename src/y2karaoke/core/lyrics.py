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


def _estimate_singing_duration(text: str, word_count: int) -> float:
    """
    Estimate how long it takes to sing a line based on text content.

    Uses character count as primary heuristic since longer words take
    longer to sing. Assumes roughly 12-15 characters per second for
    typical singing tempo.

    Args:
        text: The line text
        word_count: Number of words in the line

    Returns:
        Estimated duration in seconds
    """
    char_count = len(text.replace(" ", ""))

    # Base estimate: ~0.07 seconds per character (roughly 14 chars/sec)
    char_based = char_count * 0.07

    # Minimum based on word count (~0.25 sec per word for fast singing)
    word_based = word_count * 0.25

    # Use the larger of the two estimates
    duration = max(char_based, word_based)

    # Clamp to reasonable range
    return max(0.5, min(duration, 8.0))


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


def _fetch_lrc_text_and_timings(
    title: str,
    artist: str,
    target_duration: Optional[int] = None
) -> Tuple[Optional[str], Optional[List[Tuple[float, str]]]]:
    """Fetch raw LRC text and parsed timings from available sources.

    Args:
        title: Song title
        artist: Artist name
        target_duration: Expected track duration in seconds (for validation)

    Returns:
        Tuple of (lrc_text, parsed_timings)
    """
    try:
        if target_duration:
            # Use duration-aware fetch to find LRC matching target
            from .sync import fetch_lyrics_for_duration
            lrc_text, is_synced, source, lrc_duration = fetch_lyrics_for_duration(
                title, artist, target_duration, tolerance=20
            )
            if lrc_text and is_synced:
                lines = parse_lrc_with_timing(lrc_text, title, artist)
                logger.debug(f"Got {len(lines)} LRC lines from {source} (duration: {lrc_duration}s)")
                return lrc_text, lines
            else:
                logger.debug(f"No duration-matched LRC available")
                return None, None
        else:
            # Fallback to standard fetch without duration validation
            from .sync import fetch_lyrics_multi_source
            lrc_text, is_synced, source = fetch_lyrics_multi_source(title, artist)
            if lrc_text and is_synced:
                lines = parse_lrc_with_timing(lrc_text, title, artist)
                logger.debug(f"Got {len(lines)} LRC lines from {source}")
                return lrc_text, lines
            else:
                logger.debug(f"No synced LRC available from {source}")
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
    target_duration: Optional[int] = None,
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
        target_duration: Expected track duration in seconds (for LRC validation)

    Returns:
        Tuple of (lines, metadata)
    """
    from .genius import fetch_genius_lyrics_with_singers
    from .refine import refine_word_timing
    from .alignment import detect_song_start

    # 1. Try LRC first (preferred source), with duration validation if provided
    logger.debug(f"Fetching LRC lyrics... (target_duration={target_duration})")
    lrc_text, line_timings = _fetch_lrc_text_and_timings(title, artist, target_duration)

    # 2. Fetch Genius as fallback or for singer info
    genius_lines, metadata = None, None
    if not line_timings:
        logger.debug("No LRC found, fetching lyrics from Genius...")
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
        first_lrc_time = line_timings[0][0]
        delta = detected_vocal_start - first_lrc_time
        logger.info(
            f"Vocal timing: audio_start={detected_vocal_start:.2f}s, "
            f"LRC_start={first_lrc_time:.2f}s, delta={delta:+.2f}s"
        )
        if lyrics_offset is not None:
            offset = lyrics_offset
        else:
            # Auto-apply offset if difference is noticeable (> 0.3s) but reasonable (< 30s)
            # Songs can have long intros, so allow larger offsets
            # Negative offsets (LRC ahead of audio) are also valid
            if abs(delta) > 0.3 and abs(delta) <= 30.0:
                offset = delta
                logger.info(f"Auto-applying vocal offset: {offset:+.2f}s")
            elif abs(delta) > 30.0:
                logger.warning(f"Large timing delta ({delta:+.2f}s) - not auto-applying. Use --lyrics-offset to adjust manually.")

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
                next_line_start = line_timings[i + 1][0] if i + 1 < len(line_timings) else line_start + 5.0
                word_count = len(line.words)
                if word_count > 0:
                    # Use the full gap to next line for initial word distribution
                    # This gives the refinement step access to all onsets in the window
                    # The refinement will then use vocal end detection to trim appropriately
                    gap_to_next = next_line_start - line_start
                    # Cap at reasonable maximum to avoid extremely slow highlighting
                    line_duration = min(gap_to_next, 10.0)
                    # Ensure minimum duration
                    line_duration = max(line_duration, word_count * 0.15)

                    word_duration = (line_duration * 0.95) / word_count
                    for j, word in enumerate(line.words):
                        word.start_time = line_start + j * (line_duration / word_count)
                        word.end_time = word.start_time + word_duration
                        # Ensure last word doesn't extend past next line
                        if j == word_count - 1:
                            word.end_time = min(word.end_time, next_line_start - 0.05)
        # 5. Refine word timing using audio
        if vocals_path and len(line_timings) > 1:
            lines = refine_word_timing(lines, vocals_path)
            logger.debug("Word-level timing refined using vocals")
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

    logger.debug(f"Returning {len(lines)} lines")
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
