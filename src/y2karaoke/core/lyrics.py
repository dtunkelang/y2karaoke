"""Lyrics fetching and processing with robust LRC parsing and multilingual romanization."""

import re
import unicodedata
import time
import random
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path

import requests
from requests import Response
from bs4 import BeautifulSoup
from rapidfuzz import fuzz

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
# Data models
# ----------------------
from .models import SingerID, Word, Line, SongMetadata

# ----------------------
# Romanization
# ----------------------
from .romanization import romanize_line, romanize_multilingual

# ----------------------
# Serialization
# ----------------------
from .serialization import (
    lines_to_json as _lines_to_json,
    lines_from_json as _lines_from_json,
    metadata_to_json as _metadata_to_json,
    metadata_from_json as _metadata_from_json,
    save_lyrics_to_json,
    load_lyrics_from_json,
)

# ----------------------
# Genius lyrics fetching
# ----------------------
from .genius import (
    fetch_genius_lyrics_with_singers,
    merge_lyrics_with_singer_info,
    normalize_text,
)

# ----------------------
# Forced alignment
# ----------------------
from .forced_align import (
    forced_align,
    detect_song_start,
    alignment_quality,
    refine_with_onset_detection,
)

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
# Line splitting for display
# ----------------------
def split_long_lines(lines: List[Line], max_width_ratio: float = 0.75) -> List[Line]:
    """
    Split lines that are too long for display.

    Args:
        lines: List of Line objects
        max_width_ratio: Maximum ratio of screen width (0.75 = 75%)

    Returns:
        List of Line objects with long lines split
    """
    # Estimate max chars based on width ratio (assuming ~60 char screen width)
    max_chars = int(60 * max_width_ratio)
    result: List[Line] = []

    for line in lines:
        text = line.text
        if len(text) <= max_chars:
            result.append(line)
            continue

        # Split into roughly equal halves at word boundary
        words = line.words
        mid = len(words) // 2

        if mid == 0:
            result.append(line)
            continue

        # First half
        first_words = words[:mid]
        first_line = Line(words=first_words, singer=line.singer)

        # Second half
        second_words = words[mid:]
        second_line = Line(words=second_words, singer=line.singer)

        result.append(first_line)
        result.append(second_line)

    return result


# ----------------------
# Simplified lyrics pipeline (Phase 4 refactor)
# ----------------------
def get_lyrics_simple(
    title: str,
    artist: str,
    vocals_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    romanize: bool = True,
) -> Tuple[List[Line], Optional[SongMetadata]]:
    """
    Simplified lyrics pipeline using forced alignment.

    This function implements the clean architecture:
    1. Get canonical text + singer info from Genius
    2. Detect where vocals start in audio (offset)
    3. Use forced alignment for word-level timing
    4. Apply romanization if needed

    Note: cache_dir is accepted for API compatibility but not currently used.

    Args:
        title: Song title
        artist: Artist name
        vocals_path: Path to vocals audio (optional, for word-level timing)
        romanize: Whether to romanize non-Latin scripts

    Returns:
        Tuple of (lines, metadata)
    """
    # 1. Get canonical lyrics from Genius
    logger.info("Fetching lyrics from Genius...")
    genius_lines, metadata = fetch_genius_lyrics_with_singers(title, artist)

    if not genius_lines:
        # Fallback: placeholder
        logger.warning("No lyrics found, using placeholder")
        word = Word(text="Lyrics not available", start_time=0.0, end_time=3.0)
        line = Line(words=[word])
        return [line], SongMetadata(singers=[], is_duet=False, title=title, artist=artist)

    # Extract text lines (without singer info for alignment)
    text_lines = [text for text, _ in genius_lines if text.strip()]
    logger.info(f"Got {len(text_lines)} lines from Genius")

    # 2. If we have vocals, use forced alignment
    if vocals_path:
        logger.info("Detecting song start...")
        offset = detect_song_start(vocals_path)
        logger.info(f"Vocals start at {offset:.2f}s")

        # Get LRC timing if available (for alignment hints)
        line_timings = None
        try:
            from y2karaoke.core.sync import fetch_lyrics_multi_source
            lrc_text, is_synced, source = fetch_lyrics_multi_source(title, artist)
            if lrc_text and is_synced:
                line_timings = parse_lrc_with_timing(lrc_text, title, artist)
                logger.info(f"Got {len(line_timings)} timing hints from {source}")
        except Exception as e:
            logger.warning(f"Could not get LRC timing: {e}")

        # Forced alignment
        logger.info("Running forced alignment...")
        lines = forced_align(
            text_lines=text_lines,
            audio_path=vocals_path,
            offset=offset,
            line_timings=line_timings,
        )

        # Refine with onset detection
        if lines:
            lines = refine_with_onset_detection(lines, vocals_path)

        # Check quality
        quality = alignment_quality(lines)
        logger.info(f"Alignment quality: {quality:.2f}")

        if quality < 0.3 and line_timings:
            # Fall back to LRC timing
            logger.warning("Low alignment quality, falling back to LRC timing")
            lines = create_lines_from_lrc(lrc_text, romanize=romanize, title=title, artist=artist)
    else:
        # No audio - use estimated timing
        logger.info("No audio, using estimated timing")
        lines = []
        for i, text in enumerate(text_lines):
            line_start = i * 3.0
            word_texts = text.split()
            if not word_texts:
                continue
            word_duration = 2.5 / len(word_texts)
            words = [
                Word(
                    text=w,
                    start_time=line_start + j * word_duration,
                    end_time=line_start + (j + 1) * word_duration,
                )
                for j, w in enumerate(word_texts)
            ]
            lines.append(Line(words=words))

    # 3. Apply romanization if needed
    if romanize:
        for line in lines:
            for word in line.words:
                if any(ord(c) > 127 for c in word.text):
                    word.text = romanize_line(word.text)

    # 4. Add singer info from Genius
    if metadata and metadata.is_duet:
        # Map singer info back to lines using fuzzy matching
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


# ----------------------
# LEGACY HELPER FUNCTIONS REMOVED (Phase 5 cleanup)
# The following functions were removed as they are no longer needed:
# - _hybrid_alignment
# - _align_genius_to_whisperx_simple
# - correct_transcription_with_lyrics
# - transcribe_and_align
# - fix_word_timing
# - split_long_lines
# - _split_by_char_count
# - filter_lines_by_lyrics
# - analyze_audio_energy
# - get_vocal_activity_at_time
# - validate_and_fix_timing_with_audio
# - validate_instrumental_breaks
# - _assess_timing_quality
# ----------------------




def get_lyrics(
    title: str,
    artist: str,
    vocals_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> tuple[list[Line], Optional[SongMetadata]]:
    """
    Get lyrics with accurate word-level timing and optional singer info.

    This is a wrapper around get_lyrics_simple() for backwards compatibility.
    Uses forced alignment with Genius text as the canonical source.

    Args:
        title: Song title
        artist: Artist name
        vocals_path: Path to vocals audio (optional, for word-level timing)
        cache_dir: Cache directory (accepted for compatibility, not currently used)

    Returns:
        Tuple of (lines, metadata)
        - lines: List of Line objects with word timing (and singer info if duet)
        - metadata: SongMetadata with singer info, or None if not a duet
    """
    return get_lyrics_simple(
        title=title,
        artist=artist,
        vocals_path=vocals_path,
        cache_dir=cache_dir,
        romanize=True,
    )



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
