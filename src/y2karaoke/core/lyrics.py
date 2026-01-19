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
# Create lines from LRC timing with Genius text
# ----------------------
def _create_lines_from_lrc_timings(
    lrc_timings: list[tuple[float, str]],
    genius_lines: list[str],
) -> list[Line]:
    """
    Create Line objects using LRC timing and Genius text.

    Matches Genius lines to LRC lines by fuzzy matching, then uses
    the Genius text (canonical) with LRC timing.

    Word-level timings are always generated evenly across the line,
    even if no vocals audio is available.
    """
    from difflib import SequenceMatcher

    lines: list[Line] = []
    used_genius = set()

    for i, (start_time, lrc_text) in enumerate(lrc_timings):
        # Determine end time from next line or estimate
        if i + 1 < len(lrc_timings):
            end_time = lrc_timings[i + 1][0]
            if end_time - start_time > 10.0:
                end_time = start_time + 5.0
        else:
            end_time = start_time + 3.0

        # Match Genius line (fuzzy)
        best_match = None
        best_score = 0.0
        lrc_normalized = lrc_text.lower().strip()
        for j, genius_text in enumerate(genius_lines):
            if j in used_genius:
                continue
            genius_normalized = genius_text.lower().strip()
            score = SequenceMatcher(None, lrc_normalized, genius_normalized).ratio()
            if score > best_score:
                best_score = score
                best_match = (j, genius_text)

        # Use Genius text if good match
        if best_match and best_score > 0.5:
            used_genius.add(best_match[0])
            line_text = best_match[1]
        else:
            line_text = lrc_text

        word_texts = line_text.split()
        if not word_texts:
            continue

        # Evenly distribute word timings
        line_duration = end_time - start_time
        word_count = len(word_texts)
        word_duration = (line_duration * 0.95) / word_count
        
        words = []
        for j, word_text in enumerate(word_texts):
            word_start = start_time + j * (line_duration / word_count)
            word_end = word_start + word_duration
            words.append(Word(
                text=word_text,
                start_time=word_start,
                end_time=word_end,
            ))

        # --- Option B: skip exact duplicate text from the previous line ---
        line_text_str = " ".join([w.text for w in words]).strip()
        if lines and " ".join([w.text for w in lines[-1].words]).strip() == line_text_str:
            continue  # skip this duplicate line

        lines.append(Line(words=words))

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
    lyrics_offset: Optional[float] = None,
    romanize: bool = True,
) -> Tuple[List[Line], Optional[SongMetadata]]:
    """
    Simplified lyrics pipeline using forced alignment with safe vocal offset.

    Steps:
    1. Fetch canonical text + singer info from Genius
    2. Fetch LRC line timings from syncedlyrics
    3. Detect/apply offset between audio and LRC only if reasonable
    4. Create lines from LRC + Genius text
    5. Refine word timing using audio
    6. Apply romanization if needed
    """
    # 1. Genius lyrics
    logger.info("Fetching lyrics from Genius...")
    genius_lines, metadata = fetch_genius_lyrics_with_singers(title, artist)
    if not genius_lines:
        logger.warning("No Genius lyrics found, using placeholder")
        word = Word(text="Lyrics not available", start_time=0.0, end_time=3.0)
        return [Line(words=[word])], SongMetadata(singers=[], is_duet=False, title=title, artist=artist)

    text_lines = [text for text, _ in genius_lines if text.strip()]
    logger.info(f"Got {len(text_lines)} lines from Genius")

    # 2. LRC timings
    line_timings: Optional[List[Tuple[float, str]]] = None
    try:
        from y2karaoke.core.sync import fetch_lyrics_multi_source
        lrc_text, is_synced, source = fetch_lyrics_multi_source(title, artist)
        if lrc_text and is_synced:
            line_timings = parse_lrc_with_timing(lrc_text, title, artist)
            logger.info(f"Got {len(line_timings)} line timings from {source}")
    except Exception as e:
        logger.warning(f"Could not get LRC timing: {e}")

    # 3. Detect/apply offset with vocals
    offset = 0.0
    if vocals_path:
        detected_vocal_start = detect_song_start(vocals_path)
        logger.info(f"Detected vocal start in audio: {detected_vocal_start:.2f}s")

        if lyrics_offset is not None:
            offset = lyrics_offset
            logger.info(f"Using manual lyrics offset: {offset:.2f}s")
        elif line_timings:
            first_lrc_time = line_timings[0][0]
            delta = detected_vocal_start - first_lrc_time

            # Only apply small positive offsets; never shift lyrics earlier than LRC
            if 0.0 < delta <= 5.0:
                offset = delta
                line_timings = [(ts + offset, text) for ts, text in line_timings]
                logger.info(f"Applied small positive offset: {offset:.2f}s")
            else:
                offset = 0.0
                logger.info(f"LRC timestamps trusted, no offset applied (delta={delta:.2f}s)")

        else:
            offset = detected_vocal_start
            logger.info(f"No LRC, using detected vocal start as offset: {offset:.2f}s")

        if line_timings and offset != 0.0:
            line_timings = [(ts + offset, text) for ts, text in line_timings]
            logger.info(f"Applied offset to {len(line_timings)} LRC lines")

    # 4. Create lines
    if line_timings:
        # Create lines from LRC + Genius text
        lines = _create_lines_from_lrc_timings(line_timings, text_lines)

        # Refine word-level timing only if vocals exist and LRC timestamps are reasonable
        if vocals_path and line_timings and len(line_timings) > 1:
            from .word_timing import refine_word_timing
            lines = refine_word_timing(lines, vocals_path)
            logger.info("✅ Word-level timing refined using vocals")    
    else:
        # fallback evenly spaced lines
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

    # 5. Romanize if needed
    if romanize:
        for line in lines:
            for word in line.words:
                if any(ord(c) > 127 for c in word.text):
                    word.text = romanize_line(word.text)

    # 6. Add singer info
    if metadata and metadata.is_duet:
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
    import argparse

    parser = argparse.ArgumentParser(description="Fetch and align lyrics with word timings.")
    parser.add_argument("title", type=str, help="Song title")
    parser.add_argument("artist", type=str, help="Artist name")
    parser.add_argument("vocals", type=str, nargs="?", default=None, help="Path to vocals audio (optional)")
    parser.add_argument("--no-romanize", action="store_true", help="Disable romanization")
    parser.add_argument("--lines", type=int, default=10, help="Number of lines to display")
    args = parser.parse_args()

    lines, metadata = get_lyrics_simple(
        title=args.title,
        artist=args.artist,
        vocals_path=args.vocals,
        romanize=not args.no_romanize
    )

    if metadata and metadata.is_duet:
        print(f"\nDuet detected: {', '.join(metadata.singers)}\n")

    print(f"Found {len(lines)} lines (showing first {args.lines}):\n")
    for i, line in enumerate(lines[: args.lines]):
        singer_info = f"[{line.singer}]" if line.singer else ""
        print(f"Line {i+1} {singer_info} | {line.text}")
        word_timings = ", ".join([f"{w.text}({w.start_time:.2f}-{w.end_time:.2f}s)" for w in line.words])
        print(f"  Word timings: {word_timings}\n")
