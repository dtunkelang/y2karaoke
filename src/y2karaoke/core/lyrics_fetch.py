"""Lyrics fetching and LRC timing helpers."""

from typing import List, Tuple, Optional
import logging
from pathlib import Path

from .models import Line, Word, SongMetadata
from .romanization import romanize_line
from .lyrics_processing import create_lines_from_lrc
from .lyrics_utils import _create_lines_from_lrc_timings
from .forced_align import detect_song_start
from .sync import fetch_lyrics_multi_source
from .lrc_utils import parse_lrc_with_timing

logger = logging.getLogger(__name__)


def _fetch_lrc_timings(title: str, artist: str) -> Optional[List[Tuple[float, str]]]:
    """Fetch LRC text from available sources and parse timings."""
    try:
        lrc_text, is_synced, source = fetch_lyrics_multi_source(title, artist)
        if lrc_text and is_synced:
            lines = parse_lrc_with_timing(lrc_text, title, artist)
            logger.info(f"✅ Got {len(lines)} LRC lines from {source}")
            return lines
        else:
            logger.info(f"No synced LRC available from {source}")
            return None
    except Exception as e:
        logger.warning(f"LRC fetch failed: {e}")
        return None


def get_lyrics_simple(
    title: str,
    artist: str,
    vocals_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    lyrics_offset: Optional[float] = None,
    romanize: bool = True,
) -> Tuple[List[Line], Optional[SongMetadata]]:
    """Simplified lyrics pipeline using Genius + LRC + vocals alignment."""
    import logging
    logger = logging.getLogger(__name__)
    from .genius import fetch_genius_lyrics_with_singers
    from .refine import refine_word_timing

    logger.info("Fetching lyrics from Genius...")
    genius_lines, metadata = fetch_genius_lyrics_with_singers(title, artist)
    if not genius_lines:
        logger.warning("No Genius lyrics found, using placeholder")
        placeholder_word = Word(text="Lyrics not available", start_time=0.0, end_time=3.0)
        return [Line(words=[placeholder_word])], SongMetadata(
            singers=[], is_duet=False, title=title, artist=artist
        )

    text_lines = [text for text, _ in genius_lines if text.strip()]
    line_timings = _fetch_lrc_timings(title, artist)

    # Apply vocal offset if available
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

    # Create Line objects
    if line_timings:
        lines = _create_lines_from_lrc_timings(line_timings, text_lines)
        if vocals_path and len(line_timings) > 1:
            lines = refine_word_timing(lines, vocals_path)
            logger.info("✅ Word-level timing refined using vocals")
    else:
        # fallback evenly spaced lines
        lrc_text = "\n".join(text_lines)
        lines = create_lines_from_lrc(lrc_text, romanize=romanize, title=title, artist=artist)

    # Romanize
    if romanize:
        for line in lines:
            for word in line.words:
                if any(ord(c) > 127 for c in word.text):
                    word.text = romanize_line(word.text)

    # Apply singer info
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
