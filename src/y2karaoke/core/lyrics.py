"""Lyrics public API.

This module acts as a façade that exposes all lyric-related functionality
while allowing internal implementation to be split across files.
"""

from pathlib import Path
from typing import List, Optional, Tuple
from .lyrics_processing import split_long_lines

# ------------------------------
# Public domain models
# ------------------------------

from .models import Word, Line, SongMetadata

# ------------------------------
# Public utility functions
# ------------------------------

from .lyrics_processing import split_long_lines
from .lrc_utils import parse_lrc_timestamp, parse_lrc_with_timing
from .romanization import romanize_line

# ------------------------------
# Internal orchestration
# ------------------------------

from .lyrics_fetch import get_lyrics_simple


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
]


class LyricsProcessor:
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
            cache_dir=self.cache_dir,
            lyrics_offset=kwargs.get("lyrics_offset"),
            romanize=romanize,
        )
        return lines, metadata


def get_lyrics(
    title: str,
    artist: str,
    vocals_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
):
    return get_lyrics_simple(
        title=title,
        artist=artist,
        vocals_path=vocals_path,
        cache_dir=cache_dir,
    )
