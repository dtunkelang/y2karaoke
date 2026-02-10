"""Lyrics subsystem facade.

Public orchestration entrypoints for lyrics acquisition and timing alignment.
"""

from ...core.lyrics import LyricsProcessor, get_lyrics
from ...core.lyrics_whisper import get_lyrics_simple, get_lyrics_with_quality

__all__ = [
    "LyricsProcessor",
    "get_lyrics",
    "get_lyrics_simple",
    "get_lyrics_with_quality",
]
