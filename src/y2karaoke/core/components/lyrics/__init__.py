"""Lyrics component facade."""

__all__ = []

try:
    from .api import LyricsProcessor, get_lyrics

    __all__ += ["LyricsProcessor", "get_lyrics"]
except ImportError:
    pass

try:
    from .lyrics_whisper import get_lyrics_simple, get_lyrics_with_quality

    __all__ += ["get_lyrics_simple", "get_lyrics_with_quality"]
except ImportError:
    pass
