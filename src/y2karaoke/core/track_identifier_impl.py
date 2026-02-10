"""Compatibility facade for track identifier implementation."""

from .components.identify.implementation import (  # noqa: F401
    MusicBrainzClient,
    QueryParser,
    TrackIdentifier,
    TrackInfo,
    YouTubeSearcher,
    Y2KaraokeError,
    musicbrainzngs,
    normalize_title,
)

__all__ = [
    "TrackIdentifier",
    "TrackInfo",
    "YouTubeSearcher",
    "MusicBrainzClient",
    "QueryParser",
    "musicbrainzngs",
    "normalize_title",
    "Y2KaraokeError",
]
