"""Compatibility facade for track identification pipeline."""

from .track_identifier_impl import (
    MusicBrainzClient,
    QueryParser,
    TrackIdentifier,
    YouTubeSearcher,
    TrackInfo,
    musicbrainzngs,
    normalize_title,
    Y2KaraokeError,
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
