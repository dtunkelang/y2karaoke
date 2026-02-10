"""Compatibility facade for MusicBrainz identifier logic."""

from .components.identify.musicbrainz import MusicBrainzClient

__all__ = ["MusicBrainzClient"]
