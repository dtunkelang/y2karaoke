"""Compatibility facade for YouTube track-identification helper rules."""

from .components.identify.youtube_rules import (
    extract_youtube_candidates,
    is_likely_non_studio,
    is_preferred_audio_title,
    query_wants_non_studio,
    youtube_duration_tolerance,
)

__all__ = [
    "is_likely_non_studio",
    "is_preferred_audio_title",
    "query_wants_non_studio",
    "youtube_duration_tolerance",
    "extract_youtube_candidates",
]
