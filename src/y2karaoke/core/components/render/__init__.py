"""Render component facade."""

from .backgrounds import (
    BackgroundProcessor,
    BackgroundSegment,
    create_background_segments,
)
from .video_writer import render_karaoke_video

__all__ = [
    "BackgroundProcessor",
    "BackgroundSegment",
    "create_background_segments",
    "render_karaoke_video",
]
