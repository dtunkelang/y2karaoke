"""Render subsystem facade."""

from ...core.components.render import (
    BackgroundProcessor,
    BackgroundSegment,
    create_background_segments,
    render_karaoke_video,
)

__all__ = [
    "BackgroundProcessor",
    "BackgroundSegment",
    "create_background_segments",
    "render_karaoke_video",
]
