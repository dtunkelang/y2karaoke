"""Frame generation orchestration for video rendering."""

from __future__ import annotations
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
from PIL import ImageFont

from ...models import Line
from .frame_renderer import render_frame

if TYPE_CHECKING:
    from .backgrounds import BackgroundSegment


class FrameGenerator:
    """Stateful frame generator handling timing adjustments and caching."""

    def __init__(
        self,
        lines: list[Line],
        timing_offset: float,
        video_width: int,
        video_height: int,
        font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
        static_background: np.ndarray,
        background_segments: Optional[list[BackgroundSegment]],
        audio_duration: float,
        title: Optional[str] = None,
        artist: Optional[str] = None,
        is_duet: bool = False,
    ):
        self.lines = lines
        self.timing_offset = timing_offset
        self.video_width = video_width
        self.video_height = video_height
        self.font = font
        self.static_background = static_background
        self.background_segments = background_segments
        self.audio_duration = audio_duration
        self.title = title
        self.artist = artist
        self.is_duet = is_duet
        self.layout_cache: Dict[int, Tuple[List[str], List[float], float]] = {}

    def get_background_at_time(self, t: float) -> Optional[np.ndarray]:
        """Return the image of the segment active at time t."""
        if not self.background_segments:
            return None
        for segment in self.background_segments:
            if segment.start_time <= t <= segment.end_time:
                return segment.image
        return None

    def generate_frame(self, t: float) -> np.ndarray:
        """Generate a single frame for the video at time t."""
        adjusted_time = t - self.timing_offset

        if self.background_segments:
            bg = self.get_background_at_time(t)
            background = bg if bg is not None else self.static_background
        else:
            background = self.static_background

        return render_frame(
            self.lines,
            adjusted_time,
            self.font,
            background,
            self.title,
            self.artist,
            self.is_duet,
            self.video_width,
            self.video_height,
            self.audio_duration,
            layout_cache=self.layout_cache,
        )
