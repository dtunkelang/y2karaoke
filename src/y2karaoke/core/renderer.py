"""Karaoke video renderer with KaraFun-style word highlighting."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional, List, Dict, Any
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy import AudioFileClip, VideoClip

from ..config import (
    VIDEO_WIDTH, VIDEO_HEIGHT, FPS, Colors,
    FONT_SIZE, LINE_SPACING, SPLASH_DURATION,
    INSTRUMENTAL_BREAK_THRESHOLD, LYRICS_LEAD_TIME
)
from ..exceptions import RenderError
from ..utils.logging import get_logger
from ..utils.fonts import get_font
from .backgrounds_static import (
    create_gradient_background,
    draw_logo_screen,
    draw_splash_screen
)
from .progress import draw_progress_bar, RenderProgressBar, ProgressLogger
from .lyrics_renderer import draw_lyrics_frame, get_singer_colors
from .frame_renderer import render_frame

logger = get_logger(__name__)

if TYPE_CHECKING:
    from .backgrounds import BackgroundSegment

class VideoRenderer:
    """Render karaoke videos with word-by-word highlighting."""

    def __init__(self):
        self.width = VIDEO_WIDTH
        self.height = VIDEO_HEIGHT
        self.fps = FPS
        self.font = get_font()  # Use shared font utility
    
    def render_karaoke_video(
        self,
        lines,
        audio_path: str,
        output_path: str,
        title: str,
        artist: str,
        timing_offset: float = 0.0,
        background_segments: Optional[List] = None,
        song_metadata: Optional[Any] = None,
    ):
        """Render the complete karaoke video."""
        
        logger.info(f"Rendering karaoke video: {title} by {artist}")
        
        try:
            # Import moviepy here to avoid issues if not available
            from moviepy import AudioFileClip, VideoClip
            
            # Load audio to get duration
            audio_clip = AudioFileClip(audio_path)
            duration = audio_clip.duration
            
            # Create video clip
            def make_frame(t):
                return self._render_frame(
                    t + timing_offset, 
                    lines, 
                    title, 
                    artist, 
                    duration,
                    background_segments,
                    song_metadata
                )
            
            video_clip = VideoClip(make_frame, duration=duration)
            video_clip = video_clip.with_audio(audio_clip)
            
            # Write video
            logger.info(f"Writing video to {output_path}")
            video_clip.write_videofile(
                output_path,
                fps=self.fps,
                codec='libx264',
                audio_codec='aac',
                threads=4,
                preset='ultrafast',  # Fastest encoding
                temp_audiofile='temp-audio.m4a',
                remove_temp=True,
                logger=None,
            )
            
            # Cleanup
            video_clip.close()
            audio_clip.close()
            
            logger.info("✅ Video rendering completed")
            
        except ImportError:
            raise RenderError("MoviePy not available for video rendering")
        except Exception as e:
            raise RenderError(f"Video rendering failed: {e}")
    
    def _render_frame(
        self, 
        t: float, 
        lines, 
        title: str, 
        artist: str,
        duration: float,
        background_segments: Optional[List] = None,
        song_metadata: Optional[Any] = None
    ) -> np.ndarray:
        """Render a single frame at time t."""

        # Determine base image
        if background_segments:
            img = self._get_background_frame(t, background_segments)
        else:
            img = create_gradient_background(self.width, self.height)

        draw = ImageDraw.Draw(img)

        # Show splash screen at the beginning
        if t < SPLASH_DURATION:
            draw_splash_screen(draw, title, artist, self.width, self.height)
        else:
            # Draw lyrics frame
            is_duet = song_metadata.is_duet if song_metadata else False
            draw_lyrics_frame(
                draw,
                t,
                lines,
                self.font,
                self.height,
                is_duet=is_duet,
                song_metadata=song_metadata
            )

        return np.array(img)
    
    def _get_background_frame(self, t: float, background_segments) -> Image.Image:
        """Get the background frame for time t.

        Falls back to a gradient if no segment matches.
        """
        for segment in background_segments:
            if segment.start_time <= t <= segment.end_time:
                # Convert numpy array to PIL Image
                return Image.fromarray(segment.image)

        # Fallback to gradient background
        return create_gradient_background(self.width, self.height)