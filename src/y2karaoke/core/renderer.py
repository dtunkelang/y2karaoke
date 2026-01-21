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

def render_karaoke_video(
    lines: list[Line],
    audio_path: str,
    output_path: str,
    title: Optional[str] = None,
    artist: Optional[str] = None,
    timing_offset: float = 0.0,
    background_segments: Optional[list[BackgroundSegment]] = None,
    song_metadata: Optional[SongMetadata] = None,
    show_progress: bool = True,
    width: Optional[int] = None,
    height: Optional[int] = None,
    fps: Optional[int] = None,
    font_size: Optional[int] = None,
) -> str:
    """Render karaoke video using MoviePy (simple and fast).

    Args:
        lines: List of Line objects with word-level timing
        audio_path: Path to the audio file
        output_path: Path for the output video
        title: Song title for splash screen
        artist: Artist name for splash screen
        timing_offset: Timing offset in seconds
        background_segments: Optional background segments from video
        song_metadata: Optional song metadata for duet detection
        show_progress: Whether to show rendering progress bar
        width: Video width (default: from config)
        height: Video height (default: from config)
        fps: Video frame rate (default: from config)
        font_size: Font size for lyrics (default: from config)
    """
    from moviepy import AudioFileClip, VideoClip

    # Use provided settings or fall back to config defaults
    video_width = width or VIDEO_WIDTH
    video_height = height or VIDEO_HEIGHT
    video_fps = fps or FPS
    lyrics_font_size = font_size or FONT_SIZE

    logger.info("Rendering karaoke video...")
    logger.info(f"Resolution: {video_width}x{video_height}, FPS: {video_fps}, Font: {lyrics_font_size}px")
    if timing_offset != 0:
        logger.info(f"Applying timing offset: {timing_offset:+.2f}s")

    # Load audio to get duration
    audio = AudioFileClip(audio_path)
    audio_duration = audio.duration

    # Add extra time for outro screen (logo) after lyrics end
    OUTRO_DURATION = 5.0  # seconds
    last_lyrics_end = lines[-1].end_time if lines else 0
    # Extend duration to show outro after lyrics end (or after audio, whichever is later)
    duration = max(audio_duration, last_lyrics_end) + OUTRO_DURATION

    # Prepare rendering with custom font size
    font = get_font(lyrics_font_size)
    static_background = create_gradient_background(video_width, video_height)
    is_duet = song_metadata.is_duet if song_metadata else False

    # Track progress
    total_frames = int(duration * video_fps)
    frame_count = [0]
    last_percent = [-1]

    # Create frame generator with progress tracking
    def make_frame(t):
        adjusted_time = t - timing_offset

        if background_segments:
            from backgrounds import get_background_at_time
            bg = get_background_at_time(background_segments, t)
            background = bg if bg is not None else static_background
        else:
            background = static_background

        # Update progress
        if show_progress:
            frame_count[0] += 1
            percent = int(100 * frame_count[0] / total_frames) if total_frames > 0 else 0
            if percent != last_percent[0] and percent % 2 == 0:
                bar_len = 30
                filled = int(bar_len * percent / 100)
                bar = "█" * filled + "░" * (bar_len - filled)
                print(f"\r  Rendering: [{bar}] {percent}%", end="", flush=True)
                last_percent[0] = percent

        return render_frame(
            lines, adjusted_time, font, background, title, artist, is_duet,
            video_width, video_height
        )

    # Create video clip
    logger.info(f"Creating video ({duration:.1f}s at {video_fps}fps, {total_frames} frames)...")
    video = VideoClip(make_frame, duration=duration)
    video = video.with_fps(video_fps)
    video = video.with_audio(audio)

    # Write output
    logger.info(f"Writing video to {output_path}...")
    if show_progress:
        print()  # New line before progress bar
    video.write_videofile(
        output_path,
        fps=video_fps,
        codec='libx264',
        audio_codec='aac',
        threads=4,
        preset='medium',
        logger=None,  # We handle progress ourselves
    )

    if show_progress:
        print()  # New line after progress bar

    # Clean up
    audio.close()
    video.close()

    logger.info(f"Done! Output: {output_path}")
    return output_path


if __name__ == "__main__":
    # Test with dummy data
    from lyrics import Line, Word

    test_lines = [
        Line(
            words=[
                Word("Hello", 0.0, 0.5),
                Word("world", 0.5, 1.0),
                Word("this", 1.0, 1.3),
                Word("is", 1.3, 1.5),
                Word("a", 1.5, 1.7),
                Word("test", 1.7, 2.0),
            ],
        ),
        Line(
            words=[
                Word("Second", 2.0, 2.5),
                Word("line", 2.5, 3.0),
                Word("here", 3.0, 3.5),
            ],
        ),
    ]

    # Create a test frame
    font = get_font()
    bg = create_gradient_background()
    frame = render_frame(test_lines, 1.2, font, bg)

    # Save test frame
    Image.fromarray(frame).save("test_frame.png")
    logger.info("Saved test_frame.png")
