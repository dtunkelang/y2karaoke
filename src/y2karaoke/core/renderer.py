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
        
        # Create base image
        if background_segments:
            img = self._get_background_frame(t, background_segments)
        else:
            img = self._create_gradient_background()
        
        draw = ImageDraw.Draw(img)
        
        # Show splash screen at the beginning
        if t < SPLASH_DURATION and lines and lines[0].start_time > 1.0:
            self._draw_splash_screen(draw, title, artist)
        else:
            # Draw lyrics
            self._draw_lyrics(draw, t, lines, song_metadata)
        
        # Convert to numpy array
        return np.array(img)
    
    def _create_gradient_background(self) -> Image.Image:
        """Create gradient background."""
        img = Image.new('RGB', (self.width, self.height))
        
        for y in range(self.height):
            # Linear interpolation between top and bottom colors
            ratio = y / self.height
            r = int(Colors.BG_TOP[0] * (1 - ratio) + Colors.BG_BOTTOM[0] * ratio)
            g = int(Colors.BG_TOP[1] * (1 - ratio) + Colors.BG_BOTTOM[1] * ratio)
            b = int(Colors.BG_TOP[2] * (1 - ratio) + Colors.BG_BOTTOM[2] * ratio)
            
            for x in range(self.width):
                img.putpixel((x, y), (r, g, b))
        
        return img
    
    def _get_background_frame(self, t: float, background_segments) -> Image.Image:
        """Get background frame for time t."""
        
        # Find appropriate background segment
        for segment in background_segments:
            if segment.start_time <= t <= segment.end_time:
                # Convert numpy array to PIL Image
                return Image.fromarray(segment.image)
        
        # Fallback to gradient
        return self._create_gradient_background()
    
    def _draw_splash_screen(self, draw: ImageDraw.Draw, title: str, artist: str):
        """Draw splash screen with title and artist."""
        
        # Draw title
        title_bbox = draw.textbbox((0, 0), title, font=self.font)
        title_width = title_bbox[2] - title_bbox[0]
        title_x = (self.width - title_width) // 2
        title_y = self.height // 2 - 50
        
        draw.text((title_x, title_y), title, font=self.font, fill=Colors.HIGHLIGHT)
        
        # Draw artist
        artist_text = f"by {artist}"
        artist_bbox = draw.textbbox((0, 0), artist_text, font=self.font)
        artist_width = artist_bbox[2] - artist_bbox[0]
        artist_x = (self.width - artist_width) // 2
        artist_y = title_y + 80
        
        draw.text((artist_x, artist_y), artist_text, font=self.font, fill=Colors.TEXT)
    
    def _draw_lyrics(
        self, 
        draw: ImageDraw.Draw, 
        t: float, 
        lines,
        song_metadata: Optional[Any] = None
    ):
        """Draw lyrics with highlighting."""
        
        # Find current and next lines
        current_line_idx = None
        next_line_idx = None
        
        for i, line in enumerate(lines):
            if line.start_time <= t <= line.end_time:
                current_line_idx = i
                break
            elif line.start_time > t:
                next_line_idx = i
                break
        
        # If no current line, find the next upcoming line
        if current_line_idx is None and next_line_idx is None:
            for i, line in enumerate(lines):
                if line.start_time > t:
                    next_line_idx = i
                    break
        
        # Draw current line (if any)
        if current_line_idx is not None:
            line = lines[current_line_idx]
            y_pos = self.height // 2 - LINE_SPACING // 2
            self._draw_line(draw, line, t, y_pos, is_current=True)
        
        # Draw next line (if any)
        if next_line_idx is not None:
            line = lines[next_line_idx]
            y_pos = self.height // 2 + LINE_SPACING // 2
            self._draw_line(draw, line, t, y_pos, is_current=False)
    
    def _draw_line(
        self, 
        draw: ImageDraw.Draw, 
        line, 
        t: float, 
        y_pos: int, 
        is_current: bool
    ):
        """Draw a single line of lyrics."""
        
        # Calculate total line width for centering
        total_width = 0
        word_widths = []
        
        for word in line.words:
            bbox = draw.textbbox((0, 0), word.text + " ", font=self.font)
            width = bbox[2] - bbox[0]
            word_widths.append(width)
            total_width += width
        
        # Start position (centered)
        x_pos = (self.width - total_width) // 2
        
        # Draw each word
        for i, word in enumerate(line.words):
            # Determine word color based on timing
            if is_current and word.start_time <= t <= word.end_time:
                # Currently being sung
                color = Colors.HIGHLIGHT
            elif is_current and t > word.end_time:
                # Already sung
                color = Colors.SUNG
            else:
                # Not yet sung or upcoming line
                color = Colors.TEXT
            
            # Draw word
            draw.text((x_pos, y_pos), word.text, font=self.font, fill=color)
            
            # Move to next word position
            x_pos += word_widths[i]


def get_singer_colors(singer: str, is_highlighted: bool) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """
    Get the text and highlight colors for a singer.

    Args:
        singer: Singer identifier ("singer1", "singer2", "both", or "")
        is_highlighted: Whether the word is currently being sung

    Returns:
        Tuple of (text_color, highlight_color) for this singer
    """
    if singer == "singer1":
        return (Colors.SINGER1, Colors.SINGER1_HIGHLIGHT)
    elif singer == "singer2":
        return (Colors.SINGER2, Colors.SINGER2_HIGHLIGHT)
    elif singer == "both":
        return (Colors.BOTH, Colors.BOTH_HIGHLIGHT)
    else:
        # Default colors (gold highlight, white text)
        return (Colors.TEXT, Colors.HIGHLIGHT)


def create_gradient_background(
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> np.ndarray:
    """Create a gradient background image."""
    w = width or VIDEO_WIDTH
    h = height or VIDEO_HEIGHT
    img = Image.new('RGB', (w, h))
    draw = ImageDraw.Draw(img)

    for y in range(h):
        ratio = y / h
        r = int(Colors.BG_TOP[0] * (1 - ratio) + Colors.BG_BOTTOM[0] * ratio)
        g = int(Colors.BG_TOP[1] * (1 - ratio) + Colors.BG_BOTTOM[1] * ratio)
        b = int(Colors.BG_TOP[2] * (1 - ratio) + Colors.BG_BOTTOM[2] * ratio)
        draw.line([(0, y), (w, y)], fill=(r, g, b))

    return np.array(img)


def draw_progress_bar(
    draw: ImageDraw.Draw,
    progress: float,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> None:
    """
    Draw a horizontal progress bar at the center of the screen.

    Args:
        draw: PIL ImageDraw object
        progress: Progress value from 0.0 to 1.0
        width: Video width (default: from config)
        height: Video height (default: from config)
    """
    video_width = width or VIDEO_WIDTH
    video_height = height or VIDEO_HEIGHT

    bar_width = 600
    bar_height = 12
    border_radius = 6
    y_center = video_height // 2

    x_start = (video_width - bar_width) // 2
    x_end = x_start + bar_width
    y_start = y_center - bar_height // 2
    y_end = y_center + bar_height // 2

    draw.rounded_rectangle(
        [(x_start, y_start), (x_end, y_end)],
        radius=border_radius,
        fill=Colors.PROGRESS_BG,
    )

    if progress > 0:
        fill_width = int(bar_width * min(progress, 1.0))
        if fill_width > 0:
            draw.rounded_rectangle(
                [(x_start, y_start), (x_start + fill_width, y_end)],
                radius=border_radius,
                fill=Colors.PROGRESS_FG,
            )


def draw_logo_screen(
    draw: ImageDraw.Draw,
    font: ImageFont.FreeTypeFont,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> None:
    """Draw the y2karaoke logo screen for the outro."""
    video_width = width or VIDEO_WIDTH
    video_height = height or VIDEO_HEIGHT

    logo_font = get_font(96)
    tagline_font = get_font(36)
    url_font = get_font(28)

    logo_text = "y2karaoke"
    tagline_text = "youtube to karaoke"
    url_text = "github.com/dtunkelang/y2karaoke"

    # Center the logo
    logo_bbox = logo_font.getbbox(logo_text)
    logo_width = logo_bbox[2] - logo_bbox[0]
    logo_x = (video_width - logo_width) // 2
    logo_y = video_height // 2 - 80

    # Center the tagline
    tagline_bbox = tagline_font.getbbox(tagline_text)
    tagline_width = tagline_bbox[2] - tagline_bbox[0]
    tagline_x = (video_width - tagline_width) // 2
    tagline_y = logo_y + 100

    # Center the URL
    url_bbox = url_font.getbbox(url_text)
    url_width = url_bbox[2] - url_bbox[0]
    url_x = (video_width - url_width) // 2
    url_y = tagline_y + 60

    draw.text((logo_x, logo_y), logo_text, font=logo_font, fill=Colors.HIGHLIGHT)
    draw.text((tagline_x, tagline_y), tagline_text, font=tagline_font, fill=Colors.TEXT)
    draw.text((url_x, url_y), url_text, font=url_font, fill=Colors.SUNG)


def draw_splash_screen(
    draw: ImageDraw.Draw,
    title: str,
    artist: str,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> None:
    """Draw the intro splash screen with song title and artist."""
    video_width = width or VIDEO_WIDTH
    video_height = height or VIDEO_HEIGHT

    title_font = get_font(84)
    artist_font = get_font(48)
    logo_font = get_font(32)

    # Truncate long titles
    max_title_chars = 40
    display_title = title if len(title) <= max_title_chars else title[:max_title_chars-3] + "..."

    # Center the title
    title_bbox = title_font.getbbox(display_title)
    title_width = title_bbox[2] - title_bbox[0]
    title_x = (video_width - title_width) // 2
    title_y = video_height // 2 - 80

    # Center the artist
    artist_text = f"by {artist}"
    artist_bbox = artist_font.getbbox(artist_text)
    artist_width = artist_bbox[2] - artist_bbox[0]
    artist_x = (video_width - artist_width) // 2
    artist_y = title_y + 100

    # y2karaoke branding at bottom
    logo_text = "y2karaoke"
    logo_bbox = logo_font.getbbox(logo_text)
    logo_width = logo_bbox[2] - logo_bbox[0]
    logo_x = (video_width - logo_width) // 2
    logo_y = video_height - 100

    # Draw text
    draw.text((title_x, title_y), display_title, font=title_font, fill=Colors.HIGHLIGHT)
    draw.text((artist_x, artist_y), artist_text, font=artist_font, fill=Colors.TEXT)
    draw.text((logo_x, logo_y), logo_text, font=logo_font, fill=Colors.SUNG)


def render_frame(
    lines: list[Line],
    current_time: float,
    font: ImageFont.FreeTypeFont,
    background: np.ndarray,
    title: Optional[str] = None,
    artist: Optional[str] = None,
    is_duet: bool = False,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> np.ndarray:
    """Render a single frame at the given time."""
    video_width = width or VIDEO_WIDTH
    video_height = height or VIDEO_HEIGHT
    img = Image.fromarray(background.copy())
    draw = ImageDraw.Draw(img)

    # Check if we're in an instrumental break
    show_progress_bar = False
    show_splash = False
    progress = 0.0

    # Handle intro: before first lyrics start
    if lines and current_time < lines[0].start_time:
        first_line = lines[0]
        time_until_first = first_line.start_time - current_time

        # Show splash screen for the first SPLASH_DURATION seconds OR until lyrics start (whichever is shorter)
        if current_time < SPLASH_DURATION and title and artist:
            # Don't show splash if lyrics start very early (within 1 second)
            if first_line.start_time > 1.0:
                show_splash = True
        # Then show progress bar if there's still a long intro
        elif first_line.start_time >= INSTRUMENTAL_BREAK_THRESHOLD:
            if time_until_first > LYRICS_LEAD_TIME:
                show_progress_bar = True
                # Progress bar starts after splash ends
                bar_start = min(SPLASH_DURATION, first_line.start_time - LYRICS_LEAD_TIME)
                break_end = first_line.start_time - LYRICS_LEAD_TIME
                elapsed = current_time - bar_start
                bar_duration = break_end - bar_start
                progress = elapsed / bar_duration if bar_duration > 0 else 1.0

    # Find current line index (the line we're currently on or just finished)
    current_line_idx = 0
    for i, line in enumerate(lines):
        if line.start_time <= current_time:
            current_line_idx = i

    # Handle outro: after last lyrics end, show logo screen
    if lines and current_time >= lines[-1].end_time:
        draw_logo_screen(draw, font, video_width, video_height)
        return np.array(img)

    # Handle mid-song gaps between lines
    if not show_progress_bar and current_line_idx < len(lines):
        current_line = lines[current_line_idx]
        next_line_idx = current_line_idx + 1

        # Check if there's a next line and we're past the current line
        if next_line_idx < len(lines) and current_time >= current_line.end_time:
            next_line = lines[next_line_idx]
            gap = next_line.start_time - current_line.end_time

            # If gap is large enough for instrumental break
            if gap >= INSTRUMENTAL_BREAK_THRESHOLD:
                time_until_next = next_line.start_time - current_time

                # Show progress bar if we're not within lead time of next lyrics
                if time_until_next > LYRICS_LEAD_TIME:
                    show_progress_bar = True
                    break_start = current_line.end_time
                    break_end = next_line.start_time - LYRICS_LEAD_TIME
                    break_duration = break_end - break_start
                    elapsed = current_time - break_start
                    progress = elapsed / break_duration if break_duration > 0 else 1.0

    if show_splash:
        draw_splash_screen(draw, title, artist, video_width, video_height)
        return np.array(img)

    if show_progress_bar:
        draw_progress_bar(draw, progress, video_width, video_height)
        return np.array(img)

    # Normal lyrics display - show up to 4 lines
    # Scroll in chunks of 3: when current line reaches position 3 (4th line),
    # scroll so that line becomes position 0 (1st line)
    lines_to_show = []

    # Calculate which line should be at the top of the display
    # Scroll happens every 3 lines: display_start_idx = 0, 3, 6, 9, ...
    display_start_idx = (current_line_idx // 3) * 3

    # After an instrumental break, reset display to start from the post-break line
    # Check both current line and next line for breaks
    if current_line_idx > 0:
        prev_line = lines[current_line_idx - 1]
        curr_line = lines[current_line_idx]
        gap = curr_line.start_time - prev_line.end_time
        if gap >= INSTRUMENTAL_BREAK_THRESHOLD:
            display_start_idx = current_line_idx

    # Also check if we're in the lead-time window before a post-break line
    next_line_idx = current_line_idx + 1
    if next_line_idx < len(lines):
        curr_line = lines[current_line_idx]
        next_line = lines[next_line_idx]
        gap = next_line.start_time - curr_line.end_time
        # If there's a break before next line and we're past current line's end
        if gap >= INSTRUMENTAL_BREAK_THRESHOLD and current_time >= curr_line.end_time:
            # We're in the transition - start display from next line
            display_start_idx = next_line_idx
            current_line_idx = next_line_idx  # Update so highlighting works correctly

    # Show up to 4 lines starting from display_start_idx
    # Stop early if there's an instrumental break before a line
    for i in range(4):
        line_idx = display_start_idx + i
        if line_idx < len(lines):
            # Check if there's an instrumental break before this line
            if line_idx > 0:
                prev_line = lines[line_idx - 1]
                this_line = lines[line_idx]
                gap = this_line.start_time - prev_line.end_time
                # Don't show lines after an upcoming instrumental break
                if gap >= INSTRUMENTAL_BREAK_THRESHOLD and current_time < this_line.start_time - LYRICS_LEAD_TIME:
                    break
            is_current = (line_idx == current_line_idx and current_time >= lines[line_idx].start_time)
            lines_to_show.append((lines[line_idx], is_current))

    # Calculate vertical positioning (center the lines)
    total_height = len(lines_to_show) * LINE_SPACING
    start_y = (video_height - total_height) // 2

    for idx, (line, is_current) in enumerate(lines_to_show):
        y = start_y + idx * LINE_SPACING

        # Calculate total line width for centering
        words_with_spaces = []
        for i, word in enumerate(line.words):
            words_with_spaces.append(word.text)
            if i < len(line.words) - 1:
                words_with_spaces.append(" ")

        # Measure total width
        total_width = 0
        word_widths = []
        for text in words_with_spaces:
            bbox = font.getbbox(text)
            width = bbox[2] - bbox[0]
            word_widths.append(width)
            total_width += width
        
        # Start x position for centered text
        # Note: Line splitting is handled in lyrics.py split_long_lines()
        x = (video_width - total_width) // 2

        # Draw each word with appropriate color
        word_idx = 0
        for i, text in enumerate(words_with_spaces):
            if text == " ":
                x += word_widths[i]
                continue

            word = line.words[word_idx]
            word_idx += 1

            # Determine color based on timing and singer
            # KaraFun style: once a word is highlighted, it stays highlighted
            if is_duet and word.singer:
                # Duet mode: use singer-specific colors
                text_color, highlight_color = get_singer_colors(word.singer, False)
                if is_current:
                    if current_time >= word.start_time:
                        color = highlight_color  # Highlighted (current or already sung)
                    else:
                        color = text_color  # Not yet sung (singer's color, not highlighted)
                else:
                    color = text_color  # Next line - singer's unhighlighted color
            else:
                # Non-duet mode: use default gold/white
                if is_current:
                    if current_time >= word.start_time:
                        color = Colors.HIGHLIGHT  # Highlighted (current or already sung)
                    else:
                        color = Colors.TEXT  # Not yet sung
                else:
                    color = Colors.TEXT  # Next line - all white

            draw.text((x, y), text, font=font, fill=color)
            x += word_widths[i]

    return np.array(img)


class RenderProgressBar:
    """Custom progress bar for video rendering."""

    def __init__(self, total_frames: int):
        self.total_frames = total_frames
        self.current_frame = 0
        self.last_percent = -1

    def __call__(self, gf, t):
        """Called by MoviePy for each frame."""
        self.current_frame += 1
        percent = int(100 * self.current_frame / self.total_frames)
        if percent != self.last_percent and percent % 5 == 0:
            bar_len = 30
            filled = int(bar_len * percent / 100)
            bar = "█" * filled + "░" * (bar_len - filled)
            print(f"\r  Rendering: [{bar}] {percent}%", end="", flush=True)
            self.last_percent = percent
        return gf(t)


class ProgressLogger:
    """Custom logger for MoviePy that shows a progress bar."""

    def __init__(self, total_duration: float, fps: int):
        self.total_frames = int(total_duration * fps)
        self.last_percent = -1

    def bars_callback(self, bar, attr, value, old_value=None):
        """Callback for progress bars."""
        if attr == "index":
            percent = int(100 * value / self.total_frames) if self.total_frames > 0 else 0
            if percent != self.last_percent:
                bar_len = 30
                filled = int(bar_len * percent / 100)
                bar_str = "█" * filled + "░" * (bar_len - filled)
                print(f"\r  Rendering: [{bar_str}] {percent}%", end="", flush=True)
                self.last_percent = percent

    def callback(self, **kw):
        """General callback."""
        pass


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
            start_time=0.0,
            end_time=2.0,
        ),
        Line(
            words=[
                Word("Second", 2.0, 2.5),
                Word("line", 2.5, 3.0),
                Word("here", 3.0, 3.5),
            ],
            start_time=2.0,
            end_time=3.5,
        ),
    ]

    # Create a test frame
    font = get_font()
    bg = create_gradient_background()
    frame = render_frame(test_lines, 1.2, font, bg)

    # Save test frame
    Image.fromarray(frame).save("test_frame.png")
    logger.info("Saved test_frame.png")
