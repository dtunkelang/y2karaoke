"""Karaoke video renderer with KaraFun-style word highlighting."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy import AudioFileClip, VideoClip

from lyrics import Line, Word

if TYPE_CHECKING:
    from backgrounds import BackgroundSegment


# Video settings
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080
FPS = 30

# Colors (RGB)
BG_COLOR_TOP = (20, 20, 40)      # Dark blue
BG_COLOR_BOTTOM = (40, 20, 60)   # Dark purple
TEXT_COLOR = (255, 255, 255)     # White
HIGHLIGHT_COLOR = (255, 215, 0)  # Gold
SUNG_COLOR = (180, 180, 180)    # Gray (already sung)
PROGRESS_BAR_BG = (60, 60, 80)  # Progress bar background
PROGRESS_BAR_FG = (255, 215, 0) # Progress bar fill (gold)

# Font settings
FONT_SIZE = 72
LINE_SPACING = 100

# Instrumental break settings
INSTRUMENTAL_BREAK_THRESHOLD = 5.0  # seconds - minimum gap to show progress bar
LYRICS_LEAD_TIME = 1.0              # seconds - show lyrics before they start

# Splash screen settings
SPLASH_DURATION = 4.0               # seconds - how long to show splash before fading


def get_font(size: int = FONT_SIZE) -> ImageFont.FreeTypeFont:
    """Get a suitable font for rendering."""
    # Try common fonts
    font_paths = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSDisplay.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]

    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue

    # Fall back to default
    return ImageFont.load_default()


def create_gradient_background() -> np.ndarray:
    """Create a gradient background image."""
    img = Image.new('RGB', (VIDEO_WIDTH, VIDEO_HEIGHT))
    draw = ImageDraw.Draw(img)

    for y in range(VIDEO_HEIGHT):
        ratio = y / VIDEO_HEIGHT
        r = int(BG_COLOR_TOP[0] * (1 - ratio) + BG_COLOR_BOTTOM[0] * ratio)
        g = int(BG_COLOR_TOP[1] * (1 - ratio) + BG_COLOR_BOTTOM[1] * ratio)
        b = int(BG_COLOR_TOP[2] * (1 - ratio) + BG_COLOR_BOTTOM[2] * ratio)
        draw.line([(0, y), (VIDEO_WIDTH, y)], fill=(r, g, b))

    return np.array(img)


def draw_progress_bar(
    draw: ImageDraw.Draw,
    progress: float,
    y_center: int = VIDEO_HEIGHT // 2,
) -> None:
    """
    Draw a horizontal progress bar at the center of the screen.

    Args:
        draw: PIL ImageDraw object
        progress: Progress value from 0.0 to 1.0
        y_center: Vertical center position of the bar
    """
    bar_width = 600
    bar_height = 12
    border_radius = 6

    x_start = (VIDEO_WIDTH - bar_width) // 2
    x_end = x_start + bar_width
    y_start = y_center - bar_height // 2
    y_end = y_center + bar_height // 2

    draw.rounded_rectangle(
        [(x_start, y_start), (x_end, y_end)],
        radius=border_radius,
        fill=PROGRESS_BAR_BG,
    )

    if progress > 0:
        fill_width = int(bar_width * min(progress, 1.0))
        if fill_width > 0:
            draw.rounded_rectangle(
                [(x_start, y_start), (x_start + fill_width, y_end)],
                radius=border_radius,
                fill=PROGRESS_BAR_FG,
            )


def draw_logo_screen(
    draw: ImageDraw.Draw,
    font: ImageFont.FreeTypeFont,
) -> None:
    """Draw the y2karaoke logo screen for the outro."""
    logo_font = get_font(96)
    tagline_font = get_font(36)
    url_font = get_font(28)

    logo_text = "y2karaoke"
    tagline_text = "youtube to karaoke"
    url_text = "github.com/dtunkelang/y2karaoke"

    # Center the logo
    logo_bbox = logo_font.getbbox(logo_text)
    logo_width = logo_bbox[2] - logo_bbox[0]
    logo_x = (VIDEO_WIDTH - logo_width) // 2
    logo_y = VIDEO_HEIGHT // 2 - 80

    # Center the tagline
    tagline_bbox = tagline_font.getbbox(tagline_text)
    tagline_width = tagline_bbox[2] - tagline_bbox[0]
    tagline_x = (VIDEO_WIDTH - tagline_width) // 2
    tagline_y = logo_y + 100

    # Center the URL
    url_bbox = url_font.getbbox(url_text)
    url_width = url_bbox[2] - url_bbox[0]
    url_x = (VIDEO_WIDTH - url_width) // 2
    url_y = tagline_y + 60

    draw.text((logo_x, logo_y), logo_text, font=logo_font, fill=HIGHLIGHT_COLOR)
    draw.text((tagline_x, tagline_y), tagline_text, font=tagline_font, fill=TEXT_COLOR)
    draw.text((url_x, url_y), url_text, font=url_font, fill=SUNG_COLOR)


def draw_splash_screen(
    draw: ImageDraw.Draw,
    title: str,
    artist: str,
) -> None:
    """Draw the intro splash screen with song title and artist."""
    title_font = get_font(84)
    artist_font = get_font(48)
    logo_font = get_font(32)

    # Truncate long titles
    max_title_chars = 40
    display_title = title if len(title) <= max_title_chars else title[:max_title_chars-3] + "..."

    # Center the title
    title_bbox = title_font.getbbox(display_title)
    title_width = title_bbox[2] - title_bbox[0]
    title_x = (VIDEO_WIDTH - title_width) // 2
    title_y = VIDEO_HEIGHT // 2 - 80

    # Center the artist
    artist_text = f"by {artist}"
    artist_bbox = artist_font.getbbox(artist_text)
    artist_width = artist_bbox[2] - artist_bbox[0]
    artist_x = (VIDEO_WIDTH - artist_width) // 2
    artist_y = title_y + 100

    # y2karaoke branding at bottom
    logo_text = "y2karaoke"
    logo_bbox = logo_font.getbbox(logo_text)
    logo_width = logo_bbox[2] - logo_bbox[0]
    logo_x = (VIDEO_WIDTH - logo_width) // 2
    logo_y = VIDEO_HEIGHT - 100

    # Draw text
    draw.text((title_x, title_y), display_title, font=title_font, fill=HIGHLIGHT_COLOR)
    draw.text((artist_x, artist_y), artist_text, font=artist_font, fill=TEXT_COLOR)
    draw.text((logo_x, logo_y), logo_text, font=logo_font, fill=SUNG_COLOR)


def render_frame(
    lines: list[Line],
    current_time: float,
    font: ImageFont.FreeTypeFont,
    background: np.ndarray,
    title: Optional[str] = None,
    artist: Optional[str] = None,
) -> np.ndarray:
    """Render a single frame at the given time."""
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

        # Show splash screen for the first SPLASH_DURATION seconds
        if current_time < SPLASH_DURATION and title and artist:
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
        draw_logo_screen(draw, font)
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
        draw_splash_screen(draw, title, artist)
        return np.array(img)

    if show_progress_bar:
        draw_progress_bar(draw, progress)
        return np.array(img)

    # Normal lyrics display
    lines_to_show = []
    if current_line_idx < len(lines):
        current_line = lines[current_line_idx]
        # Only mark as "current" (with highlighting) if we're at or past the line's start
        is_actually_current = current_time >= current_line.start_time
        lines_to_show.append((current_line, is_actually_current))
    if current_line_idx + 1 < len(lines):
        lines_to_show.append((lines[current_line_idx + 1], False))

    # Calculate vertical positioning (center the lines)
    total_height = len(lines_to_show) * LINE_SPACING
    start_y = (VIDEO_HEIGHT - total_height) // 2

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
        x = (VIDEO_WIDTH - total_width) // 2

        # Draw each word with appropriate color
        word_idx = 0
        for i, text in enumerate(words_with_spaces):
            if text == " ":
                x += word_widths[i]
                continue

            word = line.words[word_idx]
            word_idx += 1

            # Determine color based on timing
            # KaraFun style: once a word is highlighted, it stays highlighted
            if is_current:
                if current_time >= word.start_time:
                    color = HIGHLIGHT_COLOR  # Highlighted (current or already sung)
                else:
                    color = TEXT_COLOR  # Not yet sung
            else:
                color = TEXT_COLOR  # Next line - all white

            draw.text((x, y), text, font=font, fill=color)
            x += word_widths[i]

    return np.array(img)


def render_karaoke_video(
    lines: list[Line],
    audio_path: str,
    output_path: str,
    title: Optional[str] = None,
    artist: Optional[str] = None,
    timing_offset: float = 0.0,
    background_segments: Optional[list[BackgroundSegment]] = None,
) -> str:
    """
    Render a complete karaoke video.

    Args:
        lines: List of Line objects with word timing
        audio_path: Path to instrumental audio
        output_path: Where to save the video
        title: Song title (for splash screen)
        artist: Artist name (for splash screen)
        timing_offset: Offset in seconds (negative = highlight earlier)
        background_segments: Optional list of background segments for dynamic backgrounds

    Returns:
        Path to the output video
    """
    print("Rendering karaoke video...")
    if timing_offset != 0:
        print(f"Applying timing offset: {timing_offset:+.2f}s")

    # Load audio to get duration
    audio = AudioFileClip(audio_path)
    duration = audio.duration

    # Prepare rendering
    font = get_font()
    static_background = create_gradient_background()

    # Import here to avoid circular imports
    if background_segments:
        from backgrounds import get_background_at_time

    # Create frame generator (apply timing offset)
    def make_frame(t):
        # Offset adjusts effective time: negative offset = highlight earlier
        adjusted_time = t - timing_offset

        # Get background - dynamic if available, otherwise static gradient
        if background_segments:
            bg = get_background_at_time(background_segments, t)
            if bg is not None:
                background = bg
            else:
                background = static_background
        else:
            background = static_background

        return render_frame(lines, adjusted_time, font, background, title, artist)

    # Create video clip
    print(f"Creating video ({duration:.1f}s at {FPS}fps)...")

    # Create video clip with frame generator
    video = VideoClip(make_frame, duration=duration)
    video = video.with_fps(FPS)

    # Add audio
    video = video.with_audio(audio)

    # Write output
    print(f"Writing video to {output_path}...")
    video.write_videofile(
        output_path,
        fps=FPS,
        codec='libx264',
        audio_codec='aac',
        threads=4,
        preset='medium',
        logger=None,
    )

    # Clean up
    audio.close()
    video.close()

    print(f"Done! Output: {output_path}")
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
    print("Saved test_frame.png")
