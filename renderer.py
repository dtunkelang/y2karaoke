"""Karaoke video renderer with KaraFun-style word highlighting."""

import os
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy import AudioFileClip, VideoClip

from lyrics import Line, Word


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

# Font settings
FONT_SIZE = 72
LINE_SPACING = 100


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


def render_frame(
    lines: list[Line],
    current_time: float,
    font: ImageFont.FreeTypeFont,
    background: np.ndarray,
) -> np.ndarray:
    """Render a single frame at the given time."""
    img = Image.fromarray(background.copy())
    draw = ImageDraw.Draw(img)

    # Find current line index
    current_line_idx = 0
    for i, line in enumerate(lines):
        if line.start_time <= current_time:
            current_line_idx = i

    # Show current line and next line
    lines_to_show = []
    if current_line_idx < len(lines):
        lines_to_show.append((lines[current_line_idx], True))  # (line, is_current)
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
            if is_current:
                if current_time >= word.end_time:
                    color = SUNG_COLOR  # Already sung
                elif current_time >= word.start_time:
                    color = HIGHLIGHT_COLOR  # Currently singing
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
    timing_offset: float = 0.0,
) -> str:
    """
    Render a complete karaoke video.

    Args:
        lines: List of Line objects with word timing
        audio_path: Path to instrumental audio
        output_path: Where to save the video
        timing_offset: Offset in seconds (negative = highlight earlier)

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
    background = create_gradient_background()

    # Create frame generator (apply timing offset)
    def make_frame(t):
        # Offset adjusts effective time: negative offset = highlight earlier
        adjusted_time = t - timing_offset
        return render_frame(lines, adjusted_time, font, background)

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
