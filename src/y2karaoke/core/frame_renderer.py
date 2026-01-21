"""Frame rendering for karaoke videos."""

from typing import Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..config import (
    VIDEO_WIDTH, VIDEO_HEIGHT, LINE_SPACING,
    SPLASH_DURATION, INSTRUMENTAL_BREAK_THRESHOLD, LYRICS_LEAD_TIME, Colors
)
from .backgrounds_static import draw_logo_screen, draw_splash_screen
from .progress import draw_progress_bar
from .lyrics_renderer import get_singer_colors
from .models import Line


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

    show_splash = current_time < SPLASH_DURATION and title and artist
    show_progress_bar = False
    progress = 0.0

    # --- Intro before first line ---
    if lines and current_time < lines[0].start_time:
        first_line = lines[0]
        time_until_first = first_line.start_time - current_time
        if first_line.start_time >= INSTRUMENTAL_BREAK_THRESHOLD and time_until_first > LYRICS_LEAD_TIME:
            show_progress_bar = True
            bar_start = min(SPLASH_DURATION, first_line.start_time - LYRICS_LEAD_TIME)
            break_end = first_line.start_time - LYRICS_LEAD_TIME
            elapsed = current_time - bar_start
            bar_duration = break_end - bar_start
            progress = elapsed / bar_duration if bar_duration > 0 else 1.0

    # --- Determine current line ---
    current_line_idx = 0
    for i, line in enumerate(lines):
        if line.start_time <= current_time:
            current_line_idx = i

    # --- Outro after last lyrics ---
    if lines and current_time >= lines[-1].end_time:
        draw_logo_screen(draw, font, video_width, video_height)
        return np.array(img)

    # --- Mid-song gaps ---
    if not show_progress_bar and current_line_idx < len(lines):
        current_line = lines[current_line_idx]
        next_line_idx = current_line_idx + 1
        if next_line_idx < len(lines) and current_time >= current_line.end_time:
            next_line = lines[next_line_idx]
            gap = next_line.start_time - current_line.end_time
            if gap >= INSTRUMENTAL_BREAK_THRESHOLD:
                time_until_next = next_line.start_time - current_time
                if time_until_next > LYRICS_LEAD_TIME:
                    show_progress_bar = True
                    break_start = current_line.end_time
                    break_end = next_line.start_time - LYRICS_LEAD_TIME
                    break_duration = break_end - break_start
                    elapsed = current_time - break_start
                    progress = elapsed / break_duration if break_duration > 0 else 1.0

    # --- Splash or progress bar ---
    if show_splash:
        draw_splash_screen(draw, title, artist, video_width, video_height)
        return np.array(img)

    if show_progress_bar:
        draw_progress_bar(draw, progress, video_width, video_height)
        return np.array(img)

    # --- Normal lyrics display ---
    lines_to_show = []
    display_start_idx = (current_line_idx // 3) * 3

    next_line_idx = current_line_idx + 1
    if next_line_idx < len(lines):
        curr_line = lines[current_line_idx]
        next_line = lines[next_line_idx]
        gap = next_line.start_time - curr_line.end_time
        if gap >= INSTRUMENTAL_BREAK_THRESHOLD and current_time >= curr_line.end_time:
            display_start_idx = next_line_idx
            current_line_idx = next_line_idx

    for i in range(4):
        line_idx = display_start_idx + i
        if line_idx >= len(lines):
            break
        if line_idx > 0:
            prev_line = lines[line_idx - 1]
            this_line = lines[line_idx]
            gap = this_line.start_time - prev_line.end_time
            if gap >= INSTRUMENTAL_BREAK_THRESHOLD and current_time < this_line.start_time - LYRICS_LEAD_TIME:
                break
        is_current = line_idx == current_line_idx and current_time >= lines[line_idx].start_time
        lines_to_show.append((lines[line_idx], is_current))

    total_height = len(lines_to_show) * LINE_SPACING
    start_y = (video_height - total_height) // 2

    for idx, (line, is_current) in enumerate(lines_to_show):
        y = start_y + idx * LINE_SPACING
        words_with_spaces = []
        for i, word in enumerate(line.words):
            words_with_spaces.append(word.text)
            if i < len(line.words) - 1:
                words_with_spaces.append(" ")

        total_width = 0
        word_widths = []
        for text in words_with_spaces:
            bbox = font.getbbox(text)
            width = bbox[2] - bbox[0]
            word_widths.append(width)
            total_width += width

        x = (video_width - total_width) // 2
        word_idx = 0
        for i, text in enumerate(words_with_spaces):
            if text == " ":
                x += word_widths[i]
                continue
            word = line.words[word_idx]
            word_idx += 1

            if is_duet and word.singer:
                text_color, highlight_color = get_singer_colors(word.singer, False)
                color = highlight_color if is_current and current_time >= word.start_time else text_color
            else:
                color = Colors.HIGHLIGHT if is_current and current_time >= word.start_time else Colors.TEXT

            draw.text((x, y), text, font=font, fill=color)
            x += word_widths[i]

    return np.array(img)
