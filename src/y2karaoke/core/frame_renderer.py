"""Frame rendering for karaoke videos."""

from typing import Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import math

from ..config import (
    VIDEO_WIDTH, VIDEO_HEIGHT, LINE_SPACING,
    SPLASH_DURATION, INSTRUMENTAL_BREAK_THRESHOLD, LYRICS_LEAD_TIME,
    HIGHLIGHT_LEAD_TIME, LYRICS_ACTIVATION_LEAD,
    CUE_INDICATOR_DURATION, CUE_INDICATOR_MIN_GAP, Colors
)
from .backgrounds_static import draw_logo_screen, draw_splash_screen
from .progress import draw_progress_bar
from .lyrics_renderer import get_singer_colors
from .models import Line


def _draw_cue_indicator(
    draw: ImageDraw.Draw,
    x: int,
    y: int,
    time_until_start: float,
    font_size: int
) -> None:
    """Draw animated cue indicator (pulsing dots) to prepare singer.

    Args:
        draw: PIL ImageDraw object
        x: X position (left side of line)
        y: Y position (vertical center of line)
        time_until_start: Seconds until the line starts
        font_size: Font size for scaling the indicator
    """
    # Three dots that pulse in sequence as countdown
    dot_radius = max(4, font_size // 12)
    dot_spacing = dot_radius * 3
    total_width = dot_spacing * 2 + dot_radius * 2

    # Position dots to the left of the line
    start_x = x - total_width - dot_spacing

    # Calculate which dots to show based on countdown
    # At 3s: 3 dots, at 2s: 2 dots, at 1s: 1 dot (pulsing)
    dots_to_show = min(3, max(1, int(time_until_start) + 1))

    # Pulse animation (sine wave for smooth pulsing)
    pulse = 0.5 + 0.5 * math.sin(time_until_start * math.pi * 3)  # 1.5 Hz pulse

    for i in range(3):
        dot_x = start_x + i * dot_spacing
        dot_y = y

        if i < dots_to_show:
            # Active dot - gold color with pulse on the leading dot
            if i == dots_to_show - 1:
                # Leading dot pulses
                alpha = int(128 + 127 * pulse)
                radius = int(dot_radius * (0.8 + 0.4 * pulse))
            else:
                # Other dots are solid
                alpha = 255
                radius = dot_radius

            color = Colors.CUE_INDICATOR
            draw.ellipse(
                [dot_x - radius, dot_y - radius, dot_x + radius, dot_y + radius],
                fill=color
            )
        else:
            # Inactive dot - dim outline
            draw.ellipse(
                [dot_x - dot_radius, dot_y - dot_radius,
                 dot_x + dot_radius, dot_y + dot_radius],
                outline=(100, 100, 100),
                width=1
            )


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
    # Use activation lead to start highlighting a line early (compensate for timing lag)
    activation_time = current_time + LYRICS_ACTIVATION_LEAD
    current_line_idx = 0
    for i, line in enumerate(lines):
        if line.start_time <= activation_time:
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
        is_current = line_idx == current_line_idx and activation_time >= lines[line_idx].start_time
        lines_to_show.append((lines[line_idx], is_current))

    total_height = len(lines_to_show) * LINE_SPACING
    start_y = (video_height - total_height) // 2

    # Determine if we should show cue indicator for the first displayed line
    show_cue = False
    cue_time_until = 0.0
    if lines_to_show:
        first_line = lines_to_show[0][0]
        time_until_first = first_line.start_time - current_time

        # Check if this is after a significant gap (or at song start)
        first_line_idx = display_start_idx
        if first_line_idx == 0:
            # Start of song - show cue if we're in the countdown window
            gap_before = first_line.start_time  # Gap from 0
        else:
            prev_line = lines[first_line_idx - 1]
            gap_before = first_line.start_time - prev_line.end_time

        # Show cue indicator if:
        # 1. There's a significant gap before this line
        # 2. We're within the cue duration window
        # 3. The line hasn't started yet
        if (gap_before >= CUE_INDICATOR_MIN_GAP and
            0 < time_until_first <= CUE_INDICATOR_DURATION):
            show_cue = True
            cue_time_until = time_until_first

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
            w = bbox[2] - bbox[0]
            word_widths.append(w)
            total_width += w

        line_x = (video_width - total_width) // 2

        # Draw cue indicator for first line if appropriate
        if idx == 0 and show_cue:
            # Get font size for scaling (estimate from line spacing)
            font_size = LINE_SPACING * 3 // 4
            _draw_cue_indicator(draw, line_x, y + font_size // 2, cue_time_until, font_size)

        # Determine default colors (used for highlight and non-duet)
        if is_duet and line.words and line.words[0].singer:
            _, highlight_color = get_singer_colors(line.words[0].singer, False)
        else:
            highlight_color = Colors.HIGHLIGHT

        # Draw unhighlighted text first (with per-word duet colors if applicable)
        x = line_x
        word_idx = 0
        for i, text in enumerate(words_with_spaces):
            if text == " ":
                x += word_widths[i]
                continue
            word = line.words[word_idx]
            if is_duet and word.singer:
                text_color, _ = get_singer_colors(word.singer, False)
            else:
                text_color = Colors.TEXT
            draw.text((x, y), text, font=font, fill=text_color)
            x += word_widths[i]
            word_idx += 1

        # For current line, draw highlighted portion with gradual sweep
        if is_current:
            # Use activation_time (which includes LYRICS_ACTIVATION_LEAD) plus small lead
            # for smooth highlight sweep that completes by the shifted end time
            highlight_time = activation_time + HIGHLIGHT_LEAD_TIME

            # Calculate line progress (0 to 1) based on line duration
            # This ensures highlight completes by line.end_time regardless of word timings
            line_duration = line.end_time - line.start_time
            if line_duration > 0:
                line_progress = (highlight_time - line.start_time) / line_duration
                line_progress = max(0.0, min(1.0, line_progress))
            else:
                line_progress = 1.0

            # Calculate highlight width as proportion of total line width
            highlight_width = int(total_width * line_progress)

            # Draw highlighted text directly on top, clipped to highlight_width
            if highlight_width > 0:
                highlight_boundary = line_x + highlight_width
                x = line_x
                word_idx = 0
                for i, text in enumerate(words_with_spaces):
                    if text == " ":
                        x += word_widths[i]
                        continue

                    word_end_x = x + word_widths[i]

                    # Skip if entirely past highlight boundary
                    if x >= highlight_boundary:
                        break

                    word = line.words[word_idx]
                    if is_duet and word.singer:
                        _, word_highlight_color = get_singer_colors(word.singer, False)
                    else:
                        word_highlight_color = Colors.HIGHLIGHT

                    if word_end_x <= highlight_boundary:
                        # Entire word is highlighted - draw directly
                        draw.text((x, y), text, font=font, fill=word_highlight_color)
                    else:
                        # Partial word - draw character by character
                        char_x = x
                        for char in text:
                            char_bbox = font.getbbox(char)
                            char_width = char_bbox[2] - char_bbox[0]
                            if char_x + char_width <= highlight_boundary:
                                draw.text((char_x, y), char, font=font, fill=word_highlight_color)
                            elif char_x < highlight_boundary:
                                # Partial character - still draw it highlighted
                                draw.text((char_x, y), char, font=font, fill=word_highlight_color)
                            char_x += char_width

                    x += word_widths[i]
                    word_idx += 1

    return np.array(img)
