"""Frame rendering for karaoke videos."""

from typing import Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import math

from ..config import (
    VIDEO_WIDTH,
    VIDEO_HEIGHT,
    LINE_SPACING,
    SPLASH_DURATION,
    INSTRUMENTAL_BREAK_THRESHOLD,
    LYRICS_LEAD_TIME,
    HIGHLIGHT_LEAD_TIME,
    LYRICS_ACTIVATION_LEAD,
    CUE_INDICATOR_DURATION,
    CUE_INDICATOR_MIN_GAP,
    Colors,
)
from .backgrounds_static import draw_logo_screen, draw_splash_screen
from .progress import draw_progress_bar
from .lyrics_renderer import get_singer_colors
from .models import Line


def _draw_cue_indicator(
    draw: ImageDraw.Draw, x: int, y: int, time_until_start: float, font_size: int
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
                radius = int(dot_radius * (0.8 + 0.4 * pulse))
            else:
                # Other dots are solid
                radius = dot_radius

            color = Colors.CUE_INDICATOR
            draw.ellipse(
                [dot_x - radius, dot_y - radius, dot_x + radius, dot_y + radius],
                fill=color,
            )
        else:
            # Inactive dot - dim outline
            draw.ellipse(
                [
                    dot_x - dot_radius,
                    dot_y - dot_radius,
                    dot_x + dot_radius,
                    dot_y + dot_radius,
                ],
                outline=(100, 100, 100),
                width=1,
            )


def _check_intro_progress(lines: list[Line], current_time: float) -> tuple[bool, float]:
    """Check if we should show intro progress bar."""
    if not lines or current_time >= lines[0].start_time:
        return False, 0.0

    first_line = lines[0]
    time_until_first = first_line.start_time - current_time

    if (
        first_line.start_time >= INSTRUMENTAL_BREAK_THRESHOLD
        and time_until_first > LYRICS_LEAD_TIME
    ):
        bar_start = min(SPLASH_DURATION, first_line.start_time - LYRICS_LEAD_TIME)
        break_end = first_line.start_time - LYRICS_LEAD_TIME
        elapsed = current_time - bar_start
        bar_duration = break_end - bar_start
        progress = elapsed / bar_duration if bar_duration > 0 else 1.0
        return True, progress

    return False, 0.0


def _check_mid_song_progress(
    lines: list[Line], current_line_idx: int, current_time: float
) -> tuple[bool, float]:
    """Check if we should show mid-song progress bar during instrumental break."""
    if current_line_idx >= len(lines):
        return False, 0.0

    current_line = lines[current_line_idx]
    next_line_idx = current_line_idx + 1

    if next_line_idx >= len(lines) or current_time < current_line.end_time:
        return False, 0.0

    next_line = lines[next_line_idx]
    gap = next_line.start_time - current_line.end_time

    if gap >= INSTRUMENTAL_BREAK_THRESHOLD:
        time_until_next = next_line.start_time - current_time
        if time_until_next > LYRICS_LEAD_TIME:
            break_start = current_line.end_time
            break_end = next_line.start_time - LYRICS_LEAD_TIME
            break_duration = break_end - break_start
            elapsed = current_time - break_start
            progress = elapsed / break_duration if break_duration > 0 else 1.0
            return True, progress

    return False, 0.0


def _get_lines_to_display(
    lines: list[Line],
    current_line_idx: int,
    current_time: float,
    activation_time: float,
) -> tuple[list[tuple[Line, bool]], int]:
    """Determine which lines to display and which is current."""
    display_start_idx = (current_line_idx // 3) * 3

    next_line_idx = current_line_idx + 1
    if next_line_idx < len(lines):
        curr_line = lines[current_line_idx]
        next_line = lines[next_line_idx]
        gap = next_line.start_time - curr_line.end_time
        if gap >= INSTRUMENTAL_BREAK_THRESHOLD and current_time >= curr_line.end_time:
            display_start_idx = next_line_idx
            current_line_idx = next_line_idx

    lines_to_show = []
    for i in range(4):
        line_idx = display_start_idx + i
        if line_idx >= len(lines):
            break
        if line_idx > 0:
            prev_line = lines[line_idx - 1]
            this_line = lines[line_idx]
            gap = this_line.start_time - prev_line.end_time
            if (
                gap >= INSTRUMENTAL_BREAK_THRESHOLD
                and current_time < this_line.start_time - LYRICS_LEAD_TIME
            ):
                break
        is_current = (
            line_idx == current_line_idx
            and activation_time >= lines[line_idx].start_time
        )
        lines_to_show.append((lines[line_idx], is_current))

    return lines_to_show, display_start_idx


def _check_cue_indicator(
    lines: list[Line],
    lines_to_show: list[tuple[Line, bool]],
    display_start_idx: int,
    current_time: float,
) -> tuple[bool, float]:
    """Check if we should show cue indicator for upcoming line."""
    if not lines_to_show:
        return False, 0.0

    first_line = lines_to_show[0][0]
    time_until_first = first_line.start_time - current_time

    if display_start_idx == 0:
        gap_before = first_line.start_time
    else:
        prev_line = lines[display_start_idx - 1]
        gap_before = first_line.start_time - prev_line.end_time

    if (
        gap_before >= CUE_INDICATOR_MIN_GAP
        and 0 < time_until_first <= CUE_INDICATOR_DURATION
    ):
        return True, time_until_first

    return False, 0.0


def _draw_line_text(
    draw: ImageDraw.Draw,
    line: Line,
    y: int,
    line_x: int,
    words_with_spaces: list[str],
    word_widths: list[int],
    font: ImageFont.FreeTypeFont,
    is_duet: bool,
) -> None:
    """Draw unhighlighted line text."""
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


def _draw_highlight_sweep(
    draw: ImageDraw.Draw,
    line: Line,
    y: int,
    line_x: int,
    total_width: int,
    words_with_spaces: list[str],
    word_widths: list[int],
    font: ImageFont.FreeTypeFont,
    highlight_width: int,
    is_duet: bool,
) -> None:
    """Draw highlighted portion of current line."""
    if highlight_width <= 0:
        return

    highlight_boundary = line_x + highlight_width
    x = line_x
    word_idx = 0

    for i, text in enumerate(words_with_spaces):
        if text == " ":
            x += word_widths[i]
            continue

        word_end_x = x + word_widths[i]
        if x >= highlight_boundary:
            break

        word = line.words[word_idx]
        if is_duet and word.singer:
            _, word_highlight_color = get_singer_colors(word.singer, False)
        else:
            word_highlight_color = Colors.HIGHLIGHT

        if word_end_x <= highlight_boundary:
            draw.text((x, y), text, font=font, fill=word_highlight_color)
        else:
            char_x = x
            for char in text:
                char_bbox = font.getbbox(char)
                char_width = char_bbox[2] - char_bbox[0]
                if char_x < highlight_boundary:
                    draw.text((char_x, y), char, font=font, fill=word_highlight_color)
                char_x += char_width

        x += word_widths[i]
        word_idx += 1


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
    show_progress_bar, progress = _check_intro_progress(lines, current_time)

    activation_time = current_time + LYRICS_ACTIVATION_LEAD
    current_line_idx = 0
    for i, line in enumerate(lines):
        if line.start_time <= activation_time:
            current_line_idx = i

    if lines and current_time >= lines[-1].end_time:
        draw_logo_screen(draw, font, video_width, video_height)
        return np.array(img)

    if not show_progress_bar:
        show_progress_bar, progress = _check_mid_song_progress(
            lines, current_line_idx, current_time
        )

    if show_splash and title and artist:
        draw_splash_screen(draw, title, artist, video_width, video_height)
        return np.array(img)

    if show_progress_bar:
        draw_progress_bar(draw, progress, video_width, video_height)
        return np.array(img)

    lines_to_show, display_start_idx = _get_lines_to_display(
        lines, current_line_idx, current_time, activation_time
    )
    total_height = len(lines_to_show) * LINE_SPACING
    start_y = (video_height - total_height) // 2

    show_cue, cue_time_until = _check_cue_indicator(
        lines, lines_to_show, display_start_idx, current_time
    )

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

        if idx == 0 and show_cue:
            font_size = LINE_SPACING * 3 // 4
            _draw_cue_indicator(
                draw, line_x, y + font_size // 2, cue_time_until, font_size
            )

        _draw_line_text(
            draw, line, y, line_x, words_with_spaces, word_widths, font, is_duet
        )

        if is_current:
            highlight_time = activation_time + HIGHLIGHT_LEAD_TIME
            line_duration = line.end_time - line.start_time
            if line_duration > 0:
                line_progress = (highlight_time - line.start_time) / line_duration
                line_progress = max(0.0, min(1.0, line_progress))
            else:
                line_progress = 1.0
            highlight_width = int(total_width * line_progress)
            _draw_highlight_sweep(
                draw,
                line,
                y,
                line_x,
                total_width,
                words_with_spaces,
                word_widths,
                font,
                highlight_width,
                is_duet,
            )

    return np.array(img)
