"""Text and highlight rendering primitives for lyric lines."""

from PIL import ImageDraw, ImageFont

from ....config import Colors, FIRST_WORD_HIGHLIGHT_DELAY
from .singer_style import get_singer_colors
from ...models import Line


def draw_line_text(
    draw: ImageDraw.ImageDraw,
    line: Line,
    y: int,
    line_x: float,
    words_with_spaces: list[str],
    word_widths: list[float],
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    is_duet: bool,
) -> None:
    """Draw unhighlighted line text."""
    x = float(line_x)
    word_idx = 0
    for i, text in enumerate(words_with_spaces):
        if text == " ":
            x += word_widths[i]
            continue
        word = line.words[word_idx]
        if is_duet and word.singer:
            text_color, _ = get_singer_colors(word.singer)
        else:
            text_color = Colors.TEXT
        draw.text((x, y), text, font=font, fill=text_color)
        x += word_widths[i]
        word_idx += 1


def draw_highlight_sweep(
    draw: ImageDraw.ImageDraw,
    line: Line,
    y: int,
    line_x: float,
    total_width: float,
    words_with_spaces: list[str],
    word_widths: list[float],
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    highlight_width: int,
    is_duet: bool,
) -> None:
    """Draw highlighted portion of current line."""
    if highlight_width <= 0:
        return

    highlight_boundary = line_x + highlight_width
    x = float(line_x)
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
            _, word_highlight_color = get_singer_colors(word.singer)
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


def compute_word_highlight_width(
    line: Line,
    words_with_spaces: list[str],
    word_widths: list[float],
    highlight_time: float,
) -> int:
    """Compute highlight width using word-level timing."""
    if not line.words:
        return 0

    word_spans: list[float] = []
    width_idx = 0
    for i, _word in enumerate(line.words):
        if width_idx >= len(word_widths):
            break
        span = word_widths[width_idx]
        width_idx += 1
        if i < len(line.words) - 1 and width_idx < len(word_widths):
            span += word_widths[width_idx]
            width_idx += 1
        word_spans.append(span)

    highlight_width = 0.0
    for word_idx, (word, span) in enumerate(zip(line.words, word_spans)):
        start_time = word.start_time
        end_time = word.end_time
        if word_idx == 0 and FIRST_WORD_HIGHLIGHT_DELAY > 0 and end_time > start_time:
            start_time += FIRST_WORD_HIGHLIGHT_DELAY
            end_time += FIRST_WORD_HIGHLIGHT_DELAY

        if highlight_time >= end_time:
            highlight_width += span
            continue
        if highlight_time <= start_time:
            break
        duration = max(end_time - start_time, 0.01)
        fraction = (highlight_time - start_time) / duration
        fraction = max(0.0, min(1.0, fraction))
        highlight_width += span * fraction
        break

    return int(highlight_width)
