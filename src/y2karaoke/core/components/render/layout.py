"""Layout helpers for lyric line text rendering."""

from typing import Dict, List, Optional, Tuple

from PIL import ImageFont

from ...models import Line


def get_or_build_line_layout(
    line: Line,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    layout_cache: Optional[Dict[int, Tuple[List[str], List[float], float]]],
) -> tuple[list[str], list[float], float]:
    """Return cached line text layout or compute and store it."""
    if layout_cache is not None and id(line) in layout_cache:
        return layout_cache[id(line)]

    words_with_spaces = []
    for i, word in enumerate(line.words):
        words_with_spaces.append(word.text)
        if i < len(line.words) - 1:
            words_with_spaces.append(" ")

    total_width = 0.0
    word_widths = []
    for text in words_with_spaces:
        bbox = font.getbbox(text)
        w = float(bbox[2] - bbox[0])
        word_widths.append(w)
        total_width += w

    if layout_cache is not None:
        layout_cache[id(line)] = (words_with_spaces, word_widths, total_width)
    return words_with_spaces, word_widths, total_width
