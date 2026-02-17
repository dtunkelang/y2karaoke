import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from PIL import ImageFont

from y2karaoke.core.components.render.frame_renderer import render_frame
from y2karaoke.core.models import Line, Word


def test_render_frame_caches_layout():
    # Setup
    line = Line(words=[Word(text="Hello", start_time=0.0, end_time=1.0)])
    lines = [line]

    real_font = ImageFont.load_default()
    font = MagicMock(wraps=real_font)

    background = np.zeros((100, 100, 3), dtype=np.uint8)
    layout_cache = {}

    # Patch highlight sweep to avoid extra getbbox calls
    with patch("y2karaoke.core.components.render.frame_renderer._draw_highlight_sweep"):
        # First call: Should call getbbox for layout
        render_frame(
            lines=lines,
            current_time=0.5,
            font=font,
            background=background,
            layout_cache=layout_cache,
            width=100,
            height=100,
        )

        assert font.getbbox.call_count > 0
        initial_call_count = font.getbbox.call_count
        assert id(line) in layout_cache

        # Second call: Should NOT call getbbox (cache hit)
        render_frame(
            lines=lines,
            current_time=0.6,
            font=font,
            background=background,
            layout_cache=layout_cache,
            width=100,
            height=100,
        )

        assert font.getbbox.call_count == initial_call_count


def test_render_frame_works_without_cache():
    # Setup
    line = Line(words=[Word(text="Hello", start_time=0.0, end_time=1.0)])
    lines = [line]

    real_font = ImageFont.load_default()
    font = MagicMock(wraps=real_font)

    background = np.zeros((100, 100, 3), dtype=np.uint8)

    with patch("y2karaoke.core.components.render.frame_renderer._draw_highlight_sweep"):
        # Call without cache
        render_frame(
            lines=lines,
            current_time=0.5,
            font=font,
            background=background,
            layout_cache=None,
            width=100,
            height=100,
        )

        count_1 = font.getbbox.call_count
        assert count_1 > 0

        # Call again
        render_frame(
            lines=lines,
            current_time=0.6,
            font=font,
            background=background,
            layout_cache=None,
            width=100,
            height=100,
        )

        # Should calculate again
        assert font.getbbox.call_count > count_1
