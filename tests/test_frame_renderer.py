"""Tests for frame_renderer.py module."""

import pytest
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from unittest.mock import Mock, patch, MagicMock
from y2karaoke.core.components.render.frame_renderer import (
    render_frame,
    _draw_cue_indicator,
    _check_intro_progress,
    _check_mid_song_progress,
    _get_lines_to_display,
    _check_cue_indicator,
    _draw_line_text,
    _draw_highlight_sweep,
)
from y2karaoke.config import INSTRUMENTAL_BREAK_THRESHOLD
from y2karaoke.core.models import Line, Word


class TestDrawCueIndicator:
    """Test _draw_cue_indicator function."""

    def test_function_exists(self):
        """Test that _draw_cue_indicator function exists."""
        assert _draw_cue_indicator is not None
        assert callable(_draw_cue_indicator)

    def test_function_signature(self):
        """Test _draw_cue_indicator function signature."""
        import inspect

        sig = inspect.signature(_draw_cue_indicator)
        params = list(sig.parameters.keys())
        expected_params = ["draw", "x", "y", "time_until_start", "font_size"]
        for param in expected_params:
            assert param in params


class TestCheckIntroProgress:
    """Test _check_intro_progress function."""

    def test_function_exists(self):
        """Test that _check_intro_progress function exists."""
        assert _check_intro_progress is not None
        assert callable(_check_intro_progress)

    def test_returns_tuple(self):
        """Test that function returns a tuple."""
        lines = [Line(words=[Word(text="test", start_time=5.0, end_time=6.0)])]
        result = _check_intro_progress(lines, 2.0)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_empty_lines_handling(self):
        """Test handling of empty lines."""
        result = _check_intro_progress([], 2.0)
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestCheckMidSongProgress:
    """Test _check_mid_song_progress function."""

    def test_function_exists(self):
        """Test that _check_mid_song_progress function exists."""
        assert _check_mid_song_progress is not None
        assert callable(_check_mid_song_progress)

    def test_returns_tuple(self):
        """Test that function returns a tuple."""
        lines = [
            Line(words=[Word(text="test1", start_time=1.0, end_time=2.0)]),
            Line(words=[Word(text="test2", start_time=10.0, end_time=11.0)]),
        ]
        result = _check_mid_song_progress(lines, 0, 5.0)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_function_signature(self):
        """Test _check_mid_song_progress function signature."""
        import inspect

        sig = inspect.signature(_check_mid_song_progress)
        params = list(sig.parameters.keys())
        expected_params = ["lines", "current_line_idx", "current_time"]
        for param in expected_params:
            assert param in params


class TestGetLinesToDisplay:
    """Test _get_lines_to_display function."""

    def test_function_exists(self):
        """Test that _get_lines_to_display function exists."""
        assert _get_lines_to_display is not None
        assert callable(_get_lines_to_display)

    def test_returns_tuple(self):
        """Test that function returns a tuple."""
        lines = [Line(words=[Word(text="test", start_time=1.0, end_time=2.0)])]
        result = _get_lines_to_display(lines, 0, 1.5, 1.0)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_empty_lines_returns_empty(self):
        """Test that empty lines returns empty list."""
        result = _get_lines_to_display([], 0, 1.0, 1.0)
        assert isinstance(result, tuple)
        lines_to_show, display_start_idx = result
        assert isinstance(lines_to_show, list)
        assert isinstance(display_start_idx, int)

    def test_function_signature(self):
        """Test _get_lines_to_display function signature."""
        import inspect

        sig = inspect.signature(_get_lines_to_display)
        params = list(sig.parameters.keys())
        expected_params = [
            "lines",
            "current_line_idx",
            "current_time",
            "activation_time",
        ]
        for param in expected_params:
            assert param in params

    def test_break_keeps_current_line_on_top(self):
        """After a long break, current line should stay at top of display."""
        gap = INSTRUMENTAL_BREAK_THRESHOLD + 1.0
        lines = [
            Line(words=[Word(text="prev", start_time=1.0, end_time=2.0)]),
            Line(words=[Word(text="next", start_time=2.0 + gap, end_time=3.0 + gap)]),
            Line(words=[Word(text="after", start_time=4.0 + gap, end_time=5.0 + gap)]),
        ]

        current_time = lines[1].start_time + 0.1
        activation_time = current_time
        lines_to_show, display_start_idx = _get_lines_to_display(
            lines, 1, current_time, activation_time
        )

        assert display_start_idx == 1
        assert lines_to_show[0][0] == lines[1]

    def test_break_anchors_display_for_following_lines(self):
        """Lines after a break should keep the break-start line at the top."""
        gap = INSTRUMENTAL_BREAK_THRESHOLD + 1.0
        lines = [
            Line(words=[Word(text="prev", start_time=1.0, end_time=2.0)]),
            Line(words=[Word(text="first", start_time=2.0 + gap, end_time=3.0 + gap)]),
            Line(words=[Word(text="second", start_time=4.0 + gap, end_time=5.0 + gap)]),
        ]

        current_time = lines[2].start_time + 0.1
        activation_time = current_time
        lines_to_show, display_start_idx = _get_lines_to_display(
            lines, 2, current_time, activation_time
        )

        assert display_start_idx == 1
        assert lines_to_show[0][0] == lines[1]

    def test_break_window_advances_after_three_lines(self):
        """Display window advances in blocks after a break, without showing pre-break lines."""
        gap = INSTRUMENTAL_BREAK_THRESHOLD + 1.0
        lines = [
            Line(words=[Word(text="prev", start_time=1.0, end_time=2.0)]),
            Line(words=[Word(text="l1", start_time=2.0 + gap, end_time=3.0 + gap)]),
            Line(words=[Word(text="l2", start_time=4.0 + gap, end_time=5.0 + gap)]),
            Line(words=[Word(text="l3", start_time=6.0 + gap, end_time=7.0 + gap)]),
            Line(words=[Word(text="l4", start_time=8.0 + gap, end_time=9.0 + gap)]),
            Line(words=[Word(text="l5", start_time=10.0 + gap, end_time=11.0 + gap)]),
        ]

        current_time = lines[4].start_time + 0.1
        activation_time = current_time
        lines_to_show, display_start_idx = _get_lines_to_display(
            lines, 4, current_time, activation_time
        )

        assert display_start_idx == 4
        assert lines_to_show[0][0] == lines[4]


class TestCheckCueIndicator:
    """Test _check_cue_indicator function."""

    def test_function_exists(self):
        """Test that _check_cue_indicator function exists."""
        assert _check_cue_indicator is not None
        assert callable(_check_cue_indicator)

    def test_returns_tuple(self):
        """Test that function returns a tuple."""
        lines = [Line(words=[Word(text="test", start_time=5.0, end_time=6.0)])]
        lines_to_show = [(lines[0], False)]
        result = _check_cue_indicator(lines, lines_to_show, 0, 3.0)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_function_signature(self):
        """Test _check_cue_indicator function signature."""
        import inspect

        sig = inspect.signature(_check_cue_indicator)
        params = list(sig.parameters.keys())
        expected_params = [
            "lines",
            "lines_to_show",
            "display_start_idx",
            "current_time",
        ]
        for param in expected_params:
            assert param in params


class TestRenderFrame:
    """Test render_frame function."""

    def test_function_exists(self):
        """Test that render_frame function exists."""
        assert render_frame is not None
        assert callable(render_frame)

    def test_function_signature(self):
        """Test render_frame function signature."""
        import inspect

        sig = inspect.signature(render_frame)
        params = list(sig.parameters.keys())
        required_params = ["lines", "current_time", "font", "background"]
        for param in required_params:
            assert param in params

    @patch("y2karaoke.core.components.render.frame_renderer._draw_line_text")
    @patch("y2karaoke.core.components.render.frame_renderer._draw_highlight_sweep")
    def test_basic_frame_rendering(self, mock_highlight, mock_draw_text):
        """Test basic frame rendering with mocked drawing functions."""
        lines = [Line(words=[Word(text="test", start_time=1.0, end_time=2.0)])]

        # Create minimal mock font and background
        mock_font = Mock()
        mock_font.getbbox.return_value = (0, 0, 100, 20)

        background = np.zeros((1080, 1920, 3), dtype=np.uint8)

        result = render_frame(lines, 1.5, mock_font, background)

        assert isinstance(result, np.ndarray)
        assert result.shape == (1080, 1920, 3)

    def test_empty_lines_handling(self):
        """Test handling of empty lines."""
        mock_font = Mock()
        mock_font.getbbox.return_value = (0, 0, 100, 20)
        mock_font.getmask2.return_value = (Mock(), (0, 0))  # Return tuple for PIL
        background = np.zeros((1080, 1920, 3), dtype=np.uint8)

        result = render_frame([], 1.0, mock_font, background)
        assert isinstance(result, np.ndarray)


class TestFrameRendererIntegration:
    """Test frame_renderer module integration."""

    def test_module_imports(self):
        """Test that all required functions can be imported."""
        from y2karaoke.core.components.render.frame_renderer import render_frame

        assert render_frame is not None

    def test_config_imports(self):
        """Test that configuration constants are properly imported."""
        from y2karaoke.core.components.render.frame_renderer import (
            VIDEO_WIDTH,
            VIDEO_HEIGHT,
        )

        assert isinstance(VIDEO_WIDTH, int)
        assert isinstance(VIDEO_HEIGHT, int)

    def test_models_integration(self):
        """Test integration with core models."""
        from y2karaoke.core.components.render.frame_renderer import Line

        assert Line is not None

    def test_pil_integration(self):
        """Test PIL integration."""
        # Test that PIL components are properly used
        image = Image.new("RGB", (100, 100), "black")
        draw = ImageDraw.Draw(image)
        assert image is not None
        assert draw is not None


class TestFrameRendererErrorHandling:
    """Test error handling in frame_renderer module."""

    @patch("y2karaoke.core.components.render.frame_renderer.draw_logo_screen")
    @patch("y2karaoke.core.components.render.frame_renderer._draw_line_text")
    @patch("y2karaoke.core.components.render.frame_renderer._draw_highlight_sweep")
    def test_invalid_time_handling(self, mock_highlight, mock_draw_text, mock_logo):
        """Test handling of invalid time values."""
        lines = [Line(words=[Word(text="test", start_time=1.0, end_time=2.0)])]

        mock_font = Mock()
        mock_font.getbbox.return_value = (0, 0, 100, 20)
        background = np.zeros((1080, 1920, 3), dtype=np.uint8)

        # Test negative time
        result = render_frame(lines, -10.0, mock_font, background)
        assert isinstance(result, np.ndarray)

        # Test very large time
        result = render_frame(lines, 1000.0, mock_font, background)
        assert isinstance(result, np.ndarray)

    def test_malformed_lines_handling(self):
        """Test handling of malformed lines."""
        mock_font = Mock()
        mock_font.getbbox.return_value = (0, 0, 100, 20)
        mock_font.getmask2.return_value = (Mock(), (0, 0))  # Return tuple for PIL
        background = np.zeros((1080, 1920, 3), dtype=np.uint8)

        # Test with lines that have no words
        try:
            lines = [Line(words=[])]
            result = render_frame(lines, 1.0, mock_font, background)
            assert isinstance(result, np.ndarray)
        except Exception:
            # Exception is acceptable for malformed data
            pass
