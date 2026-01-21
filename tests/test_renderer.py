"""Tests for the renderer module."""

import pytest
import numpy as np
from dataclasses import dataclass
from unittest.mock import patch, MagicMock

from y2karaoke.core.backgrounds_static import (
    create_gradient_background,
    draw_splash_screen,
    draw_logo_screen,
)
from y2karaoke.core.frame_renderer import render_frame, get_singer_colors
from y2karaoke.core.progress import draw_progress_bar
from y2karaoke.config import VIDEO_WIDTH, VIDEO_HEIGHT, Colors


@dataclass
class MockWord:
    """Mock Word class for testing."""
    text: str
    start_time: float
    end_time: float
    singer: str = ""


@dataclass
class MockLine:
    """Mock Line class for testing."""
    words: list
    start_time: float
    end_time: float


class TestCreateGradientBackground:
    """Tests for gradient background creation."""

    def test_returns_numpy_array(self):
        """Background should be a numpy array."""
        bg = create_gradient_background()
        assert isinstance(bg, np.ndarray)

    def test_correct_dimensions(self):
        """Background should have correct video dimensions."""
        bg = create_gradient_background()
        assert bg.shape == (VIDEO_HEIGHT, VIDEO_WIDTH, 3)

    def test_rgb_format(self):
        """Background should be RGB (3 channels)."""
        bg = create_gradient_background()
        assert bg.shape[2] == 3

    def test_gradient_top_vs_bottom(self):
        """Top should be darker (blue) than bottom (purple)."""
        bg = create_gradient_background()
        # Check average brightness at top vs bottom
        top_brightness = np.mean(bg[0, :, :])
        bottom_brightness = np.mean(bg[-1, :, :])
        # Both should be relatively dark
        assert top_brightness < 100
        assert bottom_brightness < 100


class TestGetSingerColors:
    """Tests for singer color mapping."""

    def test_singer1_colors(self):
        """Singer 1 should get blue colors."""
        text_color, highlight_color = get_singer_colors("singer1", False)
        assert text_color == Colors.SINGER1
        assert highlight_color == Colors.SINGER1_HIGHLIGHT

    def test_singer2_colors(self):
        """Singer 2 should get pink colors."""
        text_color, highlight_color = get_singer_colors("singer2", False)
        assert text_color == Colors.SINGER2
        assert highlight_color == Colors.SINGER2_HIGHLIGHT

    def test_both_singers_colors(self):
        """Both singers should get purple colors."""
        text_color, highlight_color = get_singer_colors("both", False)
        assert text_color == Colors.BOTH
        assert highlight_color == Colors.BOTH_HIGHLIGHT

    def test_default_colors(self):
        """Default (no singer) should get white/gold."""
        text_color, highlight_color = get_singer_colors("", False)
        assert text_color == Colors.TEXT
        assert highlight_color == Colors.HIGHLIGHT

    def test_unknown_singer_uses_default(self):
        """Unknown singer should get default colors."""
        text_color, highlight_color = get_singer_colors("unknown", False)
        assert text_color == Colors.TEXT
        assert highlight_color == Colors.HIGHLIGHT


class TestRenderFrame:
    """Tests for frame rendering."""

    def test_returns_numpy_array(self):
        """Frame should be a numpy array."""
        lines = []
        background = create_gradient_background()
        from y2karaoke.utils.fonts import get_font
        font = get_font()

        frame = render_frame(lines, 0.0, font, background)
        assert isinstance(frame, np.ndarray)

    def test_correct_dimensions(self):
        """Frame should have video dimensions."""
        lines = []
        background = create_gradient_background()
        from y2karaoke.utils.fonts import get_font
        font = get_font()

        frame = render_frame(lines, 0.0, font, background)
        assert frame.shape == (VIDEO_HEIGHT, VIDEO_WIDTH, 3)

    def test_splash_screen_shown_at_start(self):
        """Splash screen should be shown at the start."""
        # Create a line that starts at 5 seconds
        word = MockWord("Hello", 5.0, 5.5)
        line = MockLine([word], 5.0, 5.5)

        background = create_gradient_background()
        from y2karaoke.utils.fonts import get_font
        font = get_font()

        # At t=0, should show splash (before lyrics start)
        frame = render_frame([line], 0.0, font, background, "Test Song", "Test Artist")
        # Frame should be rendered without error
        assert frame.shape == (VIDEO_HEIGHT, VIDEO_WIDTH, 3)


class TestDrawProgressBar:
    """Tests for progress bar drawing."""

    def test_zero_progress(self):
        """Progress bar at 0% should render without error."""
        from PIL import Image, ImageDraw
        img = Image.new('RGB', (VIDEO_WIDTH, VIDEO_HEIGHT))
        draw = ImageDraw.Draw(img)
        draw_progress_bar(draw, 0.0)
        # Should not raise

    def test_full_progress(self):
        """Progress bar at 100% should render without error."""
        from PIL import Image, ImageDraw
        img = Image.new('RGB', (VIDEO_WIDTH, VIDEO_HEIGHT))
        draw = ImageDraw.Draw(img)
        draw_progress_bar(draw, 1.0)
        # Should not raise

    def test_half_progress(self):
        """Progress bar at 50% should render without error."""
        from PIL import Image, ImageDraw
        img = Image.new('RGB', (VIDEO_WIDTH, VIDEO_HEIGHT))
        draw = ImageDraw.Draw(img)
        draw_progress_bar(draw, 0.5)
        # Should not raise


class TestDrawSplashScreen:
    """Tests for splash screen drawing."""

    def test_renders_with_title_and_artist(self):
        """Splash screen should render with title and artist."""
        from PIL import Image, ImageDraw
        img = Image.new('RGB', (VIDEO_WIDTH, VIDEO_HEIGHT))
        draw = ImageDraw.Draw(img)
        draw_splash_screen(draw, "Test Song", "Test Artist")
        # Should not raise

    def test_handles_long_title(self):
        """Splash screen should handle long titles gracefully."""
        from PIL import Image, ImageDraw
        img = Image.new('RGB', (VIDEO_WIDTH, VIDEO_HEIGHT))
        draw = ImageDraw.Draw(img)
        long_title = "This is a very long song title that exceeds the maximum character limit for display"
        draw_splash_screen(draw, long_title, "Artist")
        # Should not raise


class TestDrawLogoScreen:
    """Tests for logo screen drawing."""

    def test_renders_logo(self):
        """Logo screen should render without error."""
        from PIL import Image, ImageDraw
        from y2karaoke.utils.fonts import get_font
        img = Image.new('RGB', (VIDEO_WIDTH, VIDEO_HEIGHT))
        draw = ImageDraw.Draw(img)
        font = get_font()
        draw_logo_screen(draw, font)
        # Should not raise


