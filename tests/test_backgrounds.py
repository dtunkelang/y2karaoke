"""Tests for the backgrounds module."""

import pytest
import numpy as np
from dataclasses import dataclass
from unittest.mock import patch, MagicMock

from y2karaoke.core.backgrounds import (
    BackgroundSegment,
    BackgroundProcessor,
    create_background_segments,
)
from y2karaoke.config import VIDEO_WIDTH, VIDEO_HEIGHT


def get_background_at_time(segments: list, time: float):
    """Helper to get background image at a specific time."""
    for segment in segments:
        if segment.start_time <= time < segment.end_time:
            return segment.image
    return None


class TestBackgroundSegment:
    """Tests for the BackgroundSegment dataclass."""

    def test_segment_creation(self):
        """BackgroundSegment should be created with required fields."""
        image = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH, 3), dtype=np.uint8)
        segment = BackgroundSegment(
            image=image,
            start_time=0.0,
            end_time=10.0,
        )
        assert segment.start_time == 0.0
        assert segment.end_time == 10.0
        assert segment.image is not None

    def test_segment_image_shape(self):
        """BackgroundSegment image should have correct shape."""
        image = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH, 3), dtype=np.uint8)
        segment = BackgroundSegment(
            image=image,
            start_time=0.0,
            end_time=10.0,
        )
        assert segment.image.shape == (VIDEO_HEIGHT, VIDEO_WIDTH, 3)


class TestGetBackgroundAtTime:
    """Tests for get_background_at_time helper function."""

    def test_returns_correct_segment(self):
        """Should return image for the correct time segment."""
        image1 = np.ones((VIDEO_HEIGHT, VIDEO_WIDTH, 3), dtype=np.uint8) * 100
        image2 = np.ones((VIDEO_HEIGHT, VIDEO_WIDTH, 3), dtype=np.uint8) * 200

        segments = [
            BackgroundSegment(image=image1, start_time=0.0, end_time=5.0),
            BackgroundSegment(image=image2, start_time=5.0, end_time=10.0),
        ]

        # At t=2, should get image1
        result = get_background_at_time(segments, 2.0)
        assert result is not None
        assert np.mean(result) == 100

        # At t=7, should get image2
        result = get_background_at_time(segments, 7.0)
        assert result is not None
        assert np.mean(result) == 200

    def test_returns_none_for_empty_segments(self):
        """Should return None if no segments provided."""
        result = get_background_at_time([], 5.0)
        assert result is None

    def test_returns_none_before_first_segment(self):
        """Should return None if time is before first segment."""
        image = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH, 3), dtype=np.uint8)
        segments = [
            BackgroundSegment(image=image, start_time=5.0, end_time=10.0),
        ]

        result = get_background_at_time(segments, 2.0)
        assert result is None

    def test_handles_boundary_times(self):
        """Should handle segment boundary times correctly."""
        image = np.ones((VIDEO_HEIGHT, VIDEO_WIDTH, 3), dtype=np.uint8) * 150
        segments = [
            BackgroundSegment(image=image, start_time=0.0, end_time=5.0),
        ]

        # At exactly start time
        result = get_background_at_time(segments, 0.0)
        assert result is not None

        # At exactly end time (should be None - exclusive end)
        result = get_background_at_time(segments, 5.0)
        assert result is None


class TestBackgroundProcessor:
    """Tests for the BackgroundProcessor class."""

    def test_init_sets_dimensions(self):
        """BackgroundProcessor should initialize with video dimensions."""
        processor = BackgroundProcessor()
        assert processor.width == VIDEO_WIDTH
        assert processor.height == VIDEO_HEIGHT

    def test_is_valid_frame_bright_frame(self):
        """Bright frames should be considered valid."""
        processor = BackgroundProcessor()
        bright_frame = np.ones((100, 100, 3), dtype=np.uint8) * 200
        assert processor._is_valid_frame(bright_frame) == True

    def test_is_valid_frame_dark_frame(self):
        """Very dark frames should be considered invalid."""
        processor = BackgroundProcessor()
        dark_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        assert processor._is_valid_frame(dark_frame, min_brightness=20) == False

    def test_is_valid_frame_threshold(self):
        """Frame validity should respect brightness threshold."""
        processor = BackgroundProcessor()

        # Frame just below threshold
        dim_frame = np.ones((100, 100, 3), dtype=np.uint8) * 15
        assert processor._is_valid_frame(dim_frame, min_brightness=20) == False

        # Frame at threshold
        threshold_frame = np.ones((100, 100, 3), dtype=np.uint8) * 20
        assert processor._is_valid_frame(threshold_frame, min_brightness=20) == True


class TestBackgroundProcessorIntegration:
    """Integration tests for BackgroundProcessor (require video file)."""

    @pytest.mark.skip(reason="Requires actual video file")
    def test_create_background_segments(self):
        """Should create background segments from video."""
        processor = BackgroundProcessor()
        segments = processor.create_background_segments(
            video_path="test_video.mp4",
            lines=[],
            duration=10.0
        )
        assert isinstance(segments, list)
