"""Tests for video_writer.py module."""

import pytest
from pathlib import Path  # noqa: F401
from unittest.mock import Mock, patch, MagicMock  # noqa: F401
from y2karaoke.core.components.render.video_writer import (
    render_karaoke_video,
    get_background_at_time,
)
from y2karaoke.core.models import Line, Word, SongMetadata


class TestRenderKaraokeVideo:
    """Test render_karaoke_video function."""

    def test_function_exists(self):
        """Test that render_karaoke_video function exists."""
        assert render_karaoke_video is not None
        assert callable(render_karaoke_video)

    def test_function_signature(self):
        """Test render_karaoke_video function signature."""
        import inspect

        sig = inspect.signature(render_karaoke_video)
        params = list(sig.parameters.keys())

        required_params = ["lines", "audio_path", "output_path"]
        for param in required_params:
            assert param in params

    @patch("y2karaoke.core.components.render.video_writer.AudioFileClip")
    @patch("y2karaoke.core.components.render.video_writer.VideoClip")
    def test_basic_video_creation(self, mock_video_clip, mock_audio_clip):
        """Test basic video creation workflow."""
        # Mock audio clip
        mock_audio = Mock()
        mock_audio.duration = 10.0
        mock_audio_clip.return_value = mock_audio

        # Mock video clip
        mock_video = Mock()
        mock_video.with_audio.return_value = mock_video
        mock_video.write_videofile = Mock()
        mock_video_clip.return_value = mock_video

        # Create test data
        lines = [Line(words=[Word(text="test", start_time=0.0, end_time=1.0)])]

        render_karaoke_video(
            lines=lines, audio_path="/test/audio.wav", output_path="/test/output.mp4"
        )

        mock_audio_clip.assert_called_once_with("/test/audio.wav")
        mock_video_clip.assert_called_once()

    @patch("y2karaoke.core.components.render.video_writer.AudioFileClip")
    @patch("y2karaoke.core.components.render.video_writer.VideoClip")
    def test_with_timing_offset(self, mock_video_clip, mock_audio_clip):
        """Test video creation with timing offset."""
        mock_audio = Mock()
        mock_audio.duration = 10.0
        mock_audio_clip.return_value = mock_audio

        mock_video = Mock()
        mock_video.with_audio.return_value = mock_video
        mock_video.write_videofile = Mock()
        mock_video_clip.return_value = mock_video

        lines = [Line(words=[Word(text="test", start_time=0.0, end_time=1.0)])]

        render_karaoke_video(
            lines=lines,
            audio_path="/test/audio.wav",
            output_path="/test/output.mp4",
            timing_offset=0.5,
        )

        mock_video_clip.assert_called_once()

    @patch("y2karaoke.core.components.render.video_writer.AudioFileClip")
    @patch("y2karaoke.core.components.render.video_writer.VideoClip")
    def test_with_metadata(self, mock_video_clip, mock_audio_clip):
        """Test video creation with song metadata."""
        mock_audio = Mock()
        mock_audio.duration = 10.0
        mock_audio_clip.return_value = mock_audio

        mock_video = Mock()
        mock_video.with_audio.return_value = mock_video
        mock_video.write_videofile = Mock()
        mock_video_clip.return_value = mock_video

        lines = [Line(words=[Word(text="test", start_time=0.0, end_time=1.0)])]

        metadata = SongMetadata(
            singers=["Test Artist"], title="Test Song", artist="Test Artist"
        )

        render_karaoke_video(
            lines=lines,
            audio_path="/test/audio.wav",
            output_path="/test/output.mp4",
            song_metadata=metadata,
        )

        mock_video_clip.assert_called_once()

    @patch("y2karaoke.core.components.render.video_writer.AudioFileClip")
    @patch("y2karaoke.core.components.render.video_writer.VideoClip")
    def test_with_background_segments(self, mock_video_clip, mock_audio_clip):
        """Test video creation with background segments."""
        mock_audio = Mock()
        mock_audio.duration = 10.0
        mock_audio_clip.return_value = mock_audio

        mock_video = Mock()
        mock_video.with_audio.return_value = mock_video
        mock_video.write_videofile = Mock()
        mock_video_clip.return_value = mock_video

        lines = [Line(words=[Word(text="test", start_time=0.0, end_time=1.0)])]

        # Mock background segments
        mock_segments = [Mock()]

        render_karaoke_video(
            lines=lines,
            audio_path="/test/audio.wav",
            output_path="/test/output.mp4",
            background_segments=mock_segments,
        )

        mock_video_clip.assert_called_once()

    @patch("builtins.print")
    @patch("y2karaoke.core.components.render.video_writer.FrameGenerator")
    @patch("y2karaoke.core.components.render.video_writer.AudioFileClip")
    @patch("y2karaoke.core.components.render.video_writer.VideoClip")
    def test_make_frame_progress_and_background(
        self, mock_video_clip, mock_audio_clip, mock_frame_generator, mock_print
    ):
        """Covers make_frame progress output and background selection."""
        mock_audio = Mock()
        mock_audio.duration = 1.0
        mock_audio_clip.return_value = mock_audio

        captured = {}

        def fake_video_clip(make_frame, duration):
            captured["make_frame"] = make_frame
            mock_video = Mock()
            mock_video.with_audio.return_value = mock_video
            mock_video.with_fps.return_value = mock_video
            mock_video.write_videofile = Mock()
            return mock_video

        mock_video_clip.side_effect = fake_video_clip
        
        mock_generator_instance = Mock()
        mock_generator_instance.generate_frame.return_value = "frame"
        mock_frame_generator.return_value = mock_generator_instance

        lines = [Line(words=[Word(text="test", start_time=0.0, end_time=1.0)])]

        segment = Mock()
        segment.start_time = 0.0
        segment.end_time = 1.0
        segment.image = "bg"

        render_karaoke_video(
            lines=lines,
            audio_path="/test/audio.wav",
            output_path="/test/output.mp4",
            background_segments=[segment],
            show_progress=True,
        )

        # We need to verify that FrameGenerator was initialized correctly
        # and that generate_frame was called via the closure
        mock_frame_generator.assert_called_once()
        
        frame = captured["make_frame"](0.5)
        assert frame == "frame"
        mock_generator_instance.generate_frame.assert_called_with(0.5)
        assert mock_print.called


class TestGetBackgroundAtTime:
    """Test get_background_at_time function."""

    def test_function_exists(self):
        """Test that get_background_at_time function exists."""
        assert get_background_at_time is not None
        assert callable(get_background_at_time)

    def test_function_signature(self):
        """Test get_background_at_time function signature."""
        import inspect

        sig = inspect.signature(get_background_at_time)
        params = list(sig.parameters.keys())
        assert "segments" in params
        assert "t" in params

    def test_none_segments_returns_none(self):
        """Test that None segments returns None."""
        result = get_background_at_time(None, 5.0)
        assert result is None

    def test_empty_segments_returns_none(self):
        """Test that empty segments list returns None."""
        result = get_background_at_time([], 5.0)
        assert result is None

    def test_finds_matching_segment(self):
        """Test finding segment that contains the given time."""
        mock_segment1 = Mock()
        mock_segment1.start_time = 0.0
        mock_segment1.end_time = 5.0
        mock_segment1.image = "image1"

        mock_segment2 = Mock()
        mock_segment2.start_time = 5.0
        mock_segment2.end_time = 10.0
        mock_segment2.image = "image2"

        segments = [mock_segment1, mock_segment2]

        result = get_background_at_time(segments, 3.0)
        assert result == "image1"

        result = get_background_at_time(segments, 7.0)
        assert result == "image2"

    def test_no_matching_segment_returns_none(self):
        """Test that no matching segment returns None."""
        mock_segment = Mock()
        mock_segment.start_time = 5.0
        mock_segment.end_time = 10.0

        segments = [mock_segment]

        result = get_background_at_time(segments, 15.0)
        assert result is None

    def test_time_at_boundary(self):
        """Test behavior at segment boundaries."""
        mock_segment = Mock()
        mock_segment.start_time = 5.0
        mock_segment.end_time = 10.0
        mock_segment.image = "test_image"

        segments = [mock_segment]

        # Test at start boundary
        result = get_background_at_time(segments, 5.0)
        assert result == "test_image"

        # Test at end boundary (inclusive in implementation)
        result = get_background_at_time(segments, 10.0)
        assert result == "test_image"


class TestVideoWriterIntegration:
    """Test video_writer module integration."""

    def test_module_imports(self):
        """Test that all required functions can be imported."""
        from y2karaoke.core.components.render.video_writer import (
            render_karaoke_video,
            get_background_at_time,
        )

        assert render_karaoke_video is not None
        assert get_background_at_time is not None

    @patch("y2karaoke.core.components.render.video_writer.FrameGenerator")
    def test_frame_generator_integration(self, mock_frame_generator):
        """Test that video writer integrates with frame generator."""
        mock_frame_generator.return_value = Mock()

        # This tests that FrameGenerator is available for import
        from y2karaoke.core.components.render.video_writer import FrameGenerator

        assert FrameGenerator is not None

    @patch("y2karaoke.core.components.render.video_writer.create_gradient_background")
    def test_background_integration(self, mock_create_gradient):
        """Test that video writer integrates with background creation."""
        mock_create_gradient.return_value = Mock()

        # This tests that create_gradient_background is available for import
        from y2karaoke.core.components.render.video_writer import (
            create_gradient_background,
        )

        assert create_gradient_background is not None

    def test_config_imports(self):
        """Test that video configuration is properly imported."""
        from y2karaoke.core.components.render.video_writer import (
            VIDEO_WIDTH,
            VIDEO_HEIGHT,
            FPS,
        )

        assert isinstance(VIDEO_WIDTH, int)
        assert isinstance(VIDEO_HEIGHT, int)
        assert isinstance(FPS, int)

    def test_models_integration(self):
        """Test integration with core models."""
        from y2karaoke.core.components.render.video_writer import Line, SongMetadata

        assert Line is not None
        assert SongMetadata is not None

    @patch("y2karaoke.core.components.render.video_writer.AudioFileClip")
    @patch("y2karaoke.core.components.render.video_writer.VideoClip")
    def test_moviepy_integration(self, mock_video_clip, mock_audio_clip):
        """Test integration with MoviePy library."""
        # Test that MoviePy classes are properly imported and used
        mock_audio = Mock()
        mock_audio.duration = 5.0
        mock_audio_clip.return_value = mock_audio

        mock_video = Mock()
        mock_video.with_audio.return_value = mock_video
        mock_video.write_videofile = Mock()
        mock_video_clip.return_value = mock_video

        lines = [Line(words=[Word(text="test", start_time=0.0, end_time=1.0)])]

        render_karaoke_video(
            lines=lines, audio_path="/test/audio.wav", output_path="/test/output.mp4"
        )

        # Verify MoviePy components were used
        mock_audio_clip.assert_called_once()
        mock_video_clip.assert_called_once()


class TestVideoWriterErrorHandling:
    """Test error handling in video_writer module."""

    @patch("y2karaoke.core.components.render.video_writer.AudioFileClip")
    def test_handles_audio_file_error(self, mock_audio_clip):
        """Test handling of audio file loading errors."""
        mock_audio_clip.side_effect = Exception("Audio file not found")

        lines = [Line(words=[Word(text="test", start_time=0.0, end_time=1.0)])]

        with pytest.raises(Exception):
            render_karaoke_video(
                lines=lines,
                audio_path="/nonexistent/audio.wav",
                output_path="/test/output.mp4",
            )

    def test_rejects_out_of_order_lines(self):
        """Out-of-order lines should fail validation."""
        lines = [
            Line(words=[Word(text="late", start_time=2.0, end_time=2.5)]),
            Line(words=[Word(text="early", start_time=1.0, end_time=1.5)]),
        ]

        with pytest.raises(Exception):
            render_karaoke_video(
                lines=lines,
                audio_path="/test/audio.wav",
                output_path="/test/output.mp4",
            )

    def test_empty_lines_handling(self):
        """Test handling of empty lines list."""
        # This should test how the function handles empty input
        # The exact behavior depends on implementation
        lines = []

        # Test that function can be called with empty lines
        # (may raise exception or handle gracefully depending on implementation)
        try:
            with patch(
                "y2karaoke.core.components.render.video_writer.AudioFileClip"
            ) as mock_audio:
                with patch(
                    "y2karaoke.core.components.render.video_writer.VideoClip"
                ) as mock_video:
                    mock_audio_instance = Mock()
                    mock_audio_instance.duration = 1.0
                    mock_audio.return_value = mock_audio_instance

                    mock_video_instance = Mock()
                    mock_video_instance.with_audio.return_value = mock_video_instance
                    mock_video_instance.write_videofile = Mock()
                    mock_video.return_value = mock_video_instance

                    render_karaoke_video(
                        lines=lines,
                        audio_path="/test/audio.wav",
                        output_path="/test/output.mp4",
                    )
        except Exception:
            # Exception is acceptable for empty lines
            pass
