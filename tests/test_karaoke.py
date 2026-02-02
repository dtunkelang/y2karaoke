"""Tests for karaoke.py - main orchestrator (core functionality only)."""

import pytest
from unittest.mock import MagicMock
from pathlib import Path

from y2karaoke.core.karaoke import KaraokeGenerator
from y2karaoke.core.models import Word, Line
from y2karaoke.exceptions import DownloadError


class TestKaraokeGenerator:
    def test_init_default_cache_dir(self):
        generator = KaraokeGenerator()
        assert generator.cache_manager is not None
        assert generator.downloader is not None
        assert generator.separator is not None
        assert generator.audio_processor is not None

    def test_init_custom_cache_dir(self, tmp_path):
        generator = KaraokeGenerator(cache_dir=tmp_path)
        assert generator.cache_manager is not None
        assert generator.cache_dir == tmp_path

    def test_scale_lyrics_timing(self):
        words = [Word("hello", 1.0, 1.5), Word("world", 2.0, 2.5)]
        lines = [Line(words)]
        
        generator = KaraokeGenerator()
        scaled_lines = generator._scale_lyrics_timing(lines, 0.8)  # 80% speed
        
        # Times should be scaled by 1/0.8 = 1.25
        assert scaled_lines[0].words[0].start_time == pytest.approx(1.25, abs=1e-6)
        assert scaled_lines[0].words[0].end_time == pytest.approx(1.875, abs=1e-6)
        assert scaled_lines[0].words[1].start_time == pytest.approx(2.5, abs=1e-6)
        assert scaled_lines[0].words[1].end_time == pytest.approx(3.125, abs=1e-6)

    def test_scale_lyrics_timing_no_change(self):
        words = [Word("hello", 1.0, 1.5)]
        lines = [Line(words)]
        
        generator = KaraokeGenerator()
        scaled_lines = generator._scale_lyrics_timing(lines, 1.0)  # No change
        
        assert scaled_lines[0].words[0].start_time == 1.0
        assert scaled_lines[0].words[0].end_time == 1.5

    def test_scale_lyrics_timing_speed_up(self):
        words = [Word("hello", 2.0, 3.0)]
        lines = [Line(words)]
        
        generator = KaraokeGenerator()
        scaled_lines = generator._scale_lyrics_timing(lines, 2.0)  # Double speed
        
        # Times should be scaled by 1/2.0 = 0.5
        assert scaled_lines[0].words[0].start_time == pytest.approx(1.0, abs=1e-6)
        assert scaled_lines[0].words[0].end_time == pytest.approx(1.5, abs=1e-6)

    def test_download_audio_success(self):
        generator = KaraokeGenerator()
        
        # Mock the downloader
        generator.downloader = MagicMock()
        generator.downloader.download_audio.return_value = {
            "audio_path": "/tmp/audio.wav",
            "title": "Test Song",
            "artist": "Test Artist",
            "duration": 180
        }
        
        result = generator._download_audio("test_id", "https://youtube.com/watch?v=test_id", False)
        
        assert result["audio_path"] == "/tmp/audio.wav"
        assert result["title"] == "Test Song"
        assert result["duration"] == 180
        generator.downloader.download_audio.assert_called_once()

    def test_download_video_success(self):
        generator = KaraokeGenerator()
        
        # Mock the downloader
        generator.downloader = MagicMock()
        generator.downloader.download_video.return_value = {
            "video_path": "/tmp/video.mp4"
        }
        
        result = generator._download_video("test_id", "https://youtube.com/watch?v=test_id", False)
        
        assert result["video_path"] == "/tmp/video.mp4"
        generator.downloader.download_video.assert_called_once()

    def test_download_audio_with_cache(self):
        generator = KaraokeGenerator()
        
        # Mock the downloader and cache manager
        generator.downloader = MagicMock()
        generator.cache_manager = MagicMock()
        generator.cache_manager.get_video_cache_dir.return_value = "/tmp/cache"
        generator.downloader.download_audio.return_value = {
            "audio_path": "/tmp/audio.wav",
            "title": "Test Song",
            "artist": "Test Artist",
            "duration": 180
        }
        
        result = generator._download_audio("test_id", "https://youtube.com/watch?v=test_id", False)
        
        assert result["audio_path"] == "/tmp/audio.wav"
        generator.cache_manager.save_metadata.assert_called_once()

    def test_audio_processor_integration(self):
        generator = KaraokeGenerator()
        
        # Mock the audio processor
        generator.audio_processor = MagicMock()
        generator.audio_processor.process_audio.return_value = "/tmp/processed.wav"
        
        # Test that the audio processor is available
        assert generator.audio_processor is not None
        
        # Test processing
        result = generator.audio_processor.process_audio(
            "/tmp/input.wav", "/tmp/output.wav", 2, 0.9
        )
        assert result == "/tmp/processed.wav"

    def test_temp_files_tracking(self):
        generator = KaraokeGenerator()
        
        # Initially empty
        assert len(generator._temp_files) == 0
        
        # Add some temp files
        generator._temp_files.append("/tmp/test1.wav")
        generator._temp_files.append("/tmp/test2.wav")
        
        assert len(generator._temp_files) == 2
        assert "/tmp/test1.wav" in generator._temp_files

    def test_cleanup_temp_files_empty_list(self):
        generator = KaraokeGenerator()
        generator._temp_files = []
        
        # Should not raise any errors
        generator.cleanup_temp_files()

    def test_separator_integration(self):
        generator = KaraokeGenerator()
        
        # Mock the separator
        generator.separator = MagicMock()
        
        # Test that the separator is available
        assert generator.separator is not None

    def test_cache_manager_integration(self):
        generator = KaraokeGenerator()
        
        # Test that cache manager is properly initialized
        assert generator.cache_manager is not None
        assert hasattr(generator.cache_manager, 'get_video_cache_dir')
        assert hasattr(generator.cache_manager, 'save_metadata')
        assert hasattr(generator.cache_manager, 'load_metadata')

    def test_component_initialization(self):
        """Test that all required components are properly initialized."""
        generator = KaraokeGenerator()
        
        # Check all components exist
        assert hasattr(generator, 'cache_dir')
        assert hasattr(generator, 'cache_manager')
        assert hasattr(generator, 'downloader')
        assert hasattr(generator, 'separator')
        assert hasattr(generator, 'audio_processor')
        assert hasattr(generator, '_temp_files')
        
        # Check they're not None
        assert generator.cache_dir is not None
        assert generator.cache_manager is not None
        assert generator.downloader is not None
        assert generator.separator is not None
        assert generator.audio_processor is not None
        assert generator._temp_files is not None
