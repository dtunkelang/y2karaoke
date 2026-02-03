"""Tests for downloader.py module."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from y2karaoke.core.downloader import YouTubeDownloader, download_audio, download_video
from y2karaoke.exceptions import DownloadError


class TestYouTubeDownloader:
    """Test YouTubeDownloader class."""

    def test_init_default_cache_dir(self):
        """Test initialization with default cache directory."""
        with patch("y2karaoke.core.downloader.get_cache_dir") as mock_get_cache:
            mock_get_cache.return_value = Path("/test/cache")
            downloader = YouTubeDownloader()
            assert downloader.cache_dir == Path("/test/cache")

    def test_init_custom_cache_dir(self):
        """Test initialization with custom cache directory."""
        custom_dir = Path("/custom/cache")
        downloader = YouTubeDownloader(cache_dir=custom_dir)
        assert downloader.cache_dir == custom_dir

    @patch("y2karaoke.core.downloader.yt_dlp.YoutubeDL")
    def test_get_video_title_success(self, mock_ytdl):
        """Test successful video title extraction."""
        mock_instance = Mock()
        mock_instance.extract_info.return_value = {"title": "Test Video Title"}
        mock_ytdl.return_value.__enter__.return_value = mock_instance

        downloader = YouTubeDownloader()
        title = downloader.get_video_title("https://youtube.com/watch?v=test")
        assert title == "Test Video Title"

    @patch("y2karaoke.core.downloader.yt_dlp.YoutubeDL")
    def test_get_video_title_no_title(self, mock_ytdl):
        """Test video title extraction when no title in metadata."""
        mock_instance = Mock()
        mock_instance.extract_info.return_value = {}
        mock_ytdl.return_value.__enter__.return_value = mock_instance

        downloader = YouTubeDownloader()
        title = downloader.get_video_title("https://youtube.com/watch?v=test")
        assert title == "https://youtube.com/watch?v=test"  # Returns URL when no title

    @patch("y2karaoke.core.downloader.yt_dlp.YoutubeDL")
    def test_get_video_uploader_success(self, mock_ytdl):
        """Test successful video uploader extraction."""
        mock_instance = Mock()
        mock_instance.extract_info.return_value = {"uploader": "Test Channel"}
        mock_ytdl.return_value.__enter__.return_value = mock_instance

        downloader = YouTubeDownloader()
        uploader = downloader.get_video_uploader("https://youtube.com/watch?v=test")
        assert uploader == "Test Channel"

    @patch("y2karaoke.core.downloader.yt_dlp.YoutubeDL")
    def test_get_video_uploader_no_uploader(self, mock_ytdl):
        """Test video uploader extraction when no uploader in metadata."""
        mock_instance = Mock()
        mock_instance.extract_info.return_value = {}
        mock_ytdl.return_value.__enter__.return_value = mock_instance

        downloader = YouTubeDownloader()
        uploader = downloader.get_video_uploader("https://youtube.com/watch?v=test")
        assert uploader == "Unknown"

    @patch("y2karaoke.core.downloader.yt_dlp.YoutubeDL")
    def test_get_video_uploader_exception_handling(self, mock_ytdl):
        """Test video uploader extraction handles exceptions."""
        mock_ytdl.side_effect = Exception("Network error")

        downloader = YouTubeDownloader()
        uploader = downloader.get_video_uploader("https://youtube.com/watch?v=test")
        assert uploader == "Unknown"


class TestDownloadAudio:
    """Test audio download functionality."""

    def test_download_audio_method_exists(self):
        """Test that download_audio method exists and has correct signature."""
        downloader = YouTubeDownloader()
        assert hasattr(downloader, "download_audio")
        assert callable(downloader.download_audio)

    @patch("y2karaoke.core.downloader.validate_youtube_url")
    def test_download_audio_validates_url(self, mock_validate):
        """Test that download_audio validates YouTube URL."""
        mock_validate.side_effect = DownloadError("Invalid URL")
        downloader = YouTubeDownloader()

        with pytest.raises(DownloadError):
            downloader.download_audio("invalid_url")

        mock_validate.assert_called_once_with("invalid_url")

    @patch("y2karaoke.core.downloader.validate_youtube_url")
    @patch("y2karaoke.core.downloader.extract_video_id")
    def test_download_audio_calls_validation(self, mock_extract_id, mock_validate):
        """Test that download_audio calls URL validation."""
        mock_extract_id.return_value = "test_video_id"
        mock_validate.return_value = "https://youtube.com/watch?v=test"

        downloader = YouTubeDownloader()

        # This will fail at the yt-dlp stage, but we just want to test validation is called
        try:
            downloader.download_audio("https://youtube.com/watch?v=test")
        except DownloadError:
            pass  # Expected to fail at download stage

        mock_validate.assert_called_once_with("https://youtube.com/watch?v=test")
        mock_extract_id.assert_called_once_with("https://youtube.com/watch?v=test")

    @patch("y2karaoke.core.downloader.extract_metadata_from_youtube")
    @patch("y2karaoke.core.downloader.sanitize_filename")
    @patch("y2karaoke.core.downloader.yt_dlp.YoutubeDL")
    @patch("y2karaoke.core.downloader.extract_video_id")
    @patch("y2karaoke.core.downloader.validate_youtube_url")
    def test_download_audio_falls_back_to_any_wav(
        self,
        mock_validate,
        mock_extract_id,
        mock_ytdl,
        mock_sanitize,
        mock_extract_metadata,
        tmp_path,
    ):
        """Fallback to any wav if sanitized title path missing."""
        mock_validate.return_value = "https://youtube.com/watch?v=test"
        mock_extract_id.return_value = "video123"
        mock_sanitize.return_value = "Expected"
        mock_extract_metadata.return_value = {"artist": "Artist", "title": "Clean"}

        mock_instance = Mock()
        mock_instance.extract_info.return_value = {"title": "Original Title"}
        mock_ytdl.return_value.__enter__.return_value = mock_instance

        fallback_audio = tmp_path / "fallback.wav"
        fallback_audio.write_text("audio")

        downloader = YouTubeDownloader()
        result = downloader.download_audio(
            "https://youtube.com/watch?v=test",
            output_dir=tmp_path,
        )

        assert result["audio_path"] == str(fallback_audio)
        assert result["title"] == "Clean"
        assert result["artist"] == "Artist"
        assert result["video_id"] == "video123"
        mock_instance.download.assert_called_once_with(
            ["https://youtube.com/watch?v=test"]
        )

    @patch("y2karaoke.core.downloader.sanitize_filename")
    @patch("y2karaoke.core.downloader.yt_dlp.YoutubeDL")
    @patch("y2karaoke.core.downloader.extract_video_id")
    @patch("y2karaoke.core.downloader.validate_youtube_url")
    def test_download_audio_raises_when_no_wav(
        self,
        mock_validate,
        mock_extract_id,
        mock_ytdl,
        mock_sanitize,
        tmp_path,
    ):
        """Raise DownloadError when no wav output exists."""
        mock_validate.return_value = "https://youtube.com/watch?v=test"
        mock_extract_id.return_value = "video123"
        mock_sanitize.return_value = "Expected"

        mock_instance = Mock()
        mock_instance.extract_info.return_value = {"title": "Original Title"}
        mock_ytdl.return_value.__enter__.return_value = mock_instance

        downloader = YouTubeDownloader()
        with pytest.raises(DownloadError) as excinfo:
            downloader.download_audio(
                "https://youtube.com/watch?v=test",
                output_dir=tmp_path,
            )

        assert "Downloaded audio file not found" in str(excinfo.value)


class TestDownloadVideo:
    """Test video download functionality."""

    def test_download_video_method_exists(self):
        """Test that download_video method exists and has correct signature."""
        downloader = YouTubeDownloader()
        assert hasattr(downloader, "download_video")
        assert callable(downloader.download_video)

    @patch("y2karaoke.core.downloader.validate_youtube_url")
    def test_download_video_validates_url(self, mock_validate):
        """Test that download_video validates YouTube URL."""
        mock_validate.side_effect = DownloadError("Invalid URL")
        downloader = YouTubeDownloader()

        with pytest.raises(DownloadError):
            downloader.download_video("invalid_url")

        mock_validate.assert_called_once_with("invalid_url")

    @patch("y2karaoke.core.downloader.extract_metadata_from_youtube")
    @patch("y2karaoke.core.downloader.yt_dlp.YoutubeDL")
    @patch("y2karaoke.core.downloader.extract_video_id")
    @patch("y2karaoke.core.downloader.validate_youtube_url")
    def test_download_video_returns_metadata_when_available(
        self,
        mock_validate,
        mock_extract_id,
        mock_ytdl,
        mock_extract_metadata,
        tmp_path,
    ):
        """Return cleaned metadata when available."""
        mock_validate.return_value = "https://youtube.com/watch?v=test"
        mock_extract_id.return_value = "video123"
        mock_extract_metadata.return_value = {"artist": "Artist", "title": "Clean"}

        mock_instance = Mock()
        mock_instance.extract_info.return_value = {"title": "Original Title"}
        mock_ytdl.return_value.__enter__.return_value = mock_instance

        video_file = tmp_path / "clip_video.mp4"
        video_file.write_text("video")

        downloader = YouTubeDownloader()
        result = downloader.download_video(
            "https://youtube.com/watch?v=test",
            output_dir=tmp_path,
        )

        assert result["video_path"] == str(video_file)
        assert result["title"] == "Clean"
        assert result["artist"] == "Artist"
        assert result["video_id"] == "video123"

    @patch("y2karaoke.core.downloader.extract_metadata_from_youtube")
    @patch("y2karaoke.core.downloader.yt_dlp.YoutubeDL")
    @patch("y2karaoke.core.downloader.extract_video_id")
    @patch("y2karaoke.core.downloader.validate_youtube_url")
    def test_download_video_falls_back_to_info_title_and_unknown_artist(
        self,
        mock_validate,
        mock_extract_id,
        mock_ytdl,
        mock_extract_metadata,
        tmp_path,
    ):
        """Fall back to info title and Unknown artist when metadata empty."""
        mock_validate.return_value = "https://youtube.com/watch?v=test"
        mock_extract_id.return_value = "video123"
        mock_extract_metadata.return_value = {"artist": "", "title": ""}

        mock_instance = Mock()
        mock_instance.extract_info.return_value = {"title": "Original Title"}
        mock_ytdl.return_value.__enter__.return_value = mock_instance

        video_file = tmp_path / "clip_video.webm"
        video_file.write_text("video")

        downloader = YouTubeDownloader()
        result = downloader.download_video(
            "https://youtube.com/watch?v=test",
            output_dir=tmp_path,
        )

        assert result["video_path"] == str(video_file)
        assert result["title"] == "Original Title"
        assert result["artist"] == "Unknown"
        assert result["video_id"] == "video123"


class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    @patch("y2karaoke.core.downloader.YouTubeDownloader")
    def test_download_audio_function_exists(self, mock_downloader_class):
        """Test that download_audio function exists."""
        mock_instance = Mock()
        mock_instance.download_audio.return_value = {"audio_path": "/test/audio.wav"}
        mock_downloader_class.return_value = mock_instance

        result = download_audio("https://youtube.com/watch?v=test")
        assert result == {"audio_path": "/test/audio.wav"}
        mock_downloader_class.assert_called_once()
        mock_instance.download_audio.assert_called_once_with(
            "https://youtube.com/watch?v=test", None
        )

    @patch("y2karaoke.core.downloader.YouTubeDownloader")
    def test_download_video_function_exists(self, mock_downloader_class):
        """Test that download_video function exists."""
        mock_instance = Mock()
        mock_instance.download_video.return_value = {"video_path": "/test/video.mp4"}
        mock_downloader_class.return_value = mock_instance

        result = download_video("https://youtube.com/watch?v=test")
        assert result == {"video_path": "/test/video.mp4"}
        mock_downloader_class.assert_called_once()
        mock_instance.download_video.assert_called_once_with(
            "https://youtube.com/watch?v=test", None
        )

    def test_download_audio_function_signature(self):
        """Test download_audio function signature."""
        import inspect

        sig = inspect.signature(download_audio)
        params = list(sig.parameters.keys())
        assert "url" in params
        assert "output_dir" in params

    def test_download_video_function_signature(self):
        """Test download_video function signature."""
        import inspect

        sig = inspect.signature(download_video)
        params = list(sig.parameters.keys())
        assert "url" in params
        assert "output_dir" in params


class TestModuleIntegration:
    """Test module-level integration and imports."""

    def test_module_imports(self):
        """Test that all required classes and functions can be imported."""
        from y2karaoke.core.downloader import (
            YouTubeDownloader,
            download_audio,
            download_video,
        )

        assert YouTubeDownloader is not None
        assert download_audio is not None
        assert download_video is not None

    def test_downloader_initialization(self):
        """Test YouTubeDownloader can be initialized."""
        downloader = YouTubeDownloader()
        assert downloader is not None
        assert hasattr(downloader, "cache_dir")

    def test_downloader_has_required_methods(self):
        """Test YouTubeDownloader has all required methods."""
        downloader = YouTubeDownloader()
        required_methods = [
            "get_video_title",
            "get_video_uploader",
            "download_audio",
            "download_video",
        ]

        for method in required_methods:
            assert hasattr(downloader, method)
            assert callable(getattr(downloader, method))

    def test_function_signatures(self):
        """Test that functions have expected signatures."""
        import inspect

        # Test YouTubeDownloader.__init__
        init_sig = inspect.signature(YouTubeDownloader.__init__)
        assert "cache_dir" in init_sig.parameters

        # Test convenience functions
        audio_sig = inspect.signature(download_audio)
        video_sig = inspect.signature(download_video)

        assert len(audio_sig.parameters) == 2  # url, output_dir
        assert len(video_sig.parameters) == 2  # url, output_dir

    @patch("y2karaoke.core.downloader.yt_dlp.YoutubeDL")
    def test_metadata_methods_return_format(self, mock_ytdl):
        """Test that metadata methods return expected format."""
        mock_instance = Mock()
        mock_instance.extract_info.return_value = {
            "title": "Test",
            "uploader": "Channel",
        }
        mock_ytdl.return_value.__enter__.return_value = mock_instance

        downloader = YouTubeDownloader()

        title = downloader.get_video_title("https://youtube.com/watch?v=test")
        uploader = downloader.get_video_uploader("https://youtube.com/watch?v=test")

        assert isinstance(title, str)
        assert isinstance(uploader, str)
        assert title == "Test"
        assert uploader == "Channel"
