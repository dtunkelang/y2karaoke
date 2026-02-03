from pathlib import Path
from unittest.mock import MagicMock

from y2karaoke.core.karaoke import KaraokeGenerator
from y2karaoke.utils.cache import CacheManager


def test_download_video_force_uses_downloader(tmp_path):
    generator = KaraokeGenerator(cache_dir=tmp_path)
    generator.downloader = MagicMock()

    cache = CacheManager(tmp_path)
    video_id = "abc123"
    video_dir = cache.get_video_cache_dir(video_id)
    cached_video = video_dir / "abc123_video.mp4"
    cached_video.write_bytes(b"video")

    generator.downloader.download_video.return_value = {"video_path": "fresh.mp4"}

    result = generator._download_video(video_id, "https://youtu.be/abc123", True)

    assert result["video_path"] == "fresh.mp4"
    generator.downloader.download_video.assert_called_once()


def test_download_audio_no_cache_calls_downloader(tmp_path):
    generator = KaraokeGenerator(cache_dir=tmp_path)
    generator.downloader = MagicMock()

    generator.downloader.download_audio.return_value = {
        "audio_path": "audio.wav",
        "title": "Song",
        "artist": "Artist",
    }

    result = generator._download_audio("vid", "https://youtu.be/vid", False)

    assert result["audio_path"] == "audio.wav"
    generator.downloader.download_audio.assert_called_once()


def test_cleanup_temp_files_removes_files(tmp_path):
    generator = KaraokeGenerator(cache_dir=tmp_path)

    temp_file = tmp_path / "temp.wav"
    temp_file.write_bytes(b"data")
    generator._temp_files.append(str(temp_file))

    generator.cleanup_temp_files()

    assert not temp_file.exists()
    assert generator._temp_files == []
