"""Test cache management."""

import json
import os
from pathlib import Path

import pytest

from y2karaoke.exceptions import CacheError
from y2karaoke.utils.cache import CacheManager


class TestCacheManager:
    """Test cache management functionality."""

    def test_cache_manager_init(self, temp_dir):
        """Test cache manager initialization."""
        manager = CacheManager(temp_dir)
        assert manager.cache_dir == temp_dir
        assert temp_dir.exists()

    def test_get_video_cache_dir(self, temp_dir):
        """Test video cache directory creation."""
        manager = CacheManager(temp_dir)
        video_dir = manager.get_video_cache_dir("test_video_id")

        assert video_dir.exists()
        assert video_dir.name == "test_video_id"
        assert video_dir.parent == temp_dir

    def test_save_and_load_metadata(self, temp_dir):
        """Test metadata save and load."""
        manager = CacheManager(temp_dir)
        video_id = "test_video"

        metadata = {
            "title": "Test Song",
            "artist": "Test Artist",
            "duration": 180.5,
        }

        # Save metadata
        manager.save_metadata(video_id, metadata)

        # Load metadata
        loaded = manager.load_metadata(video_id)
        assert loaded == metadata

    def test_load_nonexistent_metadata(self, temp_dir):
        """Test loading metadata that doesn't exist."""
        manager = CacheManager(temp_dir)
        result = manager.load_metadata("nonexistent")
        assert result is None

    def test_file_exists(self, temp_dir):
        """Test file existence checking."""
        manager = CacheManager(temp_dir)
        video_id = "test_video"

        # Create a test file
        cache_dir = manager.get_video_cache_dir(video_id)
        test_file = cache_dir / "test.txt"
        test_file.write_text("test content")

        assert manager.file_exists(video_id, "test.txt")
        assert not manager.file_exists(video_id, "nonexistent.txt")

    def test_find_files(self, temp_dir):
        """Test file pattern matching."""
        manager = CacheManager(temp_dir)
        video_id = "test_video"
        cache_dir = manager.get_video_cache_dir(video_id)

        # Create test files
        (cache_dir / "audio.wav").write_text("audio")
        (cache_dir / "vocals.wav").write_text("vocals")
        (cache_dir / "video.mp4").write_text("video")

        # Test pattern matching
        wav_files = manager.find_files(video_id, "*.wav")
        assert len(wav_files) == 2

        all_files = manager.find_files(video_id, "*")
        assert len(all_files) == 3

    def test_clear_video_cache(self, temp_dir):
        """Test clearing video cache."""
        manager = CacheManager(temp_dir)
        video_id = "test_video"

        # Create cache with files
        cache_dir = manager.get_video_cache_dir(video_id)
        (cache_dir / "test.txt").write_text("test")

        assert cache_dir.exists()

        # Clear cache
        manager.clear_video_cache(video_id)
        assert not cache_dir.exists()

    def test_get_cache_stats(self, temp_dir):
        """Test cache statistics."""
        manager = CacheManager(temp_dir)

        # Create some test data
        video1_dir = manager.get_video_cache_dir("video1")
        video2_dir = manager.get_video_cache_dir("video2")

        (video1_dir / "file1.txt").write_text("content1")
        (video1_dir / "file2.txt").write_text("content2")
        (video2_dir / "file3.txt").write_text("content3")

        stats = manager.get_cache_stats()

        assert stats["video_count"] == 2
        assert stats["file_count"] == 3
        assert stats["total_size_gb"] > 0
        assert stats["cache_dir"] == str(temp_dir)

    def test_load_metadata_invalid_json(self, temp_dir):
        """Invalid metadata should return None."""
        manager = CacheManager(temp_dir)
        cache_dir = manager.get_video_cache_dir("video")
        metadata_file = cache_dir / "metadata.json"
        metadata_file.write_text("{invalid json")

        assert manager.load_metadata("video") is None

    def test_save_metadata_raises_cache_error(self, temp_dir, monkeypatch):
        manager = CacheManager(temp_dir)

        def raise_error(*_args, **_kwargs):
            raise OSError("disk full")

        monkeypatch.setattr("builtins.open", raise_error)

        with pytest.raises(CacheError):
            manager.save_metadata("video", {"title": "x"})

    def test_get_cache_size_counts_files(self, temp_dir):
        """Get cache size sums file sizes."""
        manager = CacheManager(temp_dir)
        cache_dir = manager.get_video_cache_dir("video")
        (cache_dir / "file1.bin").write_bytes(b"a" * 1024)
        (cache_dir / "file2.bin").write_bytes(b"b" * 2048)

        size_gb = manager.get_cache_size()
        assert size_gb > 0

    def test_cleanup_old_files_removes_stale(self, temp_dir):
        """Cleanup removes old files and empty directories."""
        manager = CacheManager(temp_dir)
        cache_dir = manager.get_video_cache_dir("video")
        stale_file = cache_dir / "old.txt"
        stale_file.write_text("old")

        old_time = 0
        os.utime(stale_file, (old_time, old_time))

        manager.cleanup_old_files(max_age_days=1)

        assert not stale_file.exists()
        assert not cache_dir.exists()

    def test_cleanup_old_files_handles_unlink_error(self, temp_dir, monkeypatch):
        manager = CacheManager(temp_dir)
        cache_dir = manager.get_video_cache_dir("video")
        stale_file = cache_dir / "old.txt"
        stale_file.write_text("old")
        os.utime(stale_file, (0, 0))

        original_unlink = Path.unlink

        def fake_unlink(self, *args, **kwargs):
            if self == stale_file:
                raise OSError("fail")
            return original_unlink(self, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", fake_unlink)

        manager.cleanup_old_files(max_age_days=1)

        assert stale_file.exists()

    def test_cleanup_old_files_handles_rmdir_error(self, temp_dir, monkeypatch):
        manager = CacheManager(temp_dir)
        cache_dir = manager.get_video_cache_dir("video")
        empty_dir = cache_dir / "empty"
        empty_dir.mkdir(parents=True)

        original_rmdir = Path.rmdir

        def fake_rmdir(self, *args, **kwargs):
            if self == empty_dir:
                raise OSError("fail")
            return original_rmdir(self, *args, **kwargs)

        monkeypatch.setattr(Path, "rmdir", fake_rmdir)

        manager.cleanup_old_files(max_age_days=1)

        assert empty_dir.exists()

    def test_clear_video_cache_noop_when_missing(self, temp_dir, monkeypatch):
        manager = CacheManager(temp_dir)
        missing = temp_dir / "missing"

        monkeypatch.setattr(manager, "get_video_cache_dir", lambda _vid: missing)

        manager.clear_video_cache("missing")
        assert not missing.exists()

    def test_auto_cleanup_noop_when_below_threshold(self, temp_dir, monkeypatch):
        manager = CacheManager(temp_dir)

        monkeypatch.setattr(manager, "get_cache_size", lambda: 0.0)

        manager.auto_cleanup()

    def test_auto_cleanup_runs_two_passes_when_still_large(self, temp_dir, monkeypatch):
        """Auto cleanup performs multiple passes if needed."""
        manager = CacheManager(temp_dir)

        calls = []

        def fake_cleanup(days):
            calls.append(days)

        sizes = [1.0, 1.0, 0.0]

        def fake_size():
            return sizes.pop(0)

        monkeypatch.setattr(manager, "cleanup_old_files", fake_cleanup)
        monkeypatch.setattr(manager, "get_cache_size", fake_size)
        monkeypatch.setattr("y2karaoke.utils.cache.MAX_CACHE_SIZE_GB", 0.1)
        monkeypatch.setattr("y2karaoke.utils.cache.CACHE_CLEANUP_THRESHOLD", 0.5)

        manager.auto_cleanup()

        assert calls == [7, 1]

    def test_get_cache_stats_skips_non_dir_entries(self, temp_dir):
        manager = CacheManager(temp_dir)
        (temp_dir / "loose.txt").write_text("x")

        stats = manager.get_cache_stats()

        assert stats["video_count"] == 0
