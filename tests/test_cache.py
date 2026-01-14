"""Test cache management."""

import json
import pytest
from pathlib import Path
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
        
        assert stats['video_count'] == 2
        assert stats['file_count'] == 3
        assert stats['total_size_gb'] > 0
        assert stats['cache_dir'] == str(temp_dir)
