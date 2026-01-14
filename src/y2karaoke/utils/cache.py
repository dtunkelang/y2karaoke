"""Cache management system."""

import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional

from ..config import get_cache_dir, MAX_CACHE_SIZE_GB, CACHE_CLEANUP_THRESHOLD
from ..exceptions import CacheError
from ..utils.logging import get_logger

logger = get_logger(__name__)

class CacheManager:
    """Manages caching of intermediate files and metadata."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or get_cache_dir()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def get_video_cache_dir(self, video_id: str) -> Path:
        """Get cache directory for specific video."""
        video_dir = self.cache_dir / video_id
        video_dir.mkdir(parents=True, exist_ok=True)
        return video_dir
    
    def save_metadata(self, video_id: str, metadata: Dict[str, Any]):
        """Save metadata for a video."""
        cache_dir = self.get_video_cache_dir(video_id)
        metadata_file = cache_dir / "metadata.json"
        
        try:
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            logger.debug(f"Saved metadata for {video_id}")
        except Exception as e:
            raise CacheError(f"Failed to save metadata: {e}")
    
    def load_metadata(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Load metadata for a video."""
        cache_dir = self.get_video_cache_dir(video_id)
        metadata_file = cache_dir / "metadata.json"
        
        if not metadata_file.exists():
            return None
        
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            logger.debug(f"Loaded metadata for {video_id}")
            return metadata
        except Exception as e:
            logger.warning(f"Failed to load metadata for {video_id}: {e}")
            return None
    
    def file_exists(self, video_id: str, filename: str) -> bool:
        """Check if a cached file exists."""
        cache_dir = self.get_video_cache_dir(video_id)
        return (cache_dir / filename).exists()
    
    def get_file_path(self, video_id: str, filename: str) -> Path:
        """Get path to cached file."""
        cache_dir = self.get_video_cache_dir(video_id)
        return cache_dir / filename
    
    def find_files(self, video_id: str, pattern: str) -> list[Path]:
        """Find files matching pattern in video cache."""
        cache_dir = self.get_video_cache_dir(video_id)
        return list(cache_dir.glob(pattern))
    
    def clear_video_cache(self, video_id: str):
        """Clear cache for specific video."""
        cache_dir = self.get_video_cache_dir(video_id)
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            logger.info(f"Cleared cache for video {video_id}")
    
    def get_cache_size(self) -> float:
        """Get total cache size in GB."""
        total_size = 0
        for path in self.cache_dir.rglob('*'):
            if path.is_file():
                total_size += path.stat().st_size
        return total_size / (1024 ** 3)  # Convert to GB
    
    def cleanup_old_files(self, max_age_days: int = 30):
        """Remove files older than specified days."""
        cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)
        removed_count = 0
        
        for path in self.cache_dir.rglob('*'):
            if path.is_file() and path.stat().st_mtime < cutoff_time:
                try:
                    path.unlink()
                    removed_count += 1
                except OSError:
                    pass
        
        # Remove empty directories
        for path in self.cache_dir.rglob('*'):
            if path.is_dir() and not any(path.iterdir()):
                try:
                    path.rmdir()
                except OSError:
                    pass
        
        logger.info(f"Cleaned up {removed_count} old files")
    
    def auto_cleanup(self):
        """Automatically cleanup cache if it's too large."""
        cache_size = self.get_cache_size()
        
        if cache_size > MAX_CACHE_SIZE_GB * CACHE_CLEANUP_THRESHOLD:
            logger.info(f"Cache size ({cache_size:.1f}GB) exceeds threshold, cleaning up")
            
            # First try removing old files
            self.cleanup_old_files(7)  # Remove files older than 7 days
            
            # If still too large, remove older files
            cache_size = self.get_cache_size()
            if cache_size > MAX_CACHE_SIZE_GB * CACHE_CLEANUP_THRESHOLD:
                self.cleanup_old_files(1)  # Remove files older than 1 day
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_size = 0
        file_count = 0
        video_count = 0
        
        for video_dir in self.cache_dir.iterdir():
            if video_dir.is_dir():
                video_count += 1
                for file_path in video_dir.rglob('*'):
                    if file_path.is_file():
                        total_size += file_path.stat().st_size
                        file_count += 1
        
        return {
            'total_size_gb': total_size / (1024 ** 3),
            'file_count': file_count,
            'video_count': video_count,
            'cache_dir': str(self.cache_dir),
        }
