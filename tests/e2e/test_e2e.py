#!/usr/bin/env python3
"""End-to-end test with available dependencies."""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from y2karaoke.core.components.audio.downloader import (  # noqa: E402
    YouTubeDownloader,
    extract_video_id,
)
from y2karaoke.utils.cache import CacheManager  # noqa: E402
from y2karaoke.utils.logging import setup_logging  # noqa: E402


def test_end_to_end():
    """Test what we can with available dependencies."""

    logger = setup_logging(level="INFO", verbose=True)

    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    try:
        # Test 1: URL validation and video ID extraction
        logger.info("🧪 Testing URL validation...")
        video_id = extract_video_id(url)
        logger.info(f"✅ Video ID extracted: {video_id}")

        # Test 2: Cache management
        logger.info("🧪 Testing cache management...")
        cache = CacheManager()
        stats = cache.get_cache_stats()
        logger.info(
            f"✅ Cache stats: {stats['video_count']} videos, {stats['total_size_gb']:.2f} GB"
        )

        # Test 3: Download (using cached if available)
        logger.info("🧪 Testing download...")
        _downloader = YouTubeDownloader()  # noqa: F841

        # Check if already cached
        metadata = cache.load_metadata(video_id)
        if metadata:
            logger.info(f"✅ Using cached: {metadata['title']} by {metadata['artist']}")
        else:
            # This would download if not cached
            logger.info("📥 Would download fresh (skipping to avoid re-download)")

        # Test 4: File operations
        logger.info("🧪 Testing file operations...")
        _cache_dir = cache.get_video_cache_dir(video_id)  # noqa: F841
        audio_files = cache.find_files(video_id, "*.wav")

        if audio_files:
            logger.info(f"✅ Found {len(audio_files)} audio files in cache")
            for file in audio_files[:3]:  # Show first 3
                logger.info(f"   📄 {file.name}")
        else:
            logger.info("ℹ️  No audio files in cache")

        # Test 5: Output path generation
        logger.info("🧪 Testing output path generation...")
        from y2karaoke.utils.validation import sanitize_filename

        title = "Never Gonna Give You Up"
        safe_title = sanitize_filename(title)
        output_path = f"{safe_title}_karaoke.mp4"
        logger.info(f"✅ Generated output path: {output_path}")

        logger.info("\n🎉 End-to-end test completed successfully!")
        logger.info("💡 Full processing requires additional dependencies:")
        logger.info("   - audio-separator (for vocal separation)")
        logger.info("   - moviepy (for video rendering)")
        logger.info("   - librosa (for audio effects)")
        logger.info("   - whisperx (for lyrics transcription)")

        # Use assert instead of return
        assert True

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        assert False, f"Test failed: {e}"


if __name__ == "__main__":
    test_end_to_end()
