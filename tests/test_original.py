#!/usr/bin/env python3
"""Test original karaoke without uploader."""

import pytest

from y2karaoke.core.downloader import download_audio
from y2karaoke.core.separator import separate_vocals
from y2karaoke.core.lyrics import get_lyrics
from y2karaoke.core.video_writer import render_karaoke_video

@pytest.mark.skip(reason="Requires cached audio files - run manually if available")
def test_original():
    """Test rendering with cached audio files (requires manual setup)."""
    import glob
    import os

    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    # Use cached data
    cache_dir = "/Users/dtunkelang/.cache/karaoke/dQw4w9WgXcQ"

    if not os.path.exists(cache_dir):
        pytest.skip(f"Cache directory not found: {cache_dir}")

    # Find existing files
    audio_files = glob.glob(f"{cache_dir}/*.wav")
    original_audio = None
    vocals_file = None
    instrumental_file = None

    for f in audio_files:
        if "Vocals" in f:
            vocals_file = f
        elif "instrumental" in f:
            instrumental_file = f
        elif not any(x in f for x in ["Bass", "Drums", "Other", "Vocals"]):
            original_audio = f

    print(f"Using vocals: {vocals_file}")
    print(f"Using instrumental: {instrumental_file}")

    if not vocals_file or not instrumental_file:
        pytest.skip("Required audio files not found in cache")

    if vocals_file and instrumental_file:
        # Get lyrics
        lines, metadata = get_lyrics("Never Gonna Give You Up", "Rick Astley", vocals_file, cache_dir)
        print(f"Got {len(lines)} lines of lyrics")
        
        # Render video
        render_karaoke_video(
            lines=lines,
            audio_path=instrumental_file,
            output_path="test_original_working.mp4",
            title="Never Gonna Give You Up",
            artist="Rick Astley",
            timing_offset=0.0,
            song_metadata=metadata,
        )
        print("✅ Original version completed")
    else:
        print("❌ Missing required files")

if __name__ == "__main__":
    test_original()
