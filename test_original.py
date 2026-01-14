#!/usr/bin/env python3
"""Test original karaoke without uploader."""

import sys
sys.path.append('backup_old_structure')

from downloader import download_audio
from separator import separate_vocals
from lyrics import get_lyrics
from renderer import render_karaoke_video

def test_original():
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    
    # Use cached data
    cache_dir = "/Users/dtunkelang/.cache/karaoke/dQw4w9WgXcQ"
    
    # Find existing files
    import glob
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
