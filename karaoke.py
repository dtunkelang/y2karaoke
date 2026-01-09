#!/usr/bin/env python3
"""
Karaoke Video Generator

Generate karaoke videos from YouTube URLs with word-by-word highlighting.

Usage:
    python karaoke.py "https://youtube.com/watch?v=xxxxx" -o output.mp4
"""

import argparse
import glob
import json
import os
import re
import sys
import shutil

from downloader import download_audio
from separator import separate_vocals
from lyrics import get_lyrics
from renderer import render_karaoke_video


# Default cache directory
CACHE_DIR = os.path.expanduser("~/.cache/karaoke")


def extract_video_id(url: str) -> str:
    """Extract YouTube video ID from URL."""
    patterns = [
        r'(?:v=|/v/|youtu\.be/)([a-zA-Z0-9_-]{11})',
        r'(?:embed/)([a-zA-Z0-9_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    # Fallback: hash the URL
    import hashlib
    return hashlib.md5(url.encode()).hexdigest()[:11]


def find_existing_file(pattern: str) -> str | None:
    """Find an existing file matching a glob pattern."""
    matches = glob.glob(pattern)
    return matches[0] if matches else None


def main():
    parser = argparse.ArgumentParser(
        description="Generate karaoke videos from YouTube URLs"
    )
    parser.add_argument(
        "url",
        help="YouTube URL to process"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output video path (default: {title}_karaoke.mp4)",
        default=None
    )
    parser.add_argument(
        "--keep-files",
        action="store_true",
        help="Keep intermediate files (audio, stems, etc.)"
    )
    parser.add_argument(
        "--work-dir",
        help="Working directory for intermediate files (default: ~/.cache/karaoke/{video_id})",
        default=None
    )
    parser.add_argument(
        "--offset",
        type=float,
        help="Timing offset in seconds (negative = highlight earlier, positive = later)",
        default=-0.3
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download and re-process even if cached files exist"
    )

    args = parser.parse_args()

    # Extract video ID for cache directory
    video_id = extract_video_id(args.url)

    # Set up working directory (use cache by default)
    if args.work_dir:
        work_dir = args.work_dir
    else:
        work_dir = os.path.join(CACHE_DIR, video_id)

    os.makedirs(work_dir, exist_ok=True)
    cleanup = False  # Never auto-cleanup cached files

    # Metadata file for caching title/artist
    metadata_file = os.path.join(work_dir, "metadata.json")

    try:
        print("=" * 60)
        print("KARAOKE VIDEO GENERATOR")
        print("=" * 60)
        print(f"Cache directory: {work_dir}")

        # Step 1: Download audio from YouTube (skip if exists)
        print("\n[1/4] Downloading audio from YouTube...")

        # Find original audio file (not stems)
        stem_suffixes = ['_Vocals', '_Bass', '_Drums', '_Other', '_instrumental', '_htdemucs']
        existing_audio = None
        for wav_file in glob.glob(os.path.join(work_dir, "*.wav")):
            if not any(suffix in wav_file for suffix in stem_suffixes):
                existing_audio = wav_file
                break

        if existing_audio and os.path.exists(metadata_file) and not args.force:
            print(f"      Using cached audio: {os.path.basename(existing_audio)}")
            audio_path = existing_audio
            with open(metadata_file) as f:
                metadata = json.load(f)
            title = metadata['title']
            artist = metadata['artist']

        if not existing_audio or args.force:
            download_result = download_audio(args.url, output_dir=work_dir)
            audio_path = download_result['audio_path']
            title = download_result['title']
            artist = download_result['artist']
            # Save metadata
            with open(metadata_file, 'w') as f:
                json.dump({'title': title, 'artist': artist}, f)

        print(f"      Title: {title}")
        print(f"      Artist: {artist}")

        # Step 2: Separate vocals from instrumental (skip if exists)
        print("\n[2/4] Separating vocals from instrumental...")

        existing_vocals = find_existing_file(os.path.join(work_dir, "*_(Vocals)_*.wav"))
        existing_instrumental = find_existing_file(os.path.join(work_dir, "*_instrumental.wav"))

        if existing_vocals and existing_instrumental and not args.force:
            print(f"      Using cached stems")
            vocals_path = existing_vocals
            instrumental_path = existing_instrumental
        else:
            separation_result = separate_vocals(audio_path, output_dir=work_dir)
            vocals_path = separation_result['vocals_path']
            instrumental_path = separation_result['instrumental_path']

        # Step 3: Get lyrics with timing
        print("\n[3/4] Fetching lyrics and timing...")
        lines = get_lyrics(title, artist, vocals_path)
        print(f"      Found {len(lines)} lines of lyrics")

        # Step 4: Render karaoke video
        print("\n[4/4] Rendering karaoke video...")

        # Determine output path
        if args.output:
            output_path = args.output
        else:
            safe_title = "".join(c for c in title if c.isalnum() or c in ' -_')[:50]
            output_path = f"{safe_title}_karaoke.mp4"

        render_karaoke_video(
            lines=lines,
            audio_path=instrumental_path,
            output_path=output_path,
            title=title,
            timing_offset=args.offset,
        )

        print("\n" + "=" * 60)
        print("SUCCESS!")
        print(f"Output: {output_path}")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\nCancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if cleanup and os.path.exists(work_dir):
            print(f"\nCleaning up temporary files...")
            shutil.rmtree(work_dir)


if __name__ == "__main__":
    main()
