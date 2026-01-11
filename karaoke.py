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

from downloader import download_audio, download_video
from separator import separate_vocals
from lyrics import get_lyrics, Line, Word
from renderer import render_karaoke_video
from audio_effects import process_audio
from uploader import upload_video, generate_metadata
from backgrounds import create_background_segments


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


def scale_lyrics_timing(lines: list[Line], tempo_multiplier: float) -> list[Line]:
    """Scale all lyrics timestamps by the tempo multiplier."""
    if tempo_multiplier == 1.0:
        return lines

    scaled_lines = []
    for line in lines:
        scaled_words = [
            Word(
                text=word.text,
                start_time=word.start_time / tempo_multiplier,
                end_time=word.end_time / tempo_multiplier,
            )
            for word in line.words
        ]
        scaled_lines.append(
            Line(
                words=scaled_words,
                start_time=line.start_time / tempo_multiplier,
                end_time=line.end_time / tempo_multiplier,
            )
        )
    return scaled_lines


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
        default=0.0
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download and re-process even if cached files exist"
    )
    parser.add_argument(
        "--key",
        type=int,
        default=0,
        help="Shift key by N semitones (-12 to +12)"
    )
    parser.add_argument(
        "--tempo",
        type=float,
        default=1.0,
        help="Tempo multiplier (1.0 = original, 0.5 = half speed, 2.0 = double)"
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload to YouTube as unlisted video after rendering"
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Skip the upload prompt (for batch/script mode)"
    )
    parser.add_argument(
        "--backgrounds",
        action="store_true",
        help="Use video backgrounds extracted from the original YouTube video"
    )

    args = parser.parse_args()

    # Validate key and tempo
    if not -12 <= args.key <= 12:
        print("ERROR: --key must be between -12 and +12 semitones")
        sys.exit(1)
    if args.tempo <= 0:
        print("ERROR: --tempo must be positive")
        sys.exit(1)

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
        print("\n[1/5] Downloading audio from YouTube...")

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

        # Download video if backgrounds are requested
        video_path = None
        if args.backgrounds:
            existing_video = find_existing_file(os.path.join(work_dir, "*_video.mp4"))
            if existing_video and not args.force:
                print(f"      Using cached video: {os.path.basename(existing_video)}")
                video_path = existing_video
            else:
                video_result = download_video(args.url, output_dir=work_dir)
                video_path = video_result['video_path']

        # Step 2: Separate vocals from instrumental (skip if exists)
        print("\n[2/5] Separating vocals from instrumental...")

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
        print("\n[3/5] Fetching lyrics and timing...")
        lines = get_lyrics(title, artist, vocals_path)
        print(f"      Found {len(lines)} lines of lyrics")

        # Step 4: Apply audio effects (key shift / tempo change)
        if args.key != 0 or args.tempo != 1.0:
            print("\n[4/5] Applying audio effects...")
            if args.key != 0:
                print(f"      Key shift: {args.key:+d} semitones")
            if args.tempo != 1.0:
                print(f"      Tempo: {args.tempo:.2f}x")

            # Process instrumental track
            processed_instrumental = os.path.join(
                work_dir,
                f"instrumental_key{args.key:+d}_tempo{args.tempo:.2f}.wav"
            )
            process_audio(
                instrumental_path,
                processed_instrumental,
                semitones=args.key,
                tempo_multiplier=args.tempo,
            )
            instrumental_path = processed_instrumental

            # Scale lyrics timing to match tempo change
            lines = scale_lyrics_timing(lines, args.tempo)
        else:
            print("\n[4/5] Applying audio effects...")
            print("      No audio effects requested")

        # Step 5: Render karaoke video
        print("\n[5/5] Rendering karaoke video...")

        # Create background segments if video backgrounds requested
        background_segments = None
        if args.backgrounds and video_path:
            from moviepy import AudioFileClip
            audio_clip = AudioFileClip(instrumental_path)
            duration = audio_clip.duration
            audio_clip.close()

            print("      Extracting video backgrounds...")
            background_segments = create_background_segments(video_path, lines, duration)
            if background_segments:
                print(f"      Created {len(background_segments)} background segments")

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
            artist=artist,
            timing_offset=args.offset,
            background_segments=background_segments,
        )

        print("\n" + "=" * 60)
        print("SUCCESS!")
        print(f"Output: {output_path}")
        print("=" * 60)

        # Handle YouTube upload
        should_upload = args.upload
        if not should_upload and not args.no_upload:
            # Prompt user
            try:
                response = input("\nUpload to YouTube as unlisted video? [y/N]: ").strip().lower()
                should_upload = response in ('y', 'yes')
            except EOFError:
                should_upload = False

        if should_upload:
            try:
                video_title, description = generate_metadata(title, artist)
                video_url = upload_video(output_path, video_title, description)
                print("\n" + "=" * 60)
                print("UPLOAD COMPLETE!")
                print(f"Share this link: {video_url}")
                print("=" * 60)
            except FileNotFoundError as e:
                print(f"\n{e}")
            except Exception as e:
                print(f"\nUpload failed: {e}")

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
