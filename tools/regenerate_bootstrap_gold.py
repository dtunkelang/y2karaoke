#!/usr/bin/env python3
import yaml
import sys
import subprocess
from pathlib import Path
import re

# Add src to path
sys.path.insert(0, "src")
from y2karaoke.core.text_utils import make_slug

def main():
    with open("benchmarks/benchmark_songs.yaml") as f:
        config = yaml.safe_load(f)

    songs = config.get("songs", [])
    output_dir = Path("benchmarks/gold_set")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Map existing files to songs to preserve numbering
    existing_files = list(output_dir.glob("*.gold.json"))
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-existing", action="store_true", help="Skip songs that already have a gold JSON")
    args = parser.parse_args()

    print(f"Regenerating gold timings for {len(songs)} songs...")

    for i, song in enumerate(songs):
        artist = song["artist"]
        title = song["title"]
        url = song["youtube_url"]
        
        slug_artist = make_slug(artist)
        slug_title = make_slug(title)
        slug_base = f"{slug_artist}-{slug_title}"
        
        # Find existing file to get prefix
        target_file = None
        for ef in existing_files:
            if slug_base in ef.name:
                target_file = ef
                break
        
        if not target_file:
            # New file, guess prefix
            target_file = output_dir / f"{i+1:02d}_{slug_base}.gold.json"
        
        if args.skip_existing and target_file.exists():
            print(f"Skipping [{i+1}/{len(songs)}] {artist} - {title} (already exists)")
            continue

        print(f"Processing [{i+1}/{len(songs)}] {artist} - {title} -> {target_file.name}")
        
        cmd = [
            sys.executable, "tools/bootstrap_gold_from_karaoke.py",
            "--artist", artist,
            "--title", title,
            "--candidate-url", url,
            "--output", str(target_file),
            "--visual-fps", "10.0",
            "--allow-low-suitability" # Always allow for gold-set regeneration to ensure we get a file
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print("  Success.")
        except subprocess.CalledProcessError:
            print("  FAILED.")
        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    main()
