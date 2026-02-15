#!/usr/bin/env python3
"""Bulk check karaoke suitability for the gold set with caching."""

from __future__ import annotations

import json
import subprocess
import sys
import yaml
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from y2karaoke.core.audio_analysis import calculate_harmonic_suitability

# Constants
CACHE_DIR = Path("/Users/dtunkelang/.cache/karaoke")
OUTPUT_DIR = Path(".cache/bulk_suitability")
RESULTS_CACHE = OUTPUT_DIR / "results.json"

def _get_best_karaoke_url(artist: str, title: str) -> str:
    """Find the best karaoke video URL using existing bootstrap logic."""
    print(f"  Searching for karaoke candidate: {artist} - {title}")
    query = f"{artist} {title} karaoke"
    cmd = [
        ".venv/bin/yt-dlp",
        f"ytsearch1:{query}",
        "--get-id",
        "--no-playlist"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    video_id = res.stdout.strip()
    return f"https://www.youtube.com/watch?v={video_id}"

def download_audio(url: str, output_path: Path):
    """Download audio from YouTube."""
    if output_path.exists():
        return
    print(f"  Downloading karaoke audio...")
    subprocess.run([
        ".venv/bin/yt-dlp", "--no-playlist", "--format", "ba", "-x", 
        "--audio-format", "wav", "-o", str(output_path).replace(".wav", ".%(ext)s"), url
    ], capture_output=True, check=True)

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load existing results for caching
    cached_data = {}
    if RESULTS_CACHE.exists():
        try:
            raw_cached = json.loads(RESULTS_CACHE.read_text())
            for item in raw_cached:
                key = f"{item['artist']}-{item['title']}"
                cached_data[key] = item
        except Exception:
            pass

    with open("benchmarks/benchmark_songs.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    results = []
    print(f"{'Artist - Title':<40} | {'Cost':<8} | {'Key':<4} | {'Suitability'}")
    print("-" * 80)

    for song in config["songs"]:
        artist = song["artist"]
        title = song["title"]
        orig_id = song["youtube_id"]
        key = f"{artist}-{title}"
        
        if key in cached_data:
            item = cached_data[key]
            cost = item["similarity_cost"]
            print(f"{artist + ' - ' + title:<40} | {cost:0.4f}*  | {item['best_key_shift']:<4} | {item['suitability']}")
            results.append(item)
            continue

        # 1. Find official instrumental in cache
        orig_dir = CACHE_DIR / orig_id
        orig_inst = list(orig_dir.glob("*_instrumental.wav"))
        if not orig_inst:
            print(f"Skipping {artist} - {title}: Official instrumental not found in cache.")
            continue
        orig_inst_path = orig_inst[0]
        
        # 2. Find and download best karaoke candidate
        karaoke_url = _get_best_karaoke_url(artist, title)
        karaoke_id = karaoke_url.split("=")[-1]
        karaoke_wav = OUTPUT_DIR / f"{karaoke_id}.wav"
        
        try:
            download_audio(karaoke_url, karaoke_wav)
            
            # 3. Run suitability check
            print(f"  Analyzing harmonic suitability...")
            metrics = calculate_harmonic_suitability(str(orig_inst_path), str(karaoke_wav))
            
            if "error" in metrics:
                print(f"{artist + ' - ' + title:<40} | ERROR: {metrics['error']}")
                continue
                
            cost = metrics["similarity_cost"]
            suit = "EXCELLENT" if cost < 0.15 else "ACCEPTABLE" if cost < 0.35 else "POOR" if cost < 0.6 else "UNUSABLE"
            
            print(f"{artist + ' - ' + title:<40} | {cost:0.4f}   | {metrics['best_key_shift']:<4} | {suit}")
            
            new_item = {
                "artist": artist,
                "title": title,
                "similarity_cost": cost,
                "best_key_shift": metrics["best_key_shift"],
                "suitability": suit,
                "karaoke_url": karaoke_url,
                "offset_seconds": metrics.get("offset_seconds", 0.0)
            }
            results.append(new_item)
            
            # Update cache file incrementally after each success
            RESULTS_CACHE.write_text(json.dumps(results, indent=2))
            
        except Exception as e:
            print(f"{artist + ' - ' + title:<40} | FAILED: {str(e)}")

    print("\n* = Loaded from cache")

if __name__ == "__main__":
    main()
