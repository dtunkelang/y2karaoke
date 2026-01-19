#!/usr/bin/env python3
"""
CLI test for fetching Genius lyrics with singer annotations.
"""

import argparse
from y2karaoke.core.genius import fetch_genius_lyrics_with_singers

def main():
    parser = argparse.ArgumentParser(description="Fetch Genius lyrics with singers.")
    parser.add_argument("artist", type=str, help="Artist name")
    parser.add_argument("title", type=str, help="Song title")
    parser.add_argument("--lines", type=int, default=10, help="Number of lines to display")
    args = parser.parse_args()

    lyrics_with_singers, metadata = fetch_genius_lyrics_with_singers(args.title, args.artist)

    if not lyrics_with_singers or not metadata:
        print(f"No lyrics found for '{args.title}' by {args.artist}")
        return

    print(f"Song: {metadata.title} by {metadata.artist}")
    print(f"Singers: {metadata.singers} (duet: {metadata.is_duet})")
    print("\nFirst {} lines (after stripping artist-only lines):".format(args.lines))

    for i, (line, singer) in enumerate(lyrics_with_singers[:args.lines]):
        singer_info = f" [{singer}]" if singer else ""
        print(f"{i+1:02d}: {line}{singer_info}")

if __name__ == "__main__":
    main()
