#!/usr/bin/env python3
"""Prewarm multi-source timed-lyrics cache for benchmark songs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from y2karaoke.core.components.lyrics.sync import fetch_from_all_sources

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_PATH = REPO_ROOT / "benchmarks" / "benchmark_songs.yaml"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch and cache all available timed-lyrics sources for songs."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Benchmark manifest YAML path",
    )
    parser.add_argument(
        "--match",
        help="Regex matched against 'artist - title' to filter songs",
    )
    parser.add_argument(
        "--max-songs",
        type=int,
        help="Optional cap on number of songs to fetch",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help="Optional JSON summary output path",
    )
    return parser.parse_args()


def _load_songs(path: Path) -> list[dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("songs"), list):
        raise ValueError(f"Invalid benchmark manifest: {path}")
    return [song for song in data["songs"] if isinstance(song, dict)]


def _select_songs(
    songs: list[dict[str, Any]], match: str | None, max_songs: int | None
) -> list[dict[str, Any]]:
    pattern = re.compile(match, re.IGNORECASE) if match else None
    selected: list[dict[str, Any]] = []
    for song in songs:
        artist = str(song.get("artist", "")).strip()
        title = str(song.get("title", "")).strip()
        label = f"{artist} - {title}"
        if pattern and not pattern.search(label):
            continue
        selected.append(song)
        if max_songs and len(selected) >= max_songs:
            break
    return selected


def main() -> int:
    args = _parse_args()
    songs = _select_songs(
        _load_songs(args.manifest.resolve()),
        match=args.match,
        max_songs=args.max_songs,
    )
    if not songs:
        print("lyrics_source_cache_prewarm: NO_MATCHES")
        return 2

    summary: list[dict[str, Any]] = []
    print(f"Prewarming lyrics sources for {len(songs)} song(s)")
    for index, song in enumerate(songs, start=1):
        artist = str(song.get("artist", "")).strip()
        title = str(song.get("title", "")).strip()
        label = f"{artist} - {title}"
        sources = fetch_from_all_sources(title, artist)
        source_names = sorted(str(name) for name in sources.keys())
        print(f"[{index}/{len(songs)}] {label}: {len(source_names)} source(s)")
        summary.append(
            {
                "artist": artist,
                "title": title,
                "source_count": len(source_names),
                "sources": source_names,
            }
        )

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps({"songs": summary}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Saved JSON: {args.json_out}")

    print("lyrics_source_cache_prewarm: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
