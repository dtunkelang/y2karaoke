#!/usr/bin/env python3
"""Validate benchmark song manifest structure and invariants."""

from __future__ import annotations

from pathlib import Path
import re
import sys
from typing import Any

import yaml  # type: ignore[import-untyped]

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_PATH = REPO_ROOT / "benchmarks" / "benchmark_songs.yaml"
YOUTUBE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
VALID_PROVIDERS = {"lyriq", "syncedlyrics"}


def _validate_song(song: Any, index: int) -> list[str]:
    errors: list[str] = []
    if not isinstance(song, dict):
        return [f"songs[{index}] must be a mapping"]

    required = [
        "artist",
        "title",
        "youtube_id",
        "youtube_url",
        "preferred_lyrics_provider",
        "fallback_lyrics_provider",
        "lrc_duration_tolerance_sec",
    ]
    missing = [key for key in required if key not in song]
    if missing:
        errors.append(f"songs[{index}] missing required keys: {', '.join(missing)}")
        return errors

    artist = song["artist"]
    title = song["title"]
    if not isinstance(artist, str) or not artist.strip():
        errors.append(f"songs[{index}].artist must be a non-empty string")
    if not isinstance(title, str) or not title.strip():
        errors.append(f"songs[{index}].title must be a non-empty string")

    youtube_id = song["youtube_id"]
    if not isinstance(youtube_id, str) or not YOUTUBE_ID_RE.match(youtube_id):
        errors.append(f"songs[{index}].youtube_id must be an 11-char YouTube ID")

    youtube_url = song["youtube_url"]
    if not isinstance(youtube_url, str) or "youtube.com/watch?v=" not in youtube_url:
        errors.append(f"songs[{index}].youtube_url must be a YouTube watch URL")
    elif isinstance(youtube_id, str) and youtube_id not in youtube_url:
        errors.append(
            f"songs[{index}].youtube_url must include youtube_id '{youtube_id}'"
        )

    preferred = song["preferred_lyrics_provider"]
    fallback = song["fallback_lyrics_provider"]
    if preferred not in VALID_PROVIDERS:
        errors.append(
            f"songs[{index}].preferred_lyrics_provider must be one of {sorted(VALID_PROVIDERS)}"
        )
    if fallback not in VALID_PROVIDERS:
        errors.append(
            f"songs[{index}].fallback_lyrics_provider must be one of {sorted(VALID_PROVIDERS)}"
        )
    if preferred == fallback:
        errors.append(f"songs[{index}] preferred and fallback providers must differ")

    tolerance = song["lrc_duration_tolerance_sec"]
    if not isinstance(tolerance, int):
        errors.append(f"songs[{index}].lrc_duration_tolerance_sec must be an integer")
    elif tolerance < 1 or tolerance > 180:
        errors.append(
            f"songs[{index}].lrc_duration_tolerance_sec must be between 1 and 180"
        )

    if "notes" in song and not isinstance(song["notes"], str):
        errors.append(f"songs[{index}].notes must be a string when present")

    return errors


def _load_manifest(path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    if not path.exists():
        return None, [f"Manifest not found: {path}"]
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, [f"Failed to parse YAML: {exc}"]
    if not isinstance(data, dict):
        return None, ["Manifest root must be a mapping"]
    return data, []


def _validate_top_level(data: dict[str, Any]) -> tuple[list[Any], list[str]]:
    errors: list[str] = []
    for top_key in ("version", "name", "songs"):
        if top_key not in data:
            errors.append(f"Missing required top-level key: {top_key}")
    if errors:
        return [], errors
    if not isinstance(data["version"], int):
        errors.append("version must be an integer")
    if not isinstance(data["name"], str) or not data["name"].strip():
        errors.append("name must be a non-empty string")
    songs = data["songs"]
    if not isinstance(songs, list) or not songs:
        errors.append("songs must be a non-empty list")
        return [], errors
    return songs, errors


def _validate_uniqueness(songs: list[Any]) -> list[str]:
    errors: list[str] = []
    seen_ids: set[str] = set()
    seen_tracks: set[tuple[str, str]] = set()
    for idx, song in enumerate(songs):
        if not isinstance(song, dict):
            continue
        youtube_id = song.get("youtube_id")
        if isinstance(youtube_id, str):
            if youtube_id in seen_ids:
                errors.append(f"Duplicate youtube_id at songs[{idx}]: {youtube_id}")
            seen_ids.add(youtube_id)
        artist = song.get("artist")
        title = song.get("title")
        if isinstance(artist, str) and isinstance(title, str):
            track_key = (artist.strip().lower(), title.strip().lower())
            if track_key in seen_tracks:
                errors.append(
                    f"Duplicate artist/title at songs[{idx}]: {artist} - {title}"
                )
            seen_tracks.add(track_key)
    return errors


def validate_manifest(path: Path) -> list[str]:
    data, errors = _load_manifest(path)
    if data is None:
        return errors

    songs, top_errors = _validate_top_level(data)
    errors.extend(top_errors)
    if not songs:
        return errors

    for idx, song in enumerate(songs):
        errors.extend(_validate_song(song, idx))
    errors.extend(_validate_uniqueness(songs))
    return errors


def main() -> int:
    path = (
        Path(sys.argv[1]).resolve()
        if len(sys.argv) > 1
        else DEFAULT_MANIFEST_PATH.resolve()
    )
    errors = validate_manifest(path)
    if errors:
        print("benchmark_manifest: FAIL")
        for err in errors:
            print(f"- {err}")
        return 1

    print(f"benchmark_manifest: OK ({path})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
