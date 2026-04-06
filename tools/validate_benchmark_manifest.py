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


def _append_if_invalid(
    errors: list[str],
    *,
    valid: bool,
    message: str,
) -> None:
    if not valid:
        errors.append(message)


def _append_numeric_range_error(
    errors: list[str],
    *,
    value: Any,
    index: int,
    field: str,
    min_value: float | None = None,
    max_value: float | None = None,
    inclusive_min: bool = True,
) -> None:
    if not isinstance(value, (int, float)):
        errors.append(f"songs[{index}].{field} must be numeric")
        return
    value_f = float(value)
    if min_value is not None:
        if inclusive_min and value_f < min_value:
            errors.append(f"songs[{index}].{field} must be >= {min_value:g}")
            return
        if not inclusive_min and value_f <= min_value:
            errors.append(f"songs[{index}].{field} must be > {min_value:g}")
            return
    if max_value is not None and value_f > max_value:
        errors.append(f"songs[{index}].{field} must be <= {max_value:g}")


def _validate_required_keys(song: dict[str, Any], index: int) -> list[str]:
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
        return [f"songs[{index}] missing required keys: {', '.join(missing)}"]
    return []


def _validate_core_fields(song: dict[str, Any], index: int) -> list[str]:
    errors: list[str] = []
    artist = song["artist"]
    title = song["title"]
    _append_if_invalid(
        errors,
        valid=isinstance(artist, str) and bool(artist.strip()),
        message=f"songs[{index}].artist must be a non-empty string",
    )
    _append_if_invalid(
        errors,
        valid=isinstance(title, str) and bool(title.strip()),
        message=f"songs[{index}].title must be a non-empty string",
    )

    youtube_id = song["youtube_id"]
    _append_if_invalid(
        errors,
        valid=isinstance(youtube_id, str) and bool(YOUTUBE_ID_RE.match(youtube_id)),
        message=f"songs[{index}].youtube_id must be an 11-char YouTube ID",
    )

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
            "songs[{index}].preferred_lyrics_provider must be one of "
            f"{sorted(VALID_PROVIDERS)}"
        )
    if fallback not in VALID_PROVIDERS:
        errors.append(
            "songs[{index}].fallback_lyrics_provider must be one of "
            f"{sorted(VALID_PROVIDERS)}"
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

    return errors


def _validate_clip_fields(song: dict[str, Any], index: int) -> tuple[list[str], bool]:
    errors: list[str] = []
    if "clip_id" in song:
        clip_id = song["clip_id"]
        if clip_id is not None and (
            not isinstance(clip_id, str) or not clip_id.strip()
        ):
            errors.append(f"songs[{index}].clip_id must be a non-empty string")
    clip_id = song.get("clip_id")
    is_clip_entry = isinstance(clip_id, str) and bool(clip_id.strip())
    if "audio_start_sec" in song:
        _append_numeric_range_error(
            errors,
            value=song["audio_start_sec"],
            index=index,
            field="audio_start_sec",
            min_value=0.0,
        )
    elif is_clip_entry:
        errors.append(f"songs[{index}].audio_start_sec is required for clip entries")
    if "clip_duration_sec" in song:
        _append_numeric_range_error(
            errors,
            value=song["clip_duration_sec"],
            index=index,
            field="clip_duration_sec",
            min_value=0.0,
            inclusive_min=False,
        )
    elif is_clip_entry:
        errors.append(f"songs[{index}].clip_duration_sec is required for clip entries")
    return errors, is_clip_entry


def _validate_clip_tags(
    clip_tags: Any,
    *,
    index: int,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(clip_tags, list) or not clip_tags:
        return [f"songs[{index}].clip_tags must be a non-empty list when present"]
    normalized_tags: set[str] = set()
    for tag_idx, tag in enumerate(clip_tags):
        if not isinstance(tag, str) or not tag.strip():
            errors.append(
                f"songs[{index}].clip_tags[{tag_idx}] must be a non-empty string"
            )
            continue
        normalized_tags.add(tag.strip().lower())
    if len(normalized_tags) != len(clip_tags):
        errors.append(f"songs[{index}].clip_tags must not contain duplicates")
    return errors


def _validate_optional_fields(
    song: dict[str, Any],
    index: int,
    *,
    is_clip_entry: bool,
) -> list[str]:
    errors: list[str] = []
    if "notes" in song and not isinstance(song["notes"], str):
        errors.append(f"songs[{index}].notes must be a string when present")
    if "lyrics_file" in song:
        lyrics_file = song["lyrics_file"]
        if lyrics_file is not None and (
            not isinstance(lyrics_file, str) or not lyrics_file.strip()
        ):
            errors.append(f"songs[{index}].lyrics_file must be a non-empty string")
    if "clip_tags" in song:
        errors.extend(_validate_clip_tags(song["clip_tags"], index=index))
    elif is_clip_entry:
        errors.append(f"songs[{index}].clip_tags are required for clip entries")
    if is_clip_entry and not isinstance(song.get("notes"), str):
        errors.append(f"songs[{index}].notes are required for clip entries")
    return errors


def _validate_song(song: Any, index: int) -> list[str]:
    if not isinstance(song, dict):
        return [f"songs[{index}] must be a mapping"]

    errors = _validate_required_keys(song, index)
    if errors:
        return errors

    errors.extend(_validate_core_fields(song, index))
    clip_errors, is_clip_entry = _validate_clip_fields(song, index)
    errors.extend(clip_errors)
    errors.extend(_validate_optional_fields(song, index, is_clip_entry=is_clip_entry))

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
    seen_ids: set[tuple[str, str | None]] = set()
    seen_tracks: set[tuple[str, str, str | None]] = set()
    for idx, song in enumerate(songs):
        if not isinstance(song, dict):
            continue
        clip_id = song.get("clip_id")
        clip_key = clip_id.strip().lower() if isinstance(clip_id, str) else None
        youtube_id = song.get("youtube_id")
        if isinstance(youtube_id, str):
            id_key = (youtube_id, clip_key)
            if id_key in seen_ids:
                errors.append(
                    f"Duplicate youtube_id[/clip_id] at songs[{idx}]: {youtube_id}"
                )
            seen_ids.add(id_key)
        artist = song.get("artist")
        title = song.get("title")
        if isinstance(artist, str) and isinstance(title, str):
            track_key = (artist.strip().lower(), title.strip().lower(), clip_key)
            if track_key in seen_tracks:
                errors.append(
                    "Duplicate artist/title[/clip_id] at "
                    f"songs[{idx}]: {artist} - {title}"
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
