#!/usr/bin/env python3
"""Report curated clips whose saved gold audio window disagrees with the manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any
import wave

import yaml  # type: ignore[import-untyped]

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_PATH = REPO_ROOT / "benchmarks" / "curated_clip_songs.yaml"
DEFAULT_GOLD_ROOT = (
    REPO_ROOT / "benchmarks" / "clip_gold_candidate" / "20260312T_curated_clips"
)


def _parse_gold_title_and_clip_id(title: str) -> tuple[str, str | None]:
    if " [" in title and title.endswith("]"):
        base_title, clip_id = title[:-1].split(" [", 1)
        return base_title, clip_id
    return title, None


def _read_wav_duration_sec(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as wav_file:
            frame_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
    except (FileNotFoundError, OSError, wave.Error):
        return None
    if frame_rate <= 0:
        return None
    return frame_count / frame_rate


def _parse_duration_from_audio_path(audio_path: str) -> float | None:
    match = re.search(r"-(\d+(?:_\d+)?)s-(\d+(?:_\d+)?)s\.wav$", audio_path)
    if not match:
        return None
    start = float(match.group(1).replace("_", "."))
    end = float(match.group(2).replace("_", "."))
    return end - start


def _gold_audio_duration_sec(audio_path: str) -> float | None:
    duration = _read_wav_duration_sec(Path(audio_path))
    if duration is not None:
        return duration
    return _parse_duration_from_audio_path(audio_path)


def _entry_float(entry: dict[str, object], key: str) -> float:
    value = entry.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"Expected numeric {key}, got {value!r}")
    return float(value)


def _load_manifest(path: Path) -> list[dict[str, object]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("songs"), list):
        raise ValueError(f"Invalid manifest structure: {path}")
    songs = [song for song in data["songs"] if isinstance(song, dict)]
    return songs


def _build_manifest_lookup(
    songs: list[dict[str, object]],
) -> tuple[
    dict[tuple[str, str, str], dict[str, object]],
    dict[tuple[str, str], list[dict[str, object]]],
]:
    by_key: dict[tuple[str, str, str], dict[str, object]] = {}
    by_artist_title: dict[tuple[str, str], list[dict[str, object]]] = {}
    for song in songs:
        artist = song.get("artist")
        title = song.get("title")
        clip_id = song.get("clip_id")
        if not isinstance(artist, str) or not isinstance(title, str):
            continue
        key = (artist.strip().lower(), title.strip().lower())
        by_artist_title.setdefault(key, []).append(song)
        if isinstance(clip_id, str) and clip_id.strip():
            by_key[(key[0], key[1], clip_id.strip().lower())] = song
    return by_key, by_artist_title


def _matching_manifest_song(
    *,
    artist: str,
    title: str,
    clip_id: str | None,
    by_key: dict[tuple[str, str, str], dict[str, object]],
    by_artist_title: dict[tuple[str, str], list[dict[str, object]]],
) -> dict[str, object] | None:
    song = None
    artist_key = artist.strip().lower()
    title_key = title.strip().lower()
    if isinstance(clip_id, str) and clip_id.strip():
        song = by_key.get((artist_key, title_key, clip_id.strip().lower()))
    if song is None:
        matches = by_artist_title.get((artist_key, title_key), [])
        if len(matches) == 1:
            song = matches[0]
    return song


def _stale_entry_from_gold_doc(
    *,
    gold_path: Path,
    gold_doc: dict[str, Any],
    by_key: dict[tuple[str, str, str], dict[str, object]],
    by_artist_title: dict[tuple[str, str], list[dict[str, object]]],
    tolerance_sec: float,
) -> dict[str, object] | None:
    artist = gold_doc.get("artist")
    title = gold_doc.get("title")
    audio_path = gold_doc.get("audio_path")
    if not isinstance(artist, str) or not isinstance(title, str):
        return None
    if not isinstance(audio_path, str) or not audio_path:
        return None
    base_title, clip_id = _parse_gold_title_and_clip_id(title)
    song = _matching_manifest_song(
        artist=artist,
        title=base_title,
        clip_id=clip_id,
        by_key=by_key,
        by_artist_title=by_artist_title,
    )
    if song is None:
        return None
    manifest_duration = song.get("clip_duration_sec")
    clip_id_value = song.get("clip_id")
    if not isinstance(manifest_duration, (int, float)):
        return None
    if not isinstance(clip_id_value, str) or not clip_id_value.strip():
        return None
    gold_duration = _gold_audio_duration_sec(audio_path)
    if gold_duration is None:
        return None
    manifest_duration_f = float(manifest_duration)
    delta = manifest_duration_f - gold_duration
    if abs(delta) < tolerance_sec:
        return None
    return {
        "artist": artist,
        "title": base_title,
        "clip_id": clip_id_value,
        "manifest_duration_sec": manifest_duration_f,
        "gold_audio_duration_sec": gold_duration,
        "delta_sec": delta,
        "gold_path": gold_path,
        "audio_path": audio_path,
    }


def _sort_stale_key(entry: dict[str, object]) -> tuple[float, str, str, str]:
    return (
        -abs(_entry_float(entry, "delta_sec")),
        str(entry["artist"]).lower(),
        str(entry["title"]).lower(),
        str(entry["clip_id"]).lower(),
    )


def _format_stale_entry(entry: dict[str, object], index: int) -> str:
    manifest = _entry_float(entry, "manifest_duration_sec")
    gold = _entry_float(entry, "gold_audio_duration_sec")
    delta = _entry_float(entry, "delta_sec")
    return (
        f"{index}. {entry['artist']} - {entry['title']} [{entry['clip_id']}] "
        f"manifest={manifest:.2f}s gold={gold:.2f}s delta={delta:+.2f}s"
    )


def collect_stale_gold_entries(
    *,
    manifest_path: Path,
    gold_root: Path,
    tolerance_sec: float,
) -> list[dict[str, object]]:
    songs = _load_manifest(manifest_path)
    by_key, by_artist_title = _build_manifest_lookup(songs)
    stale: list[dict[str, object]] = []
    for gold_path in sorted(gold_root.glob("*.gold.json")):
        try:
            gold_doc = json.loads(gold_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(gold_doc, dict):
            continue
        entry = _stale_entry_from_gold_doc(
            gold_path=gold_path,
            gold_doc=gold_doc,
            by_key=by_key,
            by_artist_title=by_artist_title,
            tolerance_sec=tolerance_sec,
        )
        if entry is not None:
            stale.append(entry)
    stale.sort(key=_sort_stale_key)
    return stale


def _print_report(entries: list[dict[str, object]]) -> None:
    if not entries:
        print("stale_curated_gold: OK")
        return
    print(f"stale_curated_gold: FAIL ({len(entries)} mismatches)")
    for index, entry in enumerate(entries, start=1):
        print(_format_stale_entry(entry, index))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--gold-root", type=Path, default=DEFAULT_GOLD_ROOT)
    parser.add_argument("--tolerance-sec", type=float, default=0.5)
    args = parser.parse_args()

    entries = collect_stale_gold_entries(
        manifest_path=args.manifest.resolve(),
        gold_root=args.gold_root.resolve(),
        tolerance_sec=args.tolerance_sec,
    )
    _print_report(entries)
    return 1 if entries else 0


if __name__ == "__main__":
    raise SystemExit(main())
