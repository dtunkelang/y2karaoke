#!/usr/bin/env python3
"""Report curated gold files whose line timings fall outside clip bounds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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


def _load_manifest(path: Path) -> list[dict[str, object]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("songs"), list):
        raise ValueError(f"Invalid manifest structure: {path}")
    return [song for song in data["songs"] if isinstance(song, dict)]


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
    artist_key = artist.strip().lower()
    title_key = title.strip().lower()
    if isinstance(clip_id, str) and clip_id.strip():
        song = by_key.get((artist_key, title_key, clip_id.strip().lower()))
        if song is not None:
            return song
    matches = by_artist_title.get((artist_key, title_key), [])
    if len(matches) == 1:
        return matches[0]
    return None


def _line_bound_violations(
    lines: Any,
    *,
    clip_duration_sec: float,
    tolerance_sec: float,
) -> tuple[float | None, float | None]:
    if not isinstance(lines, list) or not lines:
        return None, None
    min_start: float | None = None
    max_end: float | None = None
    for line in lines:
        if not isinstance(line, dict):
            continue
        start = line.get("start")
        end = line.get("end")
        if isinstance(start, (int, float)):
            start_f = float(start)
            if start_f < -tolerance_sec:
                min_start = start_f if min_start is None else min(min_start, start_f)
        if isinstance(end, (int, float)):
            end_f = float(end)
            if end_f > clip_duration_sec + tolerance_sec:
                max_end = end_f if max_end is None else max(max_end, end_f)
    return min_start, max_end


def _entry_float(entry: dict[str, object], key: str) -> float:
    value = entry.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"Expected numeric {key}, got {value!r}")
    return float(value)


def _violation_entry_from_gold_doc(
    *,
    gold_path: Path,
    gold_doc: dict[str, Any],
    by_key: dict[tuple[str, str, str], dict[str, object]],
    by_artist_title: dict[tuple[str, str], list[dict[str, object]]],
    tolerance_sec: float,
) -> dict[str, object] | None:
    artist = gold_doc.get("artist")
    title = gold_doc.get("title")
    if not isinstance(artist, str) or not isinstance(title, str):
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
    if not isinstance(manifest_duration, (int, float)):
        return None
    manifest_duration_f = float(manifest_duration)
    min_start, max_end = _line_bound_violations(
        gold_doc.get("lines"),
        clip_duration_sec=manifest_duration_f,
        tolerance_sec=tolerance_sec,
    )
    if min_start is None and max_end is None:
        return None
    return {
        "artist": artist,
        "title": base_title,
        "clip_id": song.get("clip_id"),
        "manifest_duration_sec": manifest_duration_f,
        "min_start_sec": min_start,
        "max_end_sec": max_end,
        "gold_path": gold_path,
    }


def _sort_violation_key(entry: dict[str, object]) -> tuple[float, str, str, str]:
    min_start = _entry_float(
        {"min_start_sec": entry.get("min_start_sec") or 0.0},
        "min_start_sec",
    )
    max_end = _entry_float(
        {"max_end_sec": entry.get("max_end_sec") or 0.0}, "max_end_sec"
    )
    duration = _entry_float(entry, "manifest_duration_sec")
    return (
        -max(abs(min_start), abs(max_end - duration)),
        str(entry["artist"]).lower(),
        str(entry["title"]).lower(),
        str(entry.get("clip_id") or "").lower(),
    )


def _format_violation_entry(entry: dict[str, object], index: int) -> str:
    duration = _entry_float(entry, "manifest_duration_sec")
    min_start = entry.get("min_start_sec")
    max_end = entry.get("max_end_sec")
    clip_id = entry.get("clip_id") or "?"
    min_start_str = (
        f"{float(min_start):.2f}s" if isinstance(min_start, (int, float)) else "-"
    )
    max_end_str = f"{float(max_end):.2f}s" if isinstance(max_end, (int, float)) else "-"
    return (
        f"{index}. {entry['artist']} - {entry['title']} [{clip_id}] "
        f"duration={duration:.2f}s min_start={min_start_str} max_end={max_end_str}"
    )


def collect_curated_gold_line_bound_violations(
    *,
    manifest_path: Path,
    gold_root: Path,
    tolerance_sec: float,
) -> list[dict[str, object]]:
    songs = _load_manifest(manifest_path)
    by_key, by_artist_title = _build_manifest_lookup(songs)
    violations: list[dict[str, object]] = []
    for gold_path in sorted(gold_root.glob("*.gold.json")):
        try:
            gold_doc = json.loads(gold_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(gold_doc, dict):
            continue
        entry = _violation_entry_from_gold_doc(
            gold_path=gold_path,
            gold_doc=gold_doc,
            by_key=by_key,
            by_artist_title=by_artist_title,
            tolerance_sec=tolerance_sec,
        )
        if entry is not None:
            violations.append(entry)
    violations.sort(key=_sort_violation_key)
    return violations


def _print_report(entries: list[dict[str, object]]) -> None:
    if not entries:
        print("curated_gold_line_bounds: OK")
        return
    print(f"curated_gold_line_bounds: FAIL ({len(entries)} violations)")
    for index, entry in enumerate(entries, start=1):
        print(_format_violation_entry(entry, index))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--gold-root", type=Path, default=DEFAULT_GOLD_ROOT)
    parser.add_argument("--tolerance-sec", type=float, default=0.25)
    args = parser.parse_args()

    entries = collect_curated_gold_line_bound_violations(
        manifest_path=args.manifest.resolve(),
        gold_root=args.gold_root.resolve(),
        tolerance_sec=args.tolerance_sec,
    )
    _print_report(entries)
    return 1 if entries else 0


if __name__ == "__main__":
    raise SystemExit(main())
