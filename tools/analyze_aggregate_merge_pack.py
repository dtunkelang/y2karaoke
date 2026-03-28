#!/usr/bin/env python3
"""Scan curated clips for aggregate-only adjacent line merges."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from tools import analyze_aggregate_segment_merges as merge_tool

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLD_ROOT = (
    REPO_ROOT / "benchmarks" / "clip_gold_candidate" / "20260312T_curated_clips"
)


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _song_slug(song: dict[str, object]) -> str:
    base = _slugify(f"{song['artist']}-{song['title']}")
    clip_id = str(song.get("clip_id") or "").strip()
    clip_slug = _slugify(clip_id)
    return f"{base}-{clip_slug}" if clip_slug else base


def _load_manifest(path: Path) -> list[dict[str, object]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    songs = raw.get("songs", []) if isinstance(raw, dict) else []
    return [song for song in songs if isinstance(song, dict)]


def _gold_path(index: int, song: dict[str, object]) -> Path:
    slug = _song_slug(song)
    indexed = DEFAULT_GOLD_ROOT / f"{index:02d}_{slug}.gold.json"
    if indexed.exists():
        return indexed
    matches = sorted(DEFAULT_GOLD_ROOT.glob(f"*_{slug}.gold.json"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No gold file found for {slug}")


def _canonical_trimmed_clip_path(song: dict[str, object]) -> Path:
    start = _to_float(song.get("audio_start_sec"))
    duration = _to_float(song.get("clip_duration_sec"))
    cache_dir = Path.home() / ".cache" / "karaoke" / str(song["youtube_id"])
    return cache_dir / f"trimmed_from_{start:.2f}s_for_{duration:.2f}s.wav"


def _to_float(value: object) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value))


def _label(song: dict[str, object]) -> str:
    return f"{song['artist']} - {song['title']}"


def _matches(song: dict[str, object], pattern: str | None) -> bool:
    if not pattern:
        return True
    haystack = " ".join(
        [
            str(song.get("artist", "")),
            str(song.get("title", "")),
            str(song.get("clip_id", "")),
        ]
    )
    return re.search(pattern, haystack, re.IGNORECASE) is not None


def _cache_path_for(song: dict[str, object], *, aggressive: bool) -> Path | None:
    clip_audio = _canonical_trimmed_clip_path(song)
    stem = clip_audio.stem
    pattern = f"{stem}*_whisper_*.json"
    candidates = sorted(clip_audio.parent.glob(pattern))
    if aggressive:
        mode_candidates = [path for path in candidates if path.stem.endswith("_aggr")]
    else:
        mode_candidates = [
            path for path in candidates if not path.stem.endswith("_aggr")
        ]
    if not mode_candidates:
        return None
    preferred = sorted(
        mode_candidates,
        key=lambda path: (
            "large" not in path.name,
            "_en" not in path.name and "_auto" not in path.name,
            path.name,
        ),
    )
    return preferred[0]


def analyze(
    *,
    manifest_path: Path,
    match: str | None = None,
) -> dict[str, Any]:
    songs = _load_manifest(manifest_path)
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    songs_scanned = 0

    for index, song in enumerate(songs, start=1):
        if not _matches(song, match):
            continue
        if song.get("audio_start_sec") is None or song.get("clip_duration_sec") is None:
            continue
        songs_scanned += 1
        gold_path = _gold_path(index, song)
        aggregate_path = _cache_path_for(song, aggressive=True)
        vocals_path = _cache_path_for(song, aggressive=False)
        if aggregate_path is None or vocals_path is None:
            skipped.append(
                {
                    "song": _label(song),
                    "reason": "missing_cache",
                }
            )
            continue
        payload = merge_tool.analyze(
            aggregate_path=aggregate_path,
            vocals_path=vocals_path,
            gold_path=gold_path,
        )
        if int(payload.get("merge_count", 0) or 0) <= 0:
            continue
        rows.append(
            {
                "song": _label(song),
                "clip_id": str(song.get("clip_id", "")),
                "gold_path": str(gold_path),
                "aggregate_path": str(aggregate_path),
                "vocals_path": str(vocals_path),
                "merge_count": int(payload["merge_count"]),
                "rows": payload["rows"],
            }
        )

    return {
        "manifest_path": str(manifest_path),
        "songs_analyzed": songs_scanned,
        "songs_with_merges": len(rows),
        "rows": rows,
        "skipped": skipped,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("benchmarks/curated_clip_songs.yaml"),
    )
    parser.add_argument("--match", type=str, default=None)
    args = parser.parse_args()
    print(
        json.dumps(
            analyze(
                manifest_path=args.manifest,
                match=args.match,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
