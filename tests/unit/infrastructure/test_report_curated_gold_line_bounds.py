from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

_MODULE_PATH = (
    Path(__file__).resolve().parents[3] / "tools" / "report_curated_gold_line_bounds.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "report_curated_gold_line_bounds", _MODULE_PATH
)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_collect_curated_gold_line_bound_violations_flags_line_end_beyond_clip(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "curated_clip_songs.yaml"
    manifest.write_text(
        "\n".join(
            [
                "version: 1",
                "name: Clip Pack",
                "songs:",
                "  - artist: Artist",
                "    title: Song",
                "    youtube_id: abcdefghijk",
                "    youtube_url: https://www.youtube.com/watch?v=abcdefghijk",
                "    preferred_lyrics_provider: lyriq",
                "    fallback_lyrics_provider: syncedlyrics",
                "    lrc_duration_tolerance_sec: 10",
                "    clip_id: chorus",
                "    clip_tags:",
                "      - control",
                "    audio_start_sec: 0",
                "    clip_duration_sec: 22.0",
                "    notes: Test clip",
            ]
        ),
        encoding="utf-8",
    )
    gold_root = tmp_path / "gold"
    gold_root.mkdir()
    gold_root.joinpath("01_artist-song-chorus.gold.json").write_text(
        json.dumps(
            {
                "artist": "Artist",
                "title": "Song [chorus]",
                "audio_path": "/tmp/artist-song-0s-22s.wav",
                "lines": [
                    {"line_index": 1, "text": "ok", "start": 0.2, "end": 1.8},
                    {"line_index": 2, "text": "bad", "start": 21.5, "end": 24.1},
                ],
            }
        ),
        encoding="utf-8",
    )

    entries = _MODULE.collect_curated_gold_line_bound_violations(
        manifest_path=manifest,
        gold_root=gold_root,
        tolerance_sec=0.25,
    )

    assert len(entries) == 1
    assert entries[0]["clip_id"] == "chorus"
    assert entries[0]["max_end_sec"] == 24.1


def test_collect_curated_gold_line_bound_violations_flags_negative_start(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "curated_clip_songs.yaml"
    manifest.write_text(
        "\n".join(
            [
                "version: 1",
                "name: Clip Pack",
                "songs:",
                "  - artist: Artist",
                "    title: Song",
                "    youtube_id: abcdefghijk",
                "    youtube_url: https://www.youtube.com/watch?v=abcdefghijk",
                "    preferred_lyrics_provider: lyriq",
                "    fallback_lyrics_provider: syncedlyrics",
                "    lrc_duration_tolerance_sec: 10",
                "    clip_id: intro",
                "    clip_tags:",
                "      - control",
                "    audio_start_sec: 0",
                "    clip_duration_sec: 12.0",
                "    notes: Test clip",
            ]
        ),
        encoding="utf-8",
    )
    gold_root = tmp_path / "gold"
    gold_root.mkdir()
    gold_root.joinpath("01_artist-song-intro.gold.json").write_text(
        json.dumps(
            {
                "artist": "Artist",
                "title": "Song [intro]",
                "audio_path": "/tmp/artist-song-0s-12s.wav",
                "lines": [
                    {"line_index": 1, "text": "bad", "start": -0.5, "end": 1.0},
                ],
            }
        ),
        encoding="utf-8",
    )

    entries = _MODULE.collect_curated_gold_line_bound_violations(
        manifest_path=manifest,
        gold_root=gold_root,
        tolerance_sec=0.25,
    )

    assert len(entries) == 1
    assert entries[0]["min_start_sec"] == -0.5


def test_collect_curated_gold_line_bound_violations_skips_valid_file(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "curated_clip_songs.yaml"
    manifest.write_text(
        "\n".join(
            [
                "version: 1",
                "name: Clip Pack",
                "songs:",
                "  - artist: Artist",
                "    title: Song",
                "    youtube_id: abcdefghijk",
                "    youtube_url: https://www.youtube.com/watch?v=abcdefghijk",
                "    preferred_lyrics_provider: lyriq",
                "    fallback_lyrics_provider: syncedlyrics",
                "    lrc_duration_tolerance_sec: 10",
                "    clip_id: verse",
                "    clip_tags:",
                "      - control",
                "    audio_start_sec: 0",
                "    clip_duration_sec: 12.0",
                "    notes: Test clip",
            ]
        ),
        encoding="utf-8",
    )
    gold_root = tmp_path / "gold"
    gold_root.mkdir()
    gold_root.joinpath("01_artist-song-verse.gold.json").write_text(
        json.dumps(
            {
                "artist": "Artist",
                "title": "Song [verse]",
                "audio_path": "/tmp/artist-song-0s-12s.wav",
                "lines": [
                    {"line_index": 1, "text": "ok", "start": 0.1, "end": 1.0},
                    {"line_index": 2, "text": "ok", "start": 9.0, "end": 11.8},
                ],
            }
        ),
        encoding="utf-8",
    )

    entries = _MODULE.collect_curated_gold_line_bound_violations(
        manifest_path=manifest,
        gold_root=gold_root,
        tolerance_sec=0.25,
    )

    assert entries == []
