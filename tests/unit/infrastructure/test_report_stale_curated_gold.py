from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


_MODULE_PATH = (
    Path(__file__).resolve().parents[3] / "tools" / "report_stale_curated_gold.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "report_stale_curated_gold", _MODULE_PATH
)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_collect_stale_gold_entries_matches_unique_artist_title_without_clip_suffix(
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
                "    clip_duration_sec: 34.2",
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
                "title": "Song",
                "audio_path": "/tmp/artist-song-0s-30s.wav",
                "lines": [],
            }
        ),
        encoding="utf-8",
    )

    entries = _MODULE.collect_stale_gold_entries(
        manifest_path=manifest,
        gold_root=gold_root,
        tolerance_sec=0.5,
    )

    assert len(entries) == 1
    assert entries[0]["clip_id"] == "verse"
    assert entries[0]["manifest_duration_sec"] == 34.2
    assert entries[0]["gold_audio_duration_sec"] == 30.0


def test_collect_stale_gold_entries_prefers_explicit_clip_suffix_match(
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
                "    clip_duration_sec: 30.0",
                "    notes: Intro clip",
                "  - artist: Artist",
                "    title: Song",
                "    youtube_id: lmnopqrstuv",
                "    youtube_url: https://www.youtube.com/watch?v=lmnopqrstuv",
                "    preferred_lyrics_provider: lyriq",
                "    fallback_lyrics_provider: syncedlyrics",
                "    lrc_duration_tolerance_sec: 10",
                "    clip_id: outro",
                "    clip_tags:",
                "      - tail",
                "    audio_start_sec: 60",
                "    clip_duration_sec: 34.2",
                "    notes: Outro clip",
            ]
        ),
        encoding="utf-8",
    )
    gold_root = tmp_path / "gold"
    gold_root.mkdir()
    gold_root.joinpath("01_artist-song-outro.gold.json").write_text(
        json.dumps(
            {
                "artist": "Artist",
                "title": "Song [outro]",
                "audio_path": "/tmp/artist-song-outro-60s-90s.wav",
                "lines": [],
            }
        ),
        encoding="utf-8",
    )

    entries = _MODULE.collect_stale_gold_entries(
        manifest_path=manifest,
        gold_root=gold_root,
        tolerance_sec=0.5,
    )

    assert len(entries) == 1
    assert entries[0]["clip_id"] == "outro"
    assert entries[0]["manifest_duration_sec"] == 34.2


def test_collect_stale_gold_entries_sorts_by_absolute_delta_desc(
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
                "    title: Song A",
                "    youtube_id: abcdefghijk",
                "    youtube_url: https://www.youtube.com/watch?v=abcdefghijk",
                "    preferred_lyrics_provider: lyriq",
                "    fallback_lyrics_provider: syncedlyrics",
                "    lrc_duration_tolerance_sec: 10",
                "    clip_id: first",
                "    clip_tags:",
                "      - control",
                "    audio_start_sec: 0",
                "    clip_duration_sec: 31.0",
                "    notes: First clip",
                "  - artist: Artist",
                "    title: Song B",
                "    youtube_id: lmnopqrstuv",
                "    youtube_url: https://www.youtube.com/watch?v=lmnopqrstuv",
                "    preferred_lyrics_provider: lyriq",
                "    fallback_lyrics_provider: syncedlyrics",
                "    lrc_duration_tolerance_sec: 10",
                "    clip_id: second",
                "    clip_tags:",
                "      - control",
                "    audio_start_sec: 0",
                "    clip_duration_sec: 35.0",
                "    notes: Second clip",
            ]
        ),
        encoding="utf-8",
    )
    gold_root = tmp_path / "gold"
    gold_root.mkdir()
    gold_root.joinpath("01_artist-song-a-first.gold.json").write_text(
        json.dumps(
            {
                "artist": "Artist",
                "title": "Song A [first]",
                "audio_path": "/tmp/artist-song-a-0s-30s.wav",
                "lines": [],
            }
        ),
        encoding="utf-8",
    )
    gold_root.joinpath("02_artist-song-b-second.gold.json").write_text(
        json.dumps(
            {
                "artist": "Artist",
                "title": "Song B [second]",
                "audio_path": "/tmp/artist-song-b-0s-30s.wav",
                "lines": [],
            }
        ),
        encoding="utf-8",
    )

    entries = _MODULE.collect_stale_gold_entries(
        manifest_path=manifest,
        gold_root=gold_root,
        tolerance_sec=0.5,
    )

    assert [entry["clip_id"] for entry in entries] == ["second", "first"]


def test_print_report_includes_one_based_indices(capsys) -> None:
    _MODULE._print_report(
        [
            {
                "artist": "Artist",
                "title": "Song",
                "clip_id": "first",
                "manifest_duration_sec": 35.0,
                "gold_audio_duration_sec": 30.0,
                "delta_sec": 5.0,
            }
        ]
    )

    captured = capsys.readouterr()

    assert (
        "1. Artist - Song [first] manifest=35.00s gold=30.00s delta=+5.00s"
        in captured.out
    )
