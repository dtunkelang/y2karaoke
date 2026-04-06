"""Focused tests for benchmark manifest validation."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_module():
    module_path = (
        Path(__file__).resolve().parents[3] / "tools" / "validate_benchmark_manifest.py"
    )
    spec = importlib.util.spec_from_file_location(
        "validate_benchmark_manifest", module_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_validate_manifest_allows_duplicate_artist_title_when_clip_id_differs(tmp_path):
    module = _load_module()
    manifest = tmp_path / "manifest.yaml"
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
                "    lrc_duration_tolerance_sec: 30",
                "    clip_id: intro",
                "    clip_tags:",
                "      - control",
                "    audio_start_sec: 0",
                "    clip_duration_sec: 10",
                "    notes: Intro control clip",
                "  - artist: Artist",
                "    title: Song",
                "    youtube_id: lmnopqrstuv",
                "    youtube_url: https://www.youtube.com/watch?v=lmnopqrstuv",
                "    preferred_lyrics_provider: lyriq",
                "    fallback_lyrics_provider: syncedlyrics",
                "    lrc_duration_tolerance_sec: 30",
                "    clip_id: outro",
                "    clip_tags:",
                "      - tail",
                "    audio_start_sec: 90",
                "    clip_duration_sec: 10",
                "    notes: Outro stress clip",
            ]
        ),
        encoding="utf-8",
    )

    assert module.validate_manifest(manifest) == []


def test_validate_manifest_rejects_duplicate_artist_title_same_clip_id(tmp_path):
    module = _load_module()
    manifest = tmp_path / "manifest.yaml"
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
                "    lrc_duration_tolerance_sec: 30",
                "    clip_id: intro",
                "    clip_tags:",
                "      - control",
                "    audio_start_sec: 0",
                "    clip_duration_sec: 10",
                "    notes: Intro control clip",
                "  - artist: Artist",
                "    title: Song",
                "    youtube_id: lmnopqrstuv",
                "    youtube_url: https://www.youtube.com/watch?v=lmnopqrstuv",
                "    preferred_lyrics_provider: lyriq",
                "    fallback_lyrics_provider: syncedlyrics",
                "    lrc_duration_tolerance_sec: 30",
                "    clip_id: intro",
                "    clip_tags:",
                "      - control",
                "    audio_start_sec: 90",
                "    clip_duration_sec: 10",
                "    notes: Duplicate intro clip",
            ]
        ),
        encoding="utf-8",
    )

    errors = module.validate_manifest(manifest)

    assert any("Duplicate artist/title[/clip_id]" in error for error in errors)


def test_validate_manifest_allows_duplicate_youtube_id_when_clip_id_differs(tmp_path):
    module = _load_module()
    manifest = tmp_path / "manifest.yaml"
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
                "    lrc_duration_tolerance_sec: 30",
                "    clip_id: intro",
                "    clip_tags:",
                "      - control",
                "    audio_start_sec: 0",
                "    clip_duration_sec: 10",
                "    notes: Intro control clip",
                "  - artist: Artist",
                "    title: Song",
                "    youtube_id: abcdefghijk",
                "    youtube_url: https://www.youtube.com/watch?v=abcdefghijk",
                "    preferred_lyrics_provider: lyriq",
                "    fallback_lyrics_provider: syncedlyrics",
                "    lrc_duration_tolerance_sec: 30",
                "    clip_id: outro",
                "    clip_tags:",
                "      - tail",
                "    audio_start_sec: 90",
                "    clip_duration_sec: 10",
                "    notes: Outro stress clip",
            ]
        ),
        encoding="utf-8",
    )

    assert module.validate_manifest(manifest) == []


def test_validate_manifest_rejects_duplicate_youtube_id_same_clip_id(tmp_path):
    module = _load_module()
    manifest = tmp_path / "manifest.yaml"
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
                "    lrc_duration_tolerance_sec: 30",
                "    clip_id: intro",
                "    clip_tags:",
                "      - control",
                "    audio_start_sec: 0",
                "    clip_duration_sec: 10",
                "    notes: Intro control clip",
                "  - artist: Artist",
                "    title: Song B",
                "    youtube_id: abcdefghijk",
                "    youtube_url: https://www.youtube.com/watch?v=abcdefghijk",
                "    preferred_lyrics_provider: lyriq",
                "    fallback_lyrics_provider: syncedlyrics",
                "    lrc_duration_tolerance_sec: 30",
                "    clip_id: intro",
                "    clip_tags:",
                "      - control",
                "    audio_start_sec: 90",
                "    clip_duration_sec: 10",
                "    notes: Duplicate intro clip",
            ]
        ),
        encoding="utf-8",
    )

    errors = module.validate_manifest(manifest)

    assert any("Duplicate youtube_id[/clip_id]" in error for error in errors)


def test_validate_manifest_rejects_negative_audio_start_sec(tmp_path):
    module = _load_module()
    manifest = tmp_path / "manifest.yaml"
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
                "    lrc_duration_tolerance_sec: 30",
                "    clip_id: intro",
                "    clip_tags:",
                "      - control",
                "    audio_start_sec: -1",
                "    clip_duration_sec: 10",
                "    notes: Invalid intro clip",
            ]
        ),
        encoding="utf-8",
    )

    errors = module.validate_manifest(manifest)

    assert "songs[0].audio_start_sec must be >= 0" in errors


def test_validate_manifest_rejects_non_positive_clip_duration_sec(tmp_path):
    module = _load_module()
    manifest = tmp_path / "manifest.yaml"
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
                "    lrc_duration_tolerance_sec: 30",
                "    clip_id: intro",
                "    clip_tags:",
                "      - control",
                "    audio_start_sec: 1",
                "    clip_duration_sec: 0",
                "    notes: Invalid duration",
            ]
        ),
        encoding="utf-8",
    )

    errors = module.validate_manifest(manifest)

    assert "songs[0].clip_duration_sec must be > 0" in errors


def test_validate_manifest_requires_clip_tags_for_clip_entries(tmp_path):
    module = _load_module()
    manifest = tmp_path / "manifest.yaml"
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
                "    lrc_duration_tolerance_sec: 30",
                "    clip_id: intro",
                "    audio_start_sec: 0",
                "    clip_duration_sec: 10",
                "    notes: Intro clip",
            ]
        ),
        encoding="utf-8",
    )

    errors = module.validate_manifest(manifest)

    assert "songs[0].clip_tags are required for clip entries" in errors


def test_validate_manifest_rejects_duplicate_clip_tags(tmp_path):
    module = _load_module()
    manifest = tmp_path / "manifest.yaml"
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
                "    lrc_duration_tolerance_sec: 30",
                "    clip_id: intro",
                "    clip_tags:",
                "      - control",
                "      - Control",
                "    audio_start_sec: 0",
                "    clip_duration_sec: 10",
                "    notes: Intro clip",
            ]
        ),
        encoding="utf-8",
    )

    errors = module.validate_manifest(manifest)

    assert "songs[0].clip_tags must not contain duplicates" in errors


def test_validate_manifest_requires_clip_duration_for_clip_entries(tmp_path):
    module = _load_module()
    manifest = tmp_path / "manifest.yaml"
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
                "    lrc_duration_tolerance_sec: 30",
                "    clip_id: intro",
                "    clip_tags:",
                "      - control",
                "    audio_start_sec: 0",
                "    notes: Intro clip",
            ]
        ),
        encoding="utf-8",
    )

    errors = module.validate_manifest(manifest)

    assert "songs[0].clip_duration_sec is required for clip entries" in errors
