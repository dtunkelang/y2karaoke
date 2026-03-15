"""Focused tests for benchmark suite runner bootstrap helpers."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys


def _load_module():
    module_path = (
        Path(__file__).resolve().parents[3] / "tools" / "run_benchmark_suite.py"
    )
    spec = importlib.util.spec_from_file_location("run_benchmark_suite", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_generate_command_includes_expected_flags(tmp_path):
    module = _load_module()
    lyrics_file = tmp_path / "lyrics.txt"
    lyrics_file.write_text("hello\n", encoding="utf-8")
    song = module.BenchmarkSong(
        manifest_index=1,
        artist="X",
        title="Y",
        youtube_id="abcdefghijk",
        youtube_url="https://www.youtube.com/watch?v=abcdefghijk",
        clip_id="intro-30s",
        audio_start_sec=12.5,
        lyrics_file=str(lyrics_file),
    )
    report_path = tmp_path / "report.json"
    cache_dir = tmp_path / "cache"
    cmd = module._build_generate_command(
        python_bin="python",
        song=song,
        report_path=report_path,
        cache_dir=cache_dir,
        offline=True,
        force=True,
        whisper_map_lrc_dtw=True,
        strategy="hybrid_dtw",
        drop_lrc_line_timings=True,
        fast_clip_probe=True,
    )
    assert cmd[:4] == ["python", "-m", "y2karaoke.cli", "generate"]
    assert "--offline" in cmd
    assert "--force" in cmd
    assert "--whisper" not in cmd
    assert "--whisper-map-lrc-dtw" in cmd
    assert "--work-dir" in cmd
    assert str(cache_dir) in cmd
    assert "--drop-lrc-line-timings" in cmd
    assert "--lyrics-file" in cmd
    assert str(lyrics_file) in cmd
    assert "--audio-start" in cmd
    assert "12.5" in cmd
    assert "--skip-separation" in cmd


def test_build_generate_command_fast_clip_probe_skips_non_clip_song(tmp_path):
    module = _load_module()
    song = module.BenchmarkSong(
        manifest_index=1,
        artist="X",
        title="Y",
        youtube_id="abcdefghijk",
        youtube_url="https://www.youtube.com/watch?v=abcdefghijk",
    )
    report_path = tmp_path / "report.json"
    cmd = module._build_generate_command(
        python_bin="python",
        song=song,
        report_path=report_path,
        cache_dir=None,
        offline=False,
        force=False,
        whisper_map_lrc_dtw=True,
        strategy="hybrid_dtw",
        drop_lrc_line_timings=False,
        fast_clip_probe=True,
    )
    assert "--skip-separation" not in cmd


def test_build_generate_command_strategy_variants(tmp_path):
    module = _load_module()
    song = module.BenchmarkSong(
        manifest_index=1,
        artist="X",
        title="Y",
        youtube_id="abcdefghijk",
        youtube_url="https://www.youtube.com/watch?v=abcdefghijk",
    )
    report_path = tmp_path / "report.json"

    cmd_hybrid = module._build_generate_command(
        python_bin="python",
        song=song,
        report_path=report_path,
        cache_dir=None,
        offline=False,
        force=False,
        whisper_map_lrc_dtw=False,
        strategy="hybrid_whisper",
        drop_lrc_line_timings=False,
    )
    assert "--whisper" in cmd_hybrid
    assert "--whisper-map-lrc-dtw" not in cmd_hybrid

    cmd_only = module._build_generate_command(
        python_bin="python",
        song=song,
        report_path=report_path,
        cache_dir=None,
        offline=False,
        force=False,
        whisper_map_lrc_dtw=False,
        strategy="whisper_only",
        drop_lrc_line_timings=False,
    )
    assert "--whisper-only" in cmd_only

    cmd_lrc = module._build_generate_command(
        python_bin="python",
        song=song,
        report_path=report_path,
        cache_dir=None,
        offline=False,
        force=False,
        whisper_map_lrc_dtw=False,
        strategy="lrc_only",
        drop_lrc_line_timings=False,
    )
    assert "--whisper" not in cmd_lrc
    assert "--whisper-only" not in cmd_lrc
    assert "--whisper-map-lrc-dtw" not in cmd_lrc


def test_has_cached_benchmark_source_detects_metadata(tmp_path):
    module = _load_module()
    song = module.BenchmarkSong(
        manifest_index=1,
        artist="X",
        title="Y",
        youtube_id="abcdefghijk",
        youtube_url="https://www.youtube.com/watch?v=abcdefghijk",
    )
    cache_root = tmp_path / ".cache" / "karaoke" / song.youtube_id
    cache_root.mkdir(parents=True)
    (cache_root / "metadata.json").write_text("{}", encoding="utf-8")
    (cache_root / "song.wav").write_text("x", encoding="utf-8")

    module._benchmark_cache_roots = lambda: [tmp_path / ".cache" / "karaoke"]

    assert module._has_cached_benchmark_source(song) is True


def test_has_cached_benchmark_source_rejects_trimmed_only_cache(tmp_path):
    module = _load_module()
    song = module.BenchmarkSong(
        manifest_index=1,
        artist="X",
        title="Y",
        youtube_id="abcdefghijk",
        youtube_url="https://www.youtube.com/watch?v=abcdefghijk",
    )
    cache_root = tmp_path / ".cache" / "karaoke" / song.youtube_id
    cache_root.mkdir(parents=True)
    (cache_root / "metadata.json").write_text("{}", encoding="utf-8")
    (cache_root / "trimmed_from_10.00s.wav").write_text("x", encoding="utf-8")

    module._benchmark_cache_roots = lambda: [tmp_path / ".cache" / "karaoke"]

    assert module._has_cached_benchmark_source(song) is False


def test_run_single_song_generation_auto_enables_offline_for_cached_source(tmp_path):
    module = _load_module()
    song = module.BenchmarkSong(
        manifest_index=1,
        artist="X",
        title="Y",
        youtube_id="abcdefghijk",
        youtube_url="https://www.youtube.com/watch?v=abcdefghijk",
    )
    cache_root = tmp_path / ".cache" / "karaoke" / song.youtube_id
    cache_root.mkdir(parents=True)
    (cache_root / "metadata.json").write_text("{}", encoding="utf-8")
    (cache_root / "song.wav").write_text("x", encoding="utf-8")
    module._benchmark_cache_roots = lambda: [tmp_path / ".cache" / "karaoke"]

    captured: dict[str, object] = {}

    def fake_build_generate_command(**kwargs):
        captured["offline"] = kwargs["offline"]
        return ["python", "-m", "y2karaoke.cli", "generate"]

    def fake_run_song_command(**kwargs):
        return {"status": "ok", "report_path": str(kwargs["report_path"])}

    module._build_generate_command = fake_build_generate_command
    module._run_song_command = fake_run_song_command
    module._load_gold_doc = lambda **_kwargs: None

    class Args:
        python_bin = "python"
        cache_dir = None
        offline = False
        force = False
        no_whisper_map_lrc_dtw = False
        strategy = "hybrid_dtw"
        scenario = "default"
        timeout_sec = 30
        heartbeat_sec = 30
        evaluate_lyrics_sources = False
        rebaseline = False

    record, result_path = module._run_single_song_generation(
        args=Args(),
        index=1,
        total_songs=1,
        song=song,
        run_dir=tmp_path / "run",
        run_signature={},
        gold_root=tmp_path / "gold",
        env={},
    )

    assert captured["offline"] is True
    assert result_path.name == "01_x-y_result.json"
    assert record["status"] == "ok"


def test_infer_compute_substage_uses_shared_cache_without_work_dir(tmp_path):
    module = _load_module()
    cache_root = tmp_path / ".cache" / "karaoke" / "abcdefghijk"
    cache_root.mkdir(parents=True)
    (cache_root / "song_(Vocals)_htdemucs_ft.wav").write_text("", encoding="utf-8")

    module._benchmark_cache_roots = lambda: [tmp_path / ".cache" / "karaoke"]
    stage = module._infer_compute_substage(
        cmd=[
            "python",
            "-m",
            "y2karaoke.cli",
            "generate",
            "https://www.youtube.com/watch?v=abcdefghijk",
        ],
        proc_pid=999999,
        stage_hint=None,
        report_path=tmp_path / "missing.json",
    )

    assert stage == "whisper"


def test_infer_compute_substage_prefers_alignment_when_shared_whisper_cache_exists(
    tmp_path,
):
    module = _load_module()
    cache_root = tmp_path / ".cache" / "karaoke" / "abcdefghijk"
    cache_root.mkdir(parents=True)
    (cache_root / "song_(Vocals)_htdemucs_ft.wav").write_text("", encoding="utf-8")
    (cache_root / "song_whisper_large.json").write_text("{}", encoding="utf-8")

    module._benchmark_cache_roots = lambda: [tmp_path / ".cache" / "karaoke"]
    stage = module._infer_compute_substage(
        cmd=[
            "python",
            "-m",
            "y2karaoke.cli",
            "generate",
            "https://www.youtube.com/watch?v=abcdefghijk",
        ],
        proc_pid=999999,
        stage_hint=None,
        report_path=tmp_path / "missing.json",
    )

    assert stage == "alignment"


def test_parse_manifest_resolves_optional_lyrics_file(tmp_path):
    module = _load_module()
    manifest = tmp_path / "manifest.yaml"
    lyrics = tmp_path / "lyrics" / "song.txt"
    lyrics.parent.mkdir()
    lyrics.write_text("line one\n", encoding="utf-8")
    manifest.write_text(
        "\n".join(
            [
                "songs:",
                "  - artist: Test Artist",
                "    title: Test Song",
                "    youtube_id: abcdefghijk",
                "    youtube_url: https://www.youtube.com/watch?v=abcdefghijk",
                "    clip_id: intro-30s",
                "    audio_start_sec: 18.25",
                "    lyrics_file: lyrics/song.txt",
            ]
        ),
        encoding="utf-8",
    )

    songs = module._parse_manifest(manifest)

    assert len(songs) == 1
    assert songs[0].clip_id == "intro-30s"
    assert songs[0].audio_start_sec == 18.25
    assert songs[0].lyrics_file == str(lyrics.resolve())


def test_parse_manifest_resolves_preferred_lyrics_provider(tmp_path):
    module = _load_module()
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        "\n".join(
            [
                "songs:",
                "  - artist: Test Artist",
                "    title: Test Song",
                "    youtube_id: abcdefghijk",
                "    youtube_url: https://www.youtube.com/watch?v=abcdefghijk",
                "    preferred_lyrics_provider: syncedlyrics",
            ]
        ),
        encoding="utf-8",
    )

    songs = module._parse_manifest(manifest)

    assert len(songs) == 1
    assert songs[0].preferred_lyrics_provider == "syncedlyrics"


def test_parse_manifest_resolves_lrc_duration_tolerance(tmp_path):
    module = _load_module()
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        "\n".join(
            [
                "songs:",
                "  - artist: Test Artist",
                "    title: Test Song",
                "    youtube_id: abcdefghijk",
                "    youtube_url: https://www.youtube.com/watch?v=abcdefghijk",
                "    lrc_duration_tolerance_sec: 18",
            ]
        ),
        encoding="utf-8",
    )

    songs = module._parse_manifest(manifest)

    assert len(songs) == 1
    assert songs[0].lrc_duration_tolerance_sec == 18


def test_run_single_song_generation_exports_preferred_lyrics_provider_env(tmp_path):
    module = _load_module()
    song = module.BenchmarkSong(
        manifest_index=1,
        artist="Artist A",
        title="Alpha",
        youtube_id="aaaaaaaaaaa",
        youtube_url="https://www.youtube.com/watch?v=aaaaaaaaaaa",
        preferred_lyrics_provider="syncedlyrics",
    )
    args = type(
        "Args",
        (),
        {
            "python_bin": "python",
            "cache_dir": None,
            "offline": False,
            "force": False,
            "no_whisper_map_lrc_dtw": False,
            "strategy": "hybrid_whisper",
            "scenario": "default",
            "timeout_sec": 30,
            "heartbeat_sec": 1,
            "evaluate_lyrics_sources": False,
            "rebaseline": False,
        },
    )()
    captured_env: dict[str, str] = {}
    old_build_cmd = module._build_generate_command
    old_load_gold = module._load_gold_doc
    old_run_cmd = module._run_song_command
    module._build_generate_command = lambda **_: ["python", "-m", "noop"]  # type: ignore[assignment]
    module._load_gold_doc = lambda **_: None  # type: ignore[assignment]

    def _fake_run_song_command(**kwargs):
        captured_env.update(kwargs["env"])
        return {"status": "ok", "elapsed_sec": 1.23}

    module._run_song_command = _fake_run_song_command  # type: ignore[assignment]
    try:
        module._run_single_song_generation(
            args=args,
            index=1,
            total_songs=1,
            song=song,
            run_dir=tmp_path,
            run_signature={"k": "v"},
            gold_root=tmp_path,
            env={"BASE_ENV": "1"},
        )
    finally:
        module._build_generate_command = old_build_cmd
        module._load_gold_doc = old_load_gold
        module._run_song_command = old_run_cmd

    assert captured_env["BASE_ENV"] == "1"
    assert captured_env["Y2K_PREFERRED_LYRICS_PROVIDER"] == "syncedlyrics"


def test_run_single_song_generation_disables_auto_offline_for_non_lyriq_preference(
    tmp_path,
):
    module = _load_module()
    song = module.BenchmarkSong(
        manifest_index=1,
        artist="Artist A",
        title="Alpha",
        youtube_id="aaaaaaaaaaa",
        youtube_url="https://www.youtube.com/watch?v=aaaaaaaaaaa",
        preferred_lyrics_provider="syncedlyrics",
    )
    args = type(
        "Args",
        (),
        {
            "python_bin": "python",
            "cache_dir": None,
            "offline": False,
            "force": False,
            "no_whisper_map_lrc_dtw": False,
            "strategy": "hybrid_whisper",
            "scenario": "default",
            "timeout_sec": 30,
            "heartbeat_sec": 1,
            "evaluate_lyrics_sources": False,
            "rebaseline": False,
        },
    )()
    captured_kwargs: dict[str, object] = {}
    old_build_cmd = module._build_generate_command
    old_load_gold = module._load_gold_doc
    old_run_cmd = module._run_song_command
    old_has_cached = module._has_cached_benchmark_source
    module._load_gold_doc = lambda **_: None  # type: ignore[assignment]
    module._has_cached_benchmark_source = lambda _song: True  # type: ignore[assignment]

    def _fake_build_generate_command(**kwargs):
        captured_kwargs.update(kwargs)
        return ["python", "-m", "noop"]

    def _fake_run_song_command(**_kwargs):
        return {"status": "ok", "elapsed_sec": 1.23}

    module._build_generate_command = _fake_build_generate_command  # type: ignore[assignment]
    module._run_song_command = _fake_run_song_command  # type: ignore[assignment]
    try:
        module._run_single_song_generation(
            args=args,
            index=1,
            total_songs=1,
            song=song,
            run_dir=tmp_path,
            run_signature={"k": "v"},
            gold_root=tmp_path,
            env={"BASE_ENV": "1"},
        )
    finally:
        module._build_generate_command = old_build_cmd
        module._load_gold_doc = old_load_gold
        module._run_song_command = old_run_cmd
        module._has_cached_benchmark_source = old_has_cached

    assert captured_kwargs["offline"] is False


def test_augment_alignment_diagnostics_with_song_context_includes_offline_flags():
    module = _load_module()
    song = module.BenchmarkSong(
        manifest_index=1,
        artist="Artist A",
        title="Alpha",
        youtube_id="aaaaaaaaaaa",
        youtube_url="https://www.youtube.com/watch?v=aaaaaaaaaaa",
        preferred_lyrics_provider="syncedlyrics",
        lrc_duration_tolerance_sec=18,
    )
    diag = module._augment_alignment_diagnostics_with_song_context(
        {"lyrics_source_provider": "NetEase"},
        song=song,
        offline=False,
        auto_offline_suppressed=True,
    )
    assert diag["preferred_lyrics_provider_requested"] == "syncedlyrics"
    assert diag["lrc_duration_tolerance_sec_requested"] == 18
    assert diag["benchmark_offline_mode"] is False
    assert diag["benchmark_auto_offline_suppressed_for_provider_preference"] is True


def test_run_single_song_generation_exports_duration_tolerance_env(tmp_path):
    module = _load_module()
    song = module.BenchmarkSong(
        manifest_index=1,
        artist="Artist A",
        title="Alpha",
        youtube_id="aaaaaaaaaaa",
        youtube_url="https://www.youtube.com/watch?v=aaaaaaaaaaa",
        lrc_duration_tolerance_sec=18,
    )
    args = type(
        "Args",
        (),
        {
            "python_bin": "python",
            "cache_dir": None,
            "offline": False,
            "force": False,
            "no_whisper_map_lrc_dtw": False,
            "strategy": "hybrid_whisper",
            "scenario": "default",
            "timeout_sec": 30,
            "heartbeat_sec": 1,
            "evaluate_lyrics_sources": False,
            "rebaseline": False,
        },
    )()
    captured_env: dict[str, str] = {}
    old_build_cmd = module._build_generate_command
    old_load_gold = module._load_gold_doc
    old_run_cmd = module._run_song_command
    module._build_generate_command = lambda **_: ["python", "-m", "noop"]  # type: ignore[assignment]
    module._load_gold_doc = lambda **_: None  # type: ignore[assignment]

    def _fake_run_song_command(**kwargs):
        captured_env.update(kwargs["env"])
        return {"status": "ok", "elapsed_sec": 1.23}

    module._run_song_command = _fake_run_song_command  # type: ignore[assignment]
    try:
        module._run_single_song_generation(
            args=args,
            index=1,
            total_songs=1,
            song=song,
            run_dir=tmp_path,
            run_signature={"k": "v"},
            gold_root=tmp_path,
            env={},
        )
    finally:
        module._build_generate_command = old_build_cmd
        module._load_gold_doc = old_load_gold
        module._run_song_command = old_run_cmd

    assert captured_env["Y2K_LRC_DURATION_TOLERANCE_SEC"] == "18"


def test_benchmark_song_slug_includes_clip_id():
    module = _load_module()
    song = module.BenchmarkSong(
        manifest_index=1,
        artist="The Weepies",
        title="Take It From Me",
        youtube_id="abcdefghijk",
        youtube_url="https://www.youtube.com/watch?v=abcdefghijk",
        clip_id="Outro Reprise",
    )

    assert song.slug == "the-weepies-take-it-from-me-outro-reprise"


def test_resolve_run_dir_resume_latest(tmp_path):
    module = _load_module()
    older = tmp_path / "20260101T000000Z"
    newer = tmp_path / "20260102T000000Z"
    older.mkdir(parents=True)
    newer.mkdir(parents=True)

    now = 1739450000.0
    os.utime(older, (now, now))
    os.utime(newer, (now, now))

    run_dir, run_id = module._resolve_run_dir(
        output_root=tmp_path,
        run_id=None,
        resume_run_dir=None,
        resume_latest=True,
    )
    assert run_dir == newer
    assert run_id == "20260102T000000Z"


def test_load_song_result(tmp_path):
    module = _load_module()
    path = tmp_path / "song_result.json"
    assert module._load_song_result(path) is None
    path.write_text('{"status": "ok", "title": "x"}', encoding="utf-8")
    loaded = module._load_song_result(path)
    assert loaded is not None
    assert loaded["status"] == "ok"


def test_discover_cached_result_slugs(tmp_path):
    module = _load_module()
    (tmp_path / "01_bad-guy_result.json").write_text("{}", encoding="utf-8")
    (tmp_path / "02_shape-of-you_result.json").write_text("{}", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    slugs = module._discover_cached_result_slugs(tmp_path)
    assert slugs == {"bad-guy", "shape-of-you"}


def test_refresh_cached_metrics(tmp_path):
    module = _load_module()
    report_path = tmp_path / "timing_report.json"
    report_path.write_text(
        (
            '{"alignment_method":"whisper_hybrid","dtw_line_coverage":0.9,'
            '"dtw_word_coverage":0.8,"dtw_phonetic_similarity_coverage":0.7,'
            '"low_confidence_lines":[],"lines":[{"whisper_line_start_delta":0.1}]}'
        ),
        encoding="utf-8",
    )
    record = {
        "status": "ok",
        "report_path": str(report_path),
    }
    refreshed = module._refresh_cached_metrics(record)
    assert refreshed["metrics"]["alignment_method"] == "whisper_hybrid"
    assert refreshed["metrics"]["dtw_line_coverage"] == 0.9


def test_refresh_metrics_adds_requested_provider_context(tmp_path):
    module = _load_module()
    report = {"alignment_method": "whisper_hybrid", "lyrics_source": "NetEase"}
    song = module.BenchmarkSong(
        manifest_index=1,
        artist="Artist A",
        title="Alpha",
        youtube_id="aaaaaaaaaaa",
        youtube_url="https://www.youtube.com/watch?v=aaaaaaaaaaa",
        preferred_lyrics_provider="syncedlyrics",
        lrc_duration_tolerance_sec=18,
    )
    refreshed = module._refresh_metrics_from_loaded_report(
        {"status": "ok"},
        report=report,
        index=1,
        song=song,
        gold_root=tmp_path,
    )
    diag = refreshed["alignment_diagnostics"]
    assert diag["preferred_lyrics_provider_requested"] == "syncedlyrics"
    assert diag["lrc_duration_tolerance_sec_requested"] == 18


def test_load_aggregate_only_results_keeps_line_overlapping_clip_start(tmp_path):
    module = _load_module()
    song = module.BenchmarkSong(
        manifest_index=1,
        artist="The Weeknd",
        title="Blinding Lights",
        youtube_id="fHI8X4OXluQ",
        youtube_url="https://www.youtube.com/watch?v=fHI8X4OXluQ",
        clip_id="early-verse",
        audio_start_sec=27.0,
    )
    gold_path = tmp_path / f"01_{song.slug}.gold.json"
    gold_path.write_text(
        json.dumps(
            {
                "artist": song.artist,
                "title": song.title,
                "youtube_id": song.youtube_id,
                "clip_id": song.clip_id,
                "audio_start_sec": song.audio_start_sec,
                "lines": [
                    {
                        "index": 1,
                        "text": "I've been tryna call",
                        "start": 0.0,
                        "end": 1.08,
                        "words": [
                            {"text": "I've", "start": 0.0, "end": 0.25},
                            {"text": "been", "start": 0.25, "end": 0.52},
                            {"text": "tryna", "start": 0.52, "end": 0.8},
                            {"text": "call", "start": 0.8, "end": 1.08},
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    report_path = tmp_path / f"01_{song.base_slug}_timing_report.json"
    report_path.write_text(
        json.dumps(
            {
                "alignment_method": "whisper_hybrid",
                "lines": [
                    {
                        "index": 4,
                        "start": 26.95,
                        "end": 28.03,
                        "text": "I've been tryna call",
                        "words": [
                            {"text": "I've", "start": 26.95, "end": 27.2},
                            {"text": "been", "start": 27.2, "end": 27.47},
                            {"text": "tryna", "start": 27.47, "end": 27.75},
                            {"text": "call", "start": 27.75, "end": 28.03},
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / f"01_{song.base_slug}_result.json").write_text(
        json.dumps(
            {
                "artist": song.artist,
                "title": song.title,
                "youtube_id": song.youtube_id,
                "report_path": str(report_path),
                "status": "ok",
            }
        ),
        encoding="utf-8",
    )

    rows, skipped = module._load_aggregate_only_results(
        songs=[song],
        run_dir=tmp_path,
        gold_root=tmp_path,
        rebaseline=False,
    )

    assert skipped == []
    assert len(rows) == 1
    assert rows[0]["status"] == "ok"
    assert rows[0]["clip_scored_from_full_song"] is True
    assert rows[0]["aggregate_only_recomputed"] is True
    assert rows[0]["metrics"]["gold_word_coverage_ratio"] == 1.0


def test_build_run_signature(tmp_path):
    module = _load_module()
    args = module.argparse.Namespace(
        offline=True,
        force=False,
        strategy="hybrid_whisper",
        scenario="lyrics_no_timing",
        no_whisper_map_lrc_dtw=True,
        cache_dir=tmp_path,
        min_timing_quality_score_line_weighted=0.58,
    )
    sig = module._build_run_signature(args, tmp_path / "manifest.yaml")
    assert sig["manifest_path"].endswith("manifest.yaml")
    assert sig["offline"] is True
    assert sig["strategy"] == "hybrid_whisper"
    assert sig["scenario"] == "lyrics_no_timing"
    assert sig["whisper_map_lrc_dtw"] is False
    assert sig["cache_dir"] == str(tmp_path.resolve())
    assert sig["min_timing_quality_score_line_weighted"] == 0.58
