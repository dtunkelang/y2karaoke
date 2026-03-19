"""Cache and result-loading tests for benchmark suite runner helpers."""

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
