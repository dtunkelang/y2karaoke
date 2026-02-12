"""Unit tests for benchmark suite runner utilities."""

from __future__ import annotations

import importlib.util
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


def test_extract_song_metrics():
    module = _load_module()
    report = {
        "dtw_line_coverage": 0.8,
        "dtw_word_coverage": 0.7,
        "dtw_phonetic_similarity_coverage": 0.75,
        "low_confidence_lines": [{"index": 2}],
        "lines": [
            {"whisper_line_start_delta": -0.2},
            {"whisper_line_start_delta": 0.1},
            {"whisper_line_start_delta": None},
        ],
    }
    metrics = module._extract_song_metrics(report)
    assert metrics["line_count"] == 3
    assert metrics["low_confidence_lines"] == 1
    assert metrics["low_confidence_ratio"] == 0.3333
    assert metrics["dtw_line_coverage"] == 0.8
    assert metrics["start_delta_count"] == 2
    assert metrics["start_delta_mean_abs_sec"] == 0.15


def test_aggregate_results():
    module = _load_module()
    results = [
        {
            "artist": "A",
            "title": "T1",
            "status": "ok",
            "metrics": {
                "line_count": 10,
                "low_confidence_lines": 1,
                "dtw_line_coverage": 0.9,
                "dtw_word_coverage": 0.8,
                "dtw_phonetic_similarity_coverage": 0.85,
                "start_delta_mean_abs_sec": 0.11,
                "start_delta_p95_abs_sec": 0.24,
            },
        },
        {"artist": "B", "title": "T2", "status": "failed"},
    ]
    agg = module._aggregate(results)
    assert agg["songs_total"] == 2
    assert agg["songs_succeeded"] == 1
    assert agg["songs_failed"] == 1
    assert agg["line_count_total"] == 10
    assert agg["low_confidence_lines_total"] == 1
    assert agg["dtw_line_coverage_mean"] == 0.9
    assert agg["dtw_line_coverage_line_weighted_mean"] == 0.9
    assert agg["dtw_metric_song_count"] == 1
    assert agg["dtw_metric_song_coverage_ratio"] == 1.0
    assert agg["dtw_metric_line_count"] == 10
    assert agg["dtw_metric_line_coverage_ratio"] == 1.0
    assert agg["sum_song_elapsed_sec"] == 0.0
    assert agg["failed_songs"] == ["B - T2"]


def test_build_generate_command_includes_expected_flags(tmp_path):
    module = _load_module()
    song = module.BenchmarkSong(
        artist="X",
        title="Y",
        youtube_id="abcdefghijk",
        youtube_url="https://www.youtube.com/watch?v=abcdefghijk",
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
    )
    assert cmd[:4] == ["python", "-m", "y2karaoke.cli", "generate"]
    assert "--offline" in cmd
    assert "--force" in cmd
    assert "--whisper-map-lrc-dtw" in cmd
    assert "--work-dir" in cmd
    assert str(cache_dir) in cmd


def test_resolve_run_dir_resume_latest(tmp_path):
    module = _load_module()
    older = tmp_path / "20260101T000000Z"
    newer = tmp_path / "20260102T000000Z"
    older.mkdir(parents=True)
    newer.mkdir(parents=True)

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


def test_build_run_signature(tmp_path):
    module = _load_module()
    args = module.argparse.Namespace(
        offline=True,
        force=False,
        no_whisper_map_lrc_dtw=True,
        cache_dir=tmp_path,
    )
    sig = module._build_run_signature(args, tmp_path / "manifest.yaml")
    assert sig["manifest_path"].endswith("manifest.yaml")
    assert sig["offline"] is True
    assert sig["whisper_map_lrc_dtw"] is False
    assert sig["cache_dir"] == str(tmp_path.resolve())


def test_aggregate_tracks_missing_dtw_and_weighted_means():
    module = _load_module()
    results = [
        {
            "artist": "A",
            "title": "T1",
            "status": "ok",
            "elapsed_sec": 12.0,
            "phase_durations_sec": {"separation": 10.0, "alignment": 1.0},
            "metrics": {
                "line_count": 100,
                "low_confidence_lines": 5,
                "low_confidence_ratio": 0.05,
                "dtw_line_coverage": 0.5,
                "dtw_word_coverage": 0.4,
                "dtw_phonetic_similarity_coverage": 0.3,
                "start_delta_mean_abs_sec": 0.2,
            },
        },
        {
            "artist": "B",
            "title": "T2",
            "status": "ok",
            "elapsed_sec": 8.0,
            "phase_durations_sec": {"separation": 6.0, "whisper": 1.0},
            "metrics": {
                "line_count": 50,
                "low_confidence_lines": 1,
                "low_confidence_ratio": 0.02,
                "start_delta_mean_abs_sec": 0.1,
            },
        },
    ]
    agg = module._aggregate(results)
    assert agg["dtw_metric_song_count"] == 1
    assert agg["dtw_metric_song_coverage_ratio"] == 0.5
    assert agg["dtw_metric_line_count"] == 100
    assert agg["dtw_metric_line_coverage_ratio"] == 0.6667
    assert agg["songs_without_dtw_metrics"] == ["B - T2"]
    assert agg["dtw_line_coverage_mean"] == 0.5
    assert agg["dtw_line_coverage_line_weighted_mean"] == 0.5
    assert agg["sum_song_elapsed_sec"] == 20.0
    assert agg["phase_totals_sec"]["separation"] == 16.0
    assert agg["cache_summary"]["separation"]["miss_count"] == 0
    assert agg["cache_summary"]["separation"]["total"] == 0


def test_quality_coverage_warnings():
    module = _load_module()
    aggregate = {
        "dtw_metric_song_coverage_ratio": 0.5,
        "dtw_metric_line_coverage_ratio": 0.4,
        "sum_song_elapsed_sec": 12.0,
    }
    warnings = module._quality_coverage_warnings(
        aggregate=aggregate,
        dtw_enabled=False,
        min_song_coverage_ratio=0.8,
        min_line_coverage_ratio=0.9,
        suite_wall_elapsed_sec=10.0,
    )
    assert len(warnings) == 4


def test_cache_expectation_warnings():
    module = _load_module()
    aggregate = {
        "cache_summary": {
            "separation": {"total": 2, "miss_count": 1, "cached_ratio": 0.5},
            "whisper": {"total": 2, "miss_count": 0, "cached_ratio": 1.0},
        }
    }
    warnings = module._cache_expectation_warnings(
        aggregate=aggregate,
        expect_cached_separation=True,
        expect_cached_whisper=True,
    )
    assert len(warnings) == 1
    assert "Expected cached separation" in warnings[0]


def test_cache_expectation_warnings_no_executed_song_data():
    module = _load_module()
    aggregate = {
        "cache_summary": {
            "separation": {"total": 0, "miss_count": 0, "cached_ratio": 0.0},
            "whisper": {"total": 0, "miss_count": 0, "cached_ratio": 0.0},
        }
    }
    warnings = module._cache_expectation_warnings(
        aggregate=aggregate,
        expect_cached_separation=True,
        expect_cached_whisper=True,
    )
    assert len(warnings) == 2
    assert "no executed-song cache data was available" in warnings[0]


def test_extract_stage_hint_prefers_y2karaoke_line():
    module = _load_module()
    out = "random\n"
    err = "INFO:y2karaoke:📥 Downloading audio...\nnoise\n"
    hint = module._extract_stage_hint(out, err)
    assert hint is not None
    assert hint.startswith("[media_download_audio]")
    assert "Downloading audio" in hint


def test_extract_stage_hint_filters_progress_noise():
    module = _load_module()
    out = "%|█▉| 12/62 [00:19<01:20,  1.61s/it]\n"
    err = "\n"
    hint = module._extract_stage_hint(out, err)
    assert hint is None


def test_extract_stage_hint_classifies_separation_from_keyword():
    module = _load_module()
    out = "audio separator: demucs htdemucs processing stem 1/4\n"
    err = ""
    hint = module._extract_stage_hint(out, err)
    assert hint is not None
    assert hint.startswith("[separation]")


def test_compose_heartbeat_stage_text_promotes_compute_active():
    module = _load_module()
    text = module._compose_heartbeat_stage_text(
        stage_hint="[media_cached_audio] INFO:y2karaoke.core.karaoke:📁 Using cached audio",
        last_stage_hint=None,
        cpu_percent=240.0,
        compute_substage="separation",
    )
    assert text is not None
    assert text.startswith("[separation]")
    assert "cpu=240.0%" in text
    assert "stale_log_stage" not in text


def test_compose_heartbeat_stage_text_without_hint_uses_cpu_activity():
    module = _load_module()
    text = module._compose_heartbeat_stage_text(
        stage_hint=None,
        last_stage_hint=None,
        cpu_percent=130.5,
        compute_substage=None,
    )
    assert text == "[compute_active] cpu=130.5% (likely separation/whisper/alignment)"


def test_extract_video_id_from_command():
    module = _load_module()
    cmd = [
        "python",
        "-m",
        "y2karaoke.cli",
        "generate",
        "https://www.youtube.com/watch?v=abcdefghijk",
    ]
    assert module._extract_video_id_from_command(cmd) == "abcdefghijk"


def test_phase_from_stage_label():
    module = _load_module()
    assert module._phase_from_stage_label("media_cached_audio") == "media_prepare"
    assert module._phase_from_stage_label("whisper") == "whisper"
    assert module._phase_from_stage_label("unknown_label") == "unknown_label"


def test_infer_cache_decisions():
    module = _load_module()
    decisions = module._infer_cache_decisions(
        before={"audio_files": 1, "stem_files": 0, "whisper_files": 0},
        after={"audio_files": 1, "stem_files": 2, "whisper_files": 1},
        combined_output="INFO:y2karaoke.core.karaoke:📁 Using cached audio",
        report_exists=True,
    )
    assert decisions["audio"].startswith("hit")
    assert decisions["separation"].startswith("miss")
    assert decisions["whisper"].startswith("miss")
    assert decisions["alignment"].startswith("computed")
