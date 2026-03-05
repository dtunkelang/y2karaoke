"""Unit tests for benchmark suite runner utilities."""

from __future__ import annotations

import importlib.util
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
        strategy="hybrid_dtw",
        drop_lrc_line_timings=True,
    )
    assert cmd[:4] == ["python", "-m", "y2karaoke.cli", "generate"]
    assert "--offline" in cmd
    assert "--force" in cmd
    assert "--whisper" not in cmd
    assert "--whisper-map-lrc-dtw" in cmd
    assert "--work-dir" in cmd
    assert str(cache_dir) in cmd
    assert "--drop-lrc-line-timings" in cmd


def test_build_generate_command_strategy_variants(tmp_path):
    module = _load_module()
    song = module.BenchmarkSong(
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


def test_resolve_run_dir_resume_latest(tmp_path):
    module = _load_module()
    older = tmp_path / "20260101T000000Z"
    newer = tmp_path / "20260102T000000Z"
    older.mkdir(parents=True)
    newer.mkdir(parents=True)

    # Force identical mtimes to ensure name-based tie-breaking is tested
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
        strategy="hybrid_whisper",
        scenario="lyrics_no_timing",
        no_whisper_map_lrc_dtw=True,
        cache_dir=tmp_path,
    )
    sig = module._build_run_signature(args, tmp_path / "manifest.yaml")
    assert sig["manifest_path"].endswith("manifest.yaml")
    assert sig["offline"] is True
    assert sig["strategy"] == "hybrid_whisper"
    assert sig["scenario"] == "lyrics_no_timing"
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
                "agreement_count": 60,
                "agreement_good_lines": 30,
                "agreement_warn_lines": 20,
                "agreement_bad_lines": 10,
                "agreement_severe_lines": 4,
                "agreement_coverage_ratio": 0.6,
                "agreement_text_similarity_mean": 0.8,
                "agreement_start_mean_abs_sec": 0.55,
                "agreement_start_max_abs_sec": 1.8,
                "agreement_start_p95_abs_sec": 1.4,
                "agreement_bad_ratio": 0.1,
                "agreement_severe_ratio": 0.04,
                "timing_quality_score": 0.62,
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
                "agreement_count": 10,
                "agreement_good_lines": 8,
                "agreement_warn_lines": 1,
                "agreement_bad_lines": 1,
                "agreement_severe_lines": 0,
                "agreement_coverage_ratio": 0.2,
                "agreement_text_similarity_mean": 0.75,
                "agreement_start_mean_abs_sec": 0.4,
                "agreement_start_max_abs_sec": 1.0,
                "agreement_start_p95_abs_sec": 0.9,
                "agreement_bad_ratio": 0.02,
                "agreement_severe_ratio": 0.0,
                "timing_quality_score": 0.78,
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
    assert agg["agreement_start_max_abs_sec_mean"] == 1.4
    assert agg["timing_quality_score_mean"] == 0.7
    assert agg["timing_quality_score_line_weighted_mean"] == 0.6733
    assert agg["sum_song_elapsed_sec"] == 20.0
    assert agg["phase_totals_sec"]["separation"] == 16.0
    assert agg["cache_summary"]["separation"]["miss_count"] == 0
    assert agg["cache_summary"]["separation"]["total"] == 0
    assert "lowest_timing_quality_score" in agg["quality_hotspots"]


def test_quality_coverage_warnings():
    module = _load_module()
    aggregate = {
        "dtw_metric_song_coverage_ratio": 0.5,
        "dtw_metric_line_coverage_ratio": 0.4,
        "agreement_count_total": 20,
        "agreement_coverage_ratio_mean": 0.2,
        "agreement_start_p95_abs_sec_mean": 1.2,
        "agreement_bad_ratio_total": 0.2,
        "agreement_severe_ratio_total": 0.05,
        "sum_song_elapsed_sec": 12.0,
    }
    warnings = module._quality_coverage_warnings(
        aggregate=aggregate,
        dtw_enabled=False,
        min_song_coverage_ratio=0.8,
        min_line_coverage_ratio=0.9,
        suite_wall_elapsed_sec=10.0,
    )
    assert len(warnings) >= 8
    assert any("LRC-Whisper agreement coverage is low" in item for item in warnings)
    assert any("poor start agreement" in item for item in warnings)


def test_quality_coverage_warnings_independent_unavailable():
    module = _load_module()
    aggregate = {
        "dtw_metric_song_coverage_ratio": 0.5,
        "dtw_metric_line_coverage_ratio": 0.4,
        "agreement_count_total": 0,
        "sum_song_elapsed_sec": 5.0,
    }
    warnings = module._quality_coverage_warnings(
        aggregate=aggregate,
        dtw_enabled=True,
        min_song_coverage_ratio=0.8,
        min_line_coverage_ratio=0.9,
        suite_wall_elapsed_sec=10.0,
    )
    assert any(
        "Independent line-start agreement is unavailable" in item for item in warnings
    )


def test_quality_coverage_warnings_include_diagnosis_ratio_alerts():
    module = _load_module()
    aggregate = {
        "dtw_metric_song_coverage_ratio": 1.0,
        "dtw_metric_line_coverage_ratio": 1.0,
        "agreement_count_total": 1,
        "agreement_coverage_ratio_mean": 1.0,
        "agreement_start_p95_abs_sec_mean": 0.2,
        "agreement_bad_ratio_total": 0.0,
        "agreement_severe_ratio_total": 0.0,
        "sum_song_elapsed_sec": 5.0,
        "quality_diagnosis_ratios": {
            "needs_pipeline_work": 0.5,
            "likely_reference_divergence": 0.4,
        },
    }
    warnings = module._quality_coverage_warnings(
        aggregate=aggregate,
        dtw_enabled=True,
        min_song_coverage_ratio=0.8,
        min_line_coverage_ratio=0.9,
        suite_wall_elapsed_sec=6.0,
    )
    assert any("diagnosed as pipeline work needed" in item for item in warnings)
    assert any("diagnosed as likely reference divergence" in item for item in warnings)


def test_quality_coverage_warnings_include_timing_quality_score_alert():
    module = _load_module()
    aggregate = {
        "dtw_metric_song_coverage_ratio": 1.0,
        "dtw_metric_line_coverage_ratio": 1.0,
        "agreement_count_total": 1,
        "agreement_coverage_ratio_mean": 1.0,
        "agreement_start_p95_abs_sec_mean": 0.2,
        "agreement_bad_ratio_total": 0.0,
        "agreement_severe_ratio_total": 0.0,
        "sum_song_elapsed_sec": 5.0,
        "timing_quality_score_line_weighted_mean": 0.5,
    }
    warnings = module._quality_coverage_warnings(
        aggregate=aggregate,
        dtw_enabled=True,
        min_song_coverage_ratio=0.8,
        min_line_coverage_ratio=0.9,
        suite_wall_elapsed_sec=6.0,
    )
    assert any("timing quality score is below target" in item for item in warnings)


def test_cache_expectation_warnings():
    module = _load_module()
    aggregate = {
        "cache_summary": {
            "separation": {
                "total": 2,
                "miss_count": 1,
                "unknown_count": 0,
                "cached_ratio": 0.5,
            },
            "whisper": {
                "total": 2,
                "miss_count": 0,
                "unknown_count": 0,
                "cached_ratio": 1.0,
            },
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
            "separation": {
                "total": 0,
                "miss_count": 0,
                "unknown_count": 0,
                "cached_ratio": 0.0,
            },
            "whisper": {
                "total": 0,
                "miss_count": 0,
                "unknown_count": 0,
                "cached_ratio": 0.0,
            },
        }
    }
    warnings = module._cache_expectation_warnings(
        aggregate=aggregate,
        expect_cached_separation=True,
        expect_cached_whisper=True,
    )
    assert len(warnings) == 2
    assert "no executed-song cache data was available" in warnings[0]


def test_cache_expectation_warnings_unknown_state():
    module = _load_module()
    aggregate = {
        "cache_summary": {
            "separation": {
                "total": 3,
                "miss_count": 0,
                "unknown_count": 3,
                "cached_ratio": 0.0,
            },
            "whisper": {
                "total": 3,
                "miss_count": 0,
                "unknown_count": 2,
                "cached_ratio": 0.3333,
            },
        }
    }
    warnings = module._cache_expectation_warnings(
        aggregate=aggregate,
        expect_cached_separation=True,
        expect_cached_whisper=True,
    )
    assert len(warnings) == 2
    assert "cache state was unknown" in warnings[0]


def test_runtime_budget_warnings():
    module = _load_module()
    aggregate = {
        "phase_shares_of_song_elapsed": {
            "whisper": 0.62,
            "alignment": 0.27,
        },
        "sum_song_elapsed_sec": 10.0,
    }
    warnings = module._runtime_budget_warnings(
        aggregate=aggregate,
        suite_wall_elapsed_sec=14.5,
        max_whisper_phase_share=0.5,
        max_alignment_phase_share=0.2,
        max_scheduler_overhead_sec=3.0,
    )
    assert len(warnings) == 3
    assert any("Whisper phase share exceeds budget" in item for item in warnings)
    assert any("Alignment phase share exceeds budget" in item for item in warnings)
    assert any("Scheduler overhead exceeds budget" in item for item in warnings)


def test_runtime_budget_warnings_disabled_thresholds():
    module = _load_module()
    warnings = module._runtime_budget_warnings(
        aggregate={"phase_shares_of_song_elapsed": {}, "sum_song_elapsed_sec": 12.0},
        suite_wall_elapsed_sec=12.2,
        max_whisper_phase_share=0.0,
        max_alignment_phase_share=0.0,
        max_scheduler_overhead_sec=0.0,
    )
    assert warnings == []
