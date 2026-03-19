"""Unit tests for benchmark suite runner utilities."""

from __future__ import annotations

import importlib.util
import json
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


def test_validate_cli_args_rejects_invalid_thresholds():
    module = _load_module()
    invalid_cases = [
        ("min_dtw_song_coverage_ratio", 1.1),
        ("min_dtw_line_coverage_ratio", -0.1),
        ("min_timing_quality_score_line_weighted", 2.0),
        ("min_agreement_coverage_gain_for_bad_ratio_warning", -0.01),
        ("max_agreement_bad_ratio_increase_on_coverage_gain", -0.01),
        ("max_whisper_phase_share", 1.1),
        ("max_alignment_phase_share", -0.1),
        ("max_scheduler_overhead_sec", -1.0),
    ]
    for field, value in invalid_cases:
        args = module.argparse.Namespace(
            min_dtw_song_coverage_ratio=0.5,
            min_dtw_line_coverage_ratio=0.5,
            min_timing_quality_score_line_weighted=0.7,
            min_agreement_coverage_gain_for_bad_ratio_warning=0.0,
            max_agreement_bad_ratio_increase_on_coverage_gain=0.01,
            max_whisper_phase_share=0.95,
            max_alignment_phase_share=0.95,
            max_scheduler_overhead_sec=120.0,
        )
        setattr(args, field, value)
        try:
            module._validate_cli_args(args)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {field}={value}")


def test_filter_manifest_songs_match_and_limit():
    module = _load_module()
    songs = [
        module.BenchmarkSong(
            manifest_index=1,
            artist="Artist A",
            title="Alpha",
            youtube_id="aaaaaaaaaaa",
            youtube_url="https://www.youtube.com/watch?v=aaaaaaaaaaa",
        ),
        module.BenchmarkSong(
            manifest_index=1,
            artist="Artist B",
            title="Beta",
            youtube_id="bbbbbbbbbbb",
            youtube_url="https://www.youtube.com/watch?v=bbbbbbbbbbb",
            clip_id="hook-repeat",
        ),
    ]
    selected = module._filter_manifest_songs(songs, match="artist", max_songs=1)
    assert len(selected) == 1
    assert selected[0].title == "Alpha"
    clip_selected = module._filter_manifest_songs(
        songs, match="hook-repeat", max_songs=0
    )
    assert clip_selected == [songs[1]]
    tag_selected = module._filter_manifest_songs(
        [
            songs[0],
            module.BenchmarkSong(
                manifest_index=1,
                artist="Artist C",
                title="Gamma",
                youtube_id="ccccccccccc",
                youtube_url="https://www.youtube.com/watch?v=ccccccccccc",
                clip_id="duet-hook",
                clip_tags=("duet", "overlap"),
            ),
        ],
        match="",
        clip_tags=("duet",),
        max_songs=0,
    )
    assert len(tag_selected) == 1
    assert tag_selected[0].title == "Gamma"


def test_apply_aggregate_only_cached_scope(tmp_path):
    module = _load_module()
    song_a = module.BenchmarkSong(
        manifest_index=1,
        artist="Artist A",
        title="Alpha",
        youtube_id="aaaaaaaaaaa",
        youtube_url="https://www.youtube.com/watch?v=aaaaaaaaaaa",
    )
    song_b = module.BenchmarkSong(
        manifest_index=1,
        artist="Artist B",
        title="Beta",
        youtube_id="bbbbbbbbbbb",
        youtube_url="https://www.youtube.com/watch?v=bbbbbbbbbbb",
    )
    (tmp_path / f"01_{song_b.slug}_result.json").write_text("{}", encoding="utf-8")
    selected = module._apply_aggregate_only_cached_scope(
        [song_a, song_b],
        aggregate_only=True,
        match=None,
        max_songs=0,
        run_dir=tmp_path,
    )
    assert selected == [song_b]

    not_scoped = module._apply_aggregate_only_cached_scope(
        [song_a, song_b],
        aggregate_only=True,
        match="Artist",
        max_songs=0,
        run_dir=tmp_path,
    )
    assert not_scoped == [song_a, song_b]

    clip_song = module.BenchmarkSong(
        manifest_index=1,
        artist="Artist B",
        title="Beta",
        youtube_id="bbbbbbbbbbb",
        youtube_url="https://www.youtube.com/watch?v=bbbbbbbbbbb",
        clip_id="hook",
        audio_start_sec=30.0,
    )
    selected_clip = module._apply_aggregate_only_cached_scope(
        [clip_song],
        aggregate_only=True,
        match=None,
        max_songs=0,
        run_dir=tmp_path,
    )
    assert selected_clip == [clip_song]


def test_load_aggregate_only_results_refreshes_cached_and_marks_missing(tmp_path):
    module = _load_module()
    song_a = module.BenchmarkSong(
        manifest_index=1,
        artist="A",
        title="Song A",
        youtube_id="aaaaaaaaaaa",
        youtube_url="https://www.youtube.com/watch?v=aaaaaaaaaaa",
    )
    song_b = module.BenchmarkSong(
        manifest_index=1,
        artist="B",
        title="Song B",
        youtube_id="bbbbbbbbbbb",
        youtube_url="https://www.youtube.com/watch?v=bbbbbbbbbbb",
    )
    report_path = tmp_path / f"01_{song_a.slug}_timing_report.json"
    report_path.write_text(
        (
            '{"alignment_method":"whisper_hybrid","dtw_line_coverage":0.9,'
            '"dtw_word_coverage":0.8,"dtw_phonetic_similarity_coverage":0.7,'
            '"low_confidence_lines":[],"lines":[{"start":1.0,'
            '"nearest_segment_start":1.1,"text":"hello world",'
            '"nearest_segment_start_text":"hello world"}]}'
        ),
        encoding="utf-8",
    )
    result_path = tmp_path / f"01_{song_a.slug}_result.json"
    result_path.write_text(
        (
            '{"artist":"A","title":"Song A","youtube_id":"aaaaaaaaaaa",'
            f'"report_path":"{report_path}","status":"ok"}}'
        ).replace("}}", "}"),
        encoding="utf-8",
    )
    rows, skipped = module._load_aggregate_only_results(
        songs=[song_a, song_b],
        run_dir=tmp_path,
        gold_root=tmp_path,
        rebaseline=False,
    )
    assert len(rows) == 1
    assert skipped == ["B - Song B"]
    assert rows[0]["status"] == "ok"
    assert rows[0]["result_reused"] is True
    assert rows[0]["aggregate_only_recomputed"] is True
    assert rows[0]["metrics"]["dtw_line_coverage"] == 0.9
    refreshed = json.loads(result_path.read_text(encoding="utf-8"))
    assert refreshed["metrics"]["dtw_line_coverage"] == 0.9
    assert refreshed["aggregate_only_recomputed"] is True


def test_load_aggregate_only_results_falls_back_to_slug_match(tmp_path):
    module = _load_module()
    song = module.BenchmarkSong(
        manifest_index=1,
        artist="B",
        title="Song B",
        youtube_id="bbbbbbbbbbb",
        youtube_url="https://www.youtube.com/watch?v=bbbbbbbbbbb",
    )
    report_path = tmp_path / f"01_{song.slug}_timing_report.json"
    report_path.write_text(
        (
            '{"alignment_method":"whisper_hybrid","dtw_line_coverage":0.8,'
            '"dtw_word_coverage":0.7,"dtw_phonetic_similarity_coverage":0.6,'
            '"low_confidence_lines":[],"lines":[{"start":1.0,'
            '"nearest_segment_start":1.1,"text":"hello world",'
            '"nearest_segment_start_text":"hello world"}]}'
        ),
        encoding="utf-8",
    )
    (tmp_path / f"01_{song.slug}_result.json").write_text(
        (
            '{"artist":"B","title":"Song B","youtube_id":"bbbbbbbbbbb",'
            f'"report_path":"{report_path}","status":"ok"}}'
        ).replace("}}", "}"),
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
    assert rows[0]["metrics"]["dtw_line_coverage"] == 0.8


def test_load_aggregate_only_results_scores_clip_from_full_song_result(tmp_path):
    module = _load_module()
    clip_root = tmp_path / "clip_gold"
    clip_root.mkdir(parents=True)
    module.DEFAULT_CLIP_GOLD_ROOT = clip_root
    song = module.BenchmarkSong(
        manifest_index=1,
        artist="The Weeknd",
        title="Blinding Lights",
        youtube_id="fHI8X4OXluQ",
        youtube_url="https://www.youtube.com/watch?v=fHI8X4OXluQ",
        clip_id="hook-repeat",
        audio_start_sec=112.0,
    )
    (clip_root / f"01_{song.slug}.gold.json").write_text(
        json.dumps(
            {
                "audio_path": "",
                "lines": [
                    {
                        "start": 1.0,
                        "end": 4.0,
                        "text": "I said, ooh",
                        "words": [
                            {"text": "I", "start": 1.0, "end": 1.3},
                            {"text": "said,", "start": 1.3, "end": 1.8},
                            {"text": "ooh", "start": 1.8, "end": 4.0},
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
                        "index": 23,
                        "start": 113.0,
                        "end": 116.8,
                        "text": "I said, ooh",
                        "words": [
                            {"text": "I", "start": 113.0, "end": 113.3},
                            {"text": "said,", "start": 113.3, "end": 113.8},
                            {"text": "ooh", "start": 113.8, "end": 116.8},
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
                "timing_quality_band": "fair",
                "local_transcribe_cache_hits": 2,
                "local_transcribe_cache_misses": 1,
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
                "timing_quality_band": "good",
                "local_transcribe_cache_hits": 0,
                "local_transcribe_cache_misses": 1,
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
    assert agg["local_transcribe_cache_hits_total"] == 2
    assert agg["local_transcribe_cache_misses_total"] == 2
    assert agg["local_transcribe_cache_events_total"] == 4
    assert agg["local_transcribe_cache_hit_ratio"] == 0.5
    assert agg["sum_song_elapsed_sec"] == 20.0
    assert agg["phase_totals_sec"]["separation"] == 16.0
    assert agg["cache_summary"]["separation"]["miss_count"] == 0
    assert agg["cache_summary"]["separation"]["total"] == 0
    assert "highest_timing_quality_score" in agg["quality_hotspots"]
    assert "lowest_timing_quality_score" in agg["quality_hotspots"]
    assert agg["timing_quality_band_counts"] == {"fair": 1, "good": 1}
    assert agg["timing_quality_band_ratios"] == {"fair": 0.5, "good": 0.5}


def test_aggregate_elapsed_distinguishes_executed_from_total():
    module = _load_module()
    results = [
        {
            "artist": "A",
            "title": "Executed",
            "status": "ok",
            "elapsed_sec": 12.0,
            "result_reused": False,
            "metrics": {"line_count": 10},
        },
        {
            "artist": "B",
            "title": "Cached",
            "status": "ok",
            "elapsed_sec": 30.0,
            "result_reused": True,
            "metrics": {"line_count": 10},
        },
    ]
    agg = module._aggregate(results)
    assert agg["sum_song_elapsed_sec"] == 12.0
    assert agg["sum_song_elapsed_total_sec"] == 42.0


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
        min_timing_quality_score_line_weighted=0.58,
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
        min_timing_quality_score_line_weighted=0.58,
        suite_wall_elapsed_sec=10.0,
    )
    assert any(
        "Independent line-start agreement is unavailable" in item for item in warnings
    )


def test_quality_coverage_warnings_skip_dtw_disabled_alert_when_coverage_is_full():
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
    }
    warnings = module._quality_coverage_warnings(
        aggregate=aggregate,
        dtw_enabled=False,
        min_song_coverage_ratio=0.8,
        min_line_coverage_ratio=0.9,
        min_timing_quality_score_line_weighted=0.58,
        suite_wall_elapsed_sec=6.0,
    )
    assert not any(
        "DTW mapping is disabled (--no-whisper-map-lrc-dtw)" in item
        for item in warnings
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
        min_timing_quality_score_line_weighted=0.58,
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
        min_timing_quality_score_line_weighted=0.58,
        suite_wall_elapsed_sec=6.0,
    )
    assert any("timing quality score is below target" in item for item in warnings)


def test_quality_coverage_warnings_include_poor_band_alert():
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
        "timing_quality_score_line_weighted_mean": 0.8,
        "timing_quality_band_ratios": {"poor": 0.4},
    }
    warnings = module._quality_coverage_warnings(
        aggregate=aggregate,
        dtw_enabled=True,
        min_song_coverage_ratio=0.8,
        min_line_coverage_ratio=0.9,
        min_timing_quality_score_line_weighted=0.58,
        suite_wall_elapsed_sec=6.0,
    )
    assert any("poor timing-quality band" in item for item in warnings)


def test_agreement_tradeoff_warnings_triggered_for_coverage_gain_and_bad_regression():
    module = _load_module()
    warnings = module._agreement_tradeoff_warnings(
        aggregate={
            "agreement_coverage_ratio_mean": 0.26,
            "agreement_bad_ratio_mean": 0.05,
        },
        baseline_aggregate={
            "agreement_coverage_ratio_mean": 0.24,
            "agreement_bad_ratio_mean": 0.045,
        },
        min_coverage_gain=0.01,
        max_bad_ratio_increase=0.003,
    )
    assert len(warnings) == 1
    assert "Agreement tradeoff warning" in warnings[0]


def test_agreement_tradeoff_warnings_not_triggered_when_within_tolerance():
    module = _load_module()
    warnings = module._agreement_tradeoff_warnings(
        aggregate={
            "agreement_coverage_ratio_mean": 0.26,
            "agreement_bad_ratio_mean": 0.047,
        },
        baseline_aggregate={
            "agreement_coverage_ratio_mean": 0.24,
            "agreement_bad_ratio_mean": 0.045,
        },
        min_coverage_gain=0.01,
        max_bad_ratio_increase=0.003,
    )
    assert warnings == []


def test_aggregate_includes_agreement_comparability_report():
    module = _load_module()
    results = [
        {
            "artist": "A",
            "title": "Song 1",
            "status": "ok",
            "metrics": {
                "line_count": 10,
                "agreement_count": 4,
                "agreement_eligible_lines": 6,
                "agreement_matched_lines": 5,
                "agreement_skip_reason_counts": {
                    "low_text_similarity": 1,
                    "low_token_overlap": 2,
                },
            },
        },
        {
            "artist": "B",
            "title": "Song 2",
            "status": "ok",
            "metrics": {
                "line_count": 8,
                "agreement_count": 2,
                "agreement_eligible_lines": 5,
                "agreement_matched_lines": 2,
                "agreement_skip_reason_counts": {
                    "anchor_outside_window": 3,
                },
            },
        },
    ]
    agg = module._aggregate(results)
    assert agg["agreement_eligible_lines_total"] == 11
    assert agg["agreement_matched_anchor_lines_total"] == 7
    assert agg["agreement_skip_reason_totals"]["anchor_outside_window"] == 3
    assert agg["agreement_skip_reason_totals"]["low_token_overlap"] == 2
    report = agg["agreement_comparability_report"]
    assert len(report) == 2
    assert report[0]["song"] in {"A - Song 1", "B - Song 2"}


def test_extract_alignment_diagnostics_maps_coverage_promotion_reason():
    module = _load_module()
    report = {
        "dtw_metrics": {
            "fallback_map_decision_code": 4.0,
        },
    }
    diag = module._extract_alignment_diagnostics(report)
    assert diag["fallback_map_decision_reason"] == "selected_coverage_promotion"


def test_aggregate_includes_fallback_map_selection_summary():
    module = _load_module()
    results = [
        {
            "artist": "A",
            "title": "Song 1",
            "status": "ok",
            "metrics": {"line_count": 1},
            "alignment_diagnostics": {
                "alignment_method": "whisper_hybrid",
                "lyrics_source_provider": "lyriq",
                "issue_tag_counts": {},
                "fallback_map_attempted": True,
                "fallback_map_selected": True,
                "fallback_map_rejected": False,
                "fallback_map_decision_reason": "selected_score_gain",
                "fallback_map_score_gain": 0.2,
            },
        },
        {
            "artist": "B",
            "title": "Song 2",
            "status": "ok",
            "metrics": {"line_count": 1},
            "alignment_diagnostics": {
                "alignment_method": "whisper_hybrid",
                "lyrics_source_provider": "lyriq",
                "issue_tag_counts": {},
                "fallback_map_attempted": True,
                "fallback_map_selected": False,
                "fallback_map_rejected": True,
                "fallback_map_decision_reason": "rejected_insufficient_score_gain",
                "fallback_map_score_gain": 0.01,
            },
        },
    ]
    agg = module._aggregate(results)
    summary = agg["alignment_diagnostics_summary"]
    assert summary["fallback_map_attempted_count"] == 2
    assert summary["fallback_map_selected_count"] == 1
    assert summary["fallback_map_rejected_count"] == 1
    assert summary["fallback_map_decision_counts"]["selected_score_gain"] == 1
    assert (
        summary["fallback_map_decision_counts"]["rejected_insufficient_score_gain"] == 1
    )
    assert len(summary["fallback_map_song_report"]) == 2


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
