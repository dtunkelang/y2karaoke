from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

_MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "tools"
    / "report_suspect_curated_gold_timings.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "report_suspect_curated_gold_timings", _MODULE_PATH
)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def _write_report(tmp_path: Path, songs: list[dict[str, object]]) -> Path:
    path = tmp_path / "benchmark_report.json"
    path.write_text(json.dumps({"songs": songs}), encoding="utf-8")
    return path


def test_collect_suspect_curated_gold_timing_entries_flags_strong_internal_bad_gold(
    tmp_path: Path,
) -> None:
    report = _write_report(
        tmp_path,
        [
            {
                "artist": "Artist",
                "title": "Song",
                "status": "ok",
                "gold_path": "/tmp/01_artist-song-hook-repeat.gold.json",
                "report_path": "/tmp/report.json",
                "quality_diagnosis": {"verdict": "needs_manual_review"},
                "metrics": {
                    "gold_start_mean_abs_sec": 3.2,
                    "gold_end_mean_abs_sec": 3.0,
                    "gold_start_p95_abs_sec": 5.7,
                    "dtw_line_coverage": 1.0,
                    "dtw_word_coverage": 1.0,
                    "timing_quality_score": 0.91,
                    "gold_nearest_onset_start_mean_abs_sec": 0.08,
                    "gold_word_coverage_ratio": 1.0,
                },
            }
        ],
    )

    entries = _MODULE.collect_suspect_curated_gold_timing_entries(
        benchmark_report_path=report,
        min_gold_start_mean_abs_sec=0.75,
        min_dtw_line_coverage=0.9,
        min_dtw_word_coverage=0.9,
        min_timing_quality_score=0.75,
        max_nearest_onset_start_mean_abs_sec=0.12,
        min_gold_word_coverage_ratio=0.98,
    )

    assert len(entries) == 1
    assert entries[0]["clip_id"] == "hook-repeat"
    assert entries[0]["quality_diagnosis_verdict"] == "needs_manual_review"
    assert "perfect_dtw_coverage" in entries[0]["reasons"]


def test_collect_suspect_curated_gold_timing_entries_skips_weak_dtw_pipeline_case(
    tmp_path: Path,
) -> None:
    report = _write_report(
        tmp_path,
        [
            {
                "artist": "Artist",
                "title": "Song",
                "status": "ok",
                "gold_path": "/tmp/01_artist-song-first-chorus.gold.json",
                "metrics": {
                    "gold_start_mean_abs_sec": 1.2,
                    "gold_end_mean_abs_sec": 1.2,
                    "gold_start_p95_abs_sec": 2.2,
                    "dtw_line_coverage": 0.6,
                    "dtw_word_coverage": 0.48,
                    "timing_quality_score": 0.57,
                    "gold_nearest_onset_start_mean_abs_sec": 0.06,
                    "gold_word_coverage_ratio": 1.0,
                },
            }
        ],
    )

    entries = _MODULE.collect_suspect_curated_gold_timing_entries(
        benchmark_report_path=report,
        min_gold_start_mean_abs_sec=0.75,
        min_dtw_line_coverage=0.9,
        min_dtw_word_coverage=0.9,
        min_timing_quality_score=0.75,
        max_nearest_onset_start_mean_abs_sec=0.12,
        min_gold_word_coverage_ratio=0.98,
    )

    assert entries == []


def test_collect_suspect_curated_gold_timing_entries_sorts_by_gold_start_desc(
    tmp_path: Path,
) -> None:
    report = _write_report(
        tmp_path,
        [
            {
                "artist": "Artist",
                "title": "Song A",
                "status": "ok",
                "gold_path": "/tmp/01_artist-song-a-bridge.gold.json",
                "metrics": {
                    "gold_start_mean_abs_sec": 1.5,
                    "gold_end_mean_abs_sec": 1.4,
                    "gold_start_p95_abs_sec": 2.3,
                    "dtw_line_coverage": 1.0,
                    "dtw_word_coverage": 1.0,
                    "timing_quality_score": 0.85,
                    "gold_nearest_onset_start_mean_abs_sec": 0.05,
                    "gold_word_coverage_ratio": 1.0,
                },
            },
            {
                "artist": "Artist",
                "title": "Song B",
                "status": "ok",
                "gold_path": "/tmp/02_artist-song-b-hook.gold.json",
                "metrics": {
                    "gold_start_mean_abs_sec": 3.5,
                    "gold_end_mean_abs_sec": 3.1,
                    "gold_start_p95_abs_sec": 5.0,
                    "dtw_line_coverage": 1.0,
                    "dtw_word_coverage": 0.95,
                    "timing_quality_score": 0.86,
                    "gold_nearest_onset_start_mean_abs_sec": 0.06,
                    "gold_word_coverage_ratio": 1.0,
                },
            },
        ],
    )

    entries = _MODULE.collect_suspect_curated_gold_timing_entries(
        benchmark_report_path=report,
        min_gold_start_mean_abs_sec=0.75,
        min_dtw_line_coverage=0.9,
        min_dtw_word_coverage=0.9,
        min_timing_quality_score=0.75,
        max_nearest_onset_start_mean_abs_sec=0.12,
        min_gold_word_coverage_ratio=0.98,
    )

    assert [entry["title"] for entry in entries] == ["Song B", "Song A"]


def test_print_report_includes_reasons(capsys) -> None:
    _MODULE._print_report(
        [
            {
                "artist": "Artist",
                "title": "Song",
                "clip_id": "repeat",
                "gold_start_mean_abs_sec": 3.2,
                "gold_end_mean_abs_sec": 3.0,
                "gold_start_p95_abs_sec": 5.7,
                "dtw_line_coverage": 1.0,
                "dtw_word_coverage": 1.0,
                "timing_quality_score": 0.91,
                "gold_nearest_onset_start_mean_abs_sec": 0.08,
                "quality_diagnosis_verdict": "needs_manual_review",
                "reasons": ["perfect_dtw_coverage", "gold_far_from_audio_onsets"],
            }
        ]
    )

    out = capsys.readouterr().out
    assert "suspect_curated_gold_timings: FAIL (1 candidates)" in out
    assert "reasons=perfect_dtw_coverage,gold_far_from_audio_onsets" in out
