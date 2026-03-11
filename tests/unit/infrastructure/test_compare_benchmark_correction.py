import importlib.util
import json
import sys
from pathlib import Path

_MODULE_PATH = (
    Path(__file__).resolve().parents[3] / "tools" / "compare_benchmark_correction.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "compare_benchmark_correction_module", _MODULE_PATH
)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_metric_delta_respects_direction() -> None:
    higher = _MODULE._metric_delta(0.5, 0.6, "higher")
    lower = _MODULE._metric_delta(1.0, 0.8, "lower")
    regressed = _MODULE._metric_delta(0.7, 0.6, "higher")
    unchanged = _MODULE._metric_delta(0.7, 0.7, "higher")
    assert higher["improved"] is True
    assert lower["improved"] is True
    assert regressed["improved"] is False
    assert unchanged["improved"] is None


def test_build_comparison_reports_net_improvement() -> None:
    baseline = {
        "aggregate": {
            "timing_quality_score_line_weighted_mean": 0.70,
            "agreement_coverage_ratio_mean": 0.25,
            "agreement_start_p95_abs_sec_mean": 1.00,
            "agreement_bad_ratio_mean": 0.20,
            "dtw_word_coverage_line_weighted_mean": 0.80,
            "avg_abs_word_start_delta_sec_word_weighted_mean": 1.2,
            "curated_canary_song_count": 2,
            "curated_canary_gold_word_coverage_ratio_total": 0.98,
            "curated_canary_avg_abs_word_start_delta_sec_word_weighted_mean": 1.4,
            "curated_canary_gold_start_p95_abs_sec_mean": 3.4,
            "curated_canary_gold_line_duration_mean_abs_sec_mean": 1.9,
            "curated_canary_gold_pre_whisper_start_mean_abs_sec_mean": 0.9,
            "curated_canary_gold_nearest_onset_start_mean_abs_sec_mean": 0.12,
            "curated_canary_gold_later_onset_choice_line_count_total": 8,
            "curated_canary_gold_later_onset_choice_mean_improvement_sec_mean": 0.75,
            "curated_canary_gold_downstream_regression_line_count_total": 5,
            "curated_canary_gold_downstream_regression_mean_improvement_sec_mean": 0.62,
            "curated_canary_gold_parenthetical_interjection_start_mean_abs_sec_mean": 2.1,
            "curated_canary_gold_parenthetical_interjection_start_p95_abs_sec_mean": 2.6,
            "lexical_review_song_count": 2,
            "lexical_hook_boundary_variant_song_count": 2,
            "lexical_hook_boundary_variant_ratio_mean": 0.21,
            "lexical_truncation_pattern_ratio_mean": 0.41,
            "lexical_repetitive_phrase_line_ratio_mean": 0.08,
            "curated_canary_reference_watchlist_count": 1,
            "curated_canary_reference_watchlist": ["Artist B - Song B"],
        },
        "songs": [
            {
                "artist": "Artist A",
                "title": "Song A",
                "metrics": {
                    "timing_quality_score": 0.70,
                    "agreement_coverage_ratio": 0.20,
                    "agreement_start_p95_abs_sec": 1.10,
                    "agreement_bad_ratio": 0.30,
                    "gold_start_mean_abs_sec": 1.4,
                    "dtw_word_coverage": 0.70,
                },
            }
        ],
    }
    corrected = {
        "aggregate": {
            "timing_quality_score_line_weighted_mean": 0.75,
            "agreement_coverage_ratio_mean": 0.30,
            "agreement_start_p95_abs_sec_mean": 0.90,
            "agreement_bad_ratio_mean": 0.18,
            "dtw_word_coverage_line_weighted_mean": 0.84,
            "avg_abs_word_start_delta_sec_word_weighted_mean": 1.0,
            "curated_canary_song_count": 2,
            "curated_canary_gold_word_coverage_ratio_total": 0.99,
            "curated_canary_avg_abs_word_start_delta_sec_word_weighted_mean": 1.1,
            "curated_canary_gold_start_p95_abs_sec_mean": 2.9,
            "curated_canary_gold_line_duration_mean_abs_sec_mean": 0.7,
            "curated_canary_gold_pre_whisper_start_mean_abs_sec_mean": 0.5,
            "curated_canary_gold_nearest_onset_start_mean_abs_sec_mean": 0.05,
            "curated_canary_gold_later_onset_choice_line_count_total": 2,
            "curated_canary_gold_later_onset_choice_mean_improvement_sec_mean": 0.31,
            "curated_canary_gold_downstream_regression_line_count_total": 1,
            "curated_canary_gold_downstream_regression_mean_improvement_sec_mean": 0.21,
            "curated_canary_gold_parenthetical_interjection_start_mean_abs_sec_mean": 0.8,
            "curated_canary_gold_parenthetical_interjection_start_p95_abs_sec_mean": 1.0,
            "lexical_review_song_count": 1,
            "lexical_hook_boundary_variant_song_count": 1,
            "lexical_hook_boundary_variant_ratio_mean": 0.09,
            "lexical_truncation_pattern_ratio_mean": 0.22,
            "lexical_repetitive_phrase_line_ratio_mean": 0.03,
            "curated_canary_reference_watchlist_count": 0,
            "curated_canary_reference_watchlist": [],
        },
        "songs": [
            {
                "artist": "Artist A",
                "title": "Song A",
                "metrics": {
                    "timing_quality_score": 0.80,
                    "agreement_coverage_ratio": 0.25,
                    "agreement_start_p95_abs_sec": 0.95,
                    "agreement_bad_ratio": 0.20,
                    "gold_start_mean_abs_sec": 1.1,
                    "dtw_word_coverage": 0.75,
                },
            }
        ],
    }

    report = _MODULE._build_comparison(
        baseline_report=baseline,
        corrected_report=corrected,
        baseline_label="baseline_run",
        corrected_label="corrected_run",
    )

    assert report["summary"]["songs_compared"] == 1
    assert report["summary"]["songs_net_improved"] == 1
    assert (
        report["aggregate_deltas"]["timing_quality_score_line_weighted_mean"][
            "improved"
        ]
        is True
    )
    assert (
        report["curated_canary_deltas"][
            "curated_canary_avg_abs_word_start_delta_sec_word_weighted_mean"
        ]["improved"]
        is True
    )
    assert (
        report["curated_canary_deltas"][
            "curated_canary_gold_parenthetical_interjection_start_mean_abs_sec_mean"
        ]["improved"]
        is True
    )
    assert (
        report["curated_canary_deltas"][
            "curated_canary_gold_line_duration_mean_abs_sec_mean"
        ]["improved"]
        is True
    )
    assert (
        report["curated_canary_deltas"][
            "curated_canary_gold_nearest_onset_start_mean_abs_sec_mean"
        ]["improved"]
        is True
    )
    assert (
        report["curated_canary_deltas"][
            "curated_canary_gold_later_onset_choice_line_count_total"
        ]["improved"]
        is True
    )
    assert (
        report["curated_canary_deltas"][
            "curated_canary_gold_later_onset_choice_mean_improvement_sec_mean"
        ]["improved"]
        is True
    )
    assert (
        report["curated_canary_deltas"][
            "curated_canary_gold_pre_whisper_start_mean_abs_sec_mean"
        ]["improved"]
        is True
    )
    assert (
        report["curated_canary_deltas"][
            "curated_canary_gold_downstream_regression_line_count_total"
        ]["improved"]
        is True
    )
    assert (
        report["curated_canary_deltas"][
            "curated_canary_gold_downstream_regression_mean_improvement_sec_mean"
        ]["improved"]
        is True
    )
    assert (
        report["curated_canary_deltas"]["lexical_review_song_count"]["improved"] is True
    )
    assert (
        report["curated_canary_deltas"]["lexical_hook_boundary_variant_song_count"][
            "improved"
        ]
        is True
    )
    assert (
        report["curated_canary_deltas"]["lexical_hook_boundary_variant_ratio_mean"][
            "improved"
        ]
        is True
    )
    assert (
        report["curated_canary_deltas"]["lexical_truncation_pattern_ratio_mean"][
            "improved"
        ]
        is True
    )
    assert (
        report["curated_canary_deltas"]["lexical_repetitive_phrase_line_ratio_mean"][
            "improved"
        ]
        is True
    )
    assert report["curated_canary_watchlist"]["baseline"] == ["Artist B - Song B"]
    assert report["curated_canary_watchlist"]["corrected"] == []
    song_row = report["song_deltas"][0]
    assert song_row["net_score"] > 0


def test_main_writes_json_markdown_and_curated_canary_cli_summary(
    tmp_path, monkeypatch, capsys
) -> None:
    baseline_dir = tmp_path / "baseline"
    corrected_dir = tmp_path / "corrected"
    baseline_dir.mkdir()
    corrected_dir.mkdir()

    minimal_report = {
        "aggregate": {
            "curated_canary_song_count": 2,
            "curated_canary_gold_word_coverage_ratio_total": 0.99,
            "curated_canary_avg_abs_word_start_delta_sec_word_weighted_mean": 1.4,
            "curated_canary_gold_start_p95_abs_sec_mean": 3.2,
            "curated_canary_gold_line_duration_mean_abs_sec_mean": 1.7,
            "curated_canary_gold_pre_whisper_start_mean_abs_sec_mean": 0.8,
            "curated_canary_gold_nearest_onset_start_mean_abs_sec_mean": 0.2,
            "curated_canary_gold_later_onset_choice_line_count_total": 6,
            "curated_canary_gold_later_onset_choice_mean_improvement_sec_mean": 0.4,
            "curated_canary_gold_downstream_regression_line_count_total": 3,
            "curated_canary_gold_downstream_regression_mean_improvement_sec_mean": 0.5,
            "curated_canary_gold_parenthetical_interjection_start_mean_abs_sec_mean": 2.0,
            "curated_canary_gold_parenthetical_interjection_start_p95_abs_sec_mean": 2.4,
            "lexical_review_song_count": 1,
            "lexical_hook_boundary_variant_song_count": 1,
            "lexical_hook_boundary_variant_ratio_mean": 0.21,
            "lexical_truncation_pattern_ratio_mean": 0.37,
            "lexical_repetitive_phrase_line_ratio_mean": 0.03,
            "curated_canary_reference_watchlist_count": 1,
            "curated_canary_reference_watchlist": ["Artist Y - Song Y"],
        },
        "songs": [
            {
                "artist": "Artist X",
                "title": "Song X",
                "metrics": {},
            }
        ],
    }
    (baseline_dir / "benchmark_report.json").write_text(
        json.dumps(minimal_report),
        encoding="utf-8",
    )
    (corrected_dir / "benchmark_report.json").write_text(
        json.dumps(minimal_report),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_benchmark_correction.py",
            "--baseline",
            str(baseline_dir),
            "--corrected",
            str(corrected_dir),
        ],
    )

    rc = _MODULE.main()
    captured = capsys.readouterr()
    assert rc == 0
    assert (corrected_dir / "human_correction_delta.json").exists()
    assert (corrected_dir / "human_correction_delta.md").exists()
    markdown = (corrected_dir / "human_correction_delta.md").read_text(encoding="utf-8")
    assert "## Curated Canary Deltas" in markdown
    assert "curated_canary:" in captured.out
    assert "gold_line_duration_abs_sec=1.7000 -> 1.7000" in captured.out
    assert "pre_whisper_gold_start_abs_sec=0.8000 -> 0.8000" in captured.out
    assert "downstream_regression_line_count=3.0000 -> 3.0000" in captured.out
    assert "gold_nearest_onset_start_abs_sec=0.2000 -> 0.2000" in captured.out
    assert "later_onset_choice_line_count=6.0000 -> 6.0000" in captured.out
    assert "later_onset_choice_mean_improvement_sec=0.4000 -> 0.4000" in captured.out
    assert "parenthetical_interjection_start_abs_sec=2.0000 -> 2.0000" in captured.out
    assert "lexical_review_song_count=1.0000 -> 1.0000" in captured.out
    assert "lexical_hook_boundary_variant_song_count=1.0000 -> 1.0000" in captured.out
    assert "lexical_hook_boundary_variant_ratio=0.2100 -> 0.2100" in captured.out
    assert "lexical_truncation_pattern_ratio=0.3700 -> 0.3700" in captured.out
    assert "lexical_repetitive_phrase_ratio=0.0300 -> 0.0300" in captured.out
    assert "watchlist_corrected=Artist Y - Song Y" in captured.out


def test_main_assert_tradeoff_fails_on_bad_ratio_regression(
    tmp_path, monkeypatch
) -> None:
    baseline_dir = tmp_path / "baseline"
    corrected_dir = tmp_path / "corrected"
    baseline_dir.mkdir()
    corrected_dir.mkdir()

    baseline = {
        "aggregate": {
            "agreement_coverage_ratio_mean": 0.20,
            "agreement_bad_ratio_mean": 0.02,
        },
        "songs": [],
    }
    corrected = {
        "aggregate": {
            "agreement_coverage_ratio_mean": 0.23,
            "agreement_bad_ratio_mean": 0.03,
        },
        "songs": [],
    }
    (baseline_dir / "benchmark_report.json").write_text(
        json.dumps(baseline),
        encoding="utf-8",
    )
    (corrected_dir / "benchmark_report.json").write_text(
        json.dumps(corrected),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_benchmark_correction.py",
            "--baseline",
            str(baseline_dir),
            "--corrected",
            str(corrected_dir),
            "--assert-agreement-tradeoff",
            "--min-coverage-gain",
            "0.01",
            "--max-bad-ratio-increase",
            "0.005",
        ],
    )
    rc = _MODULE.main()
    assert rc == 1


def test_main_assert_tradeoff_passes_when_bad_ratio_within_tolerance(
    tmp_path, monkeypatch
) -> None:
    baseline_dir = tmp_path / "baseline"
    corrected_dir = tmp_path / "corrected"
    baseline_dir.mkdir()
    corrected_dir.mkdir()

    baseline = {
        "aggregate": {
            "agreement_coverage_ratio_mean": 0.20,
            "agreement_bad_ratio_mean": 0.02,
        },
        "songs": [],
    }
    corrected = {
        "aggregate": {
            "agreement_coverage_ratio_mean": 0.225,
            "agreement_bad_ratio_mean": 0.024,
        },
        "songs": [],
    }
    (baseline_dir / "benchmark_report.json").write_text(
        json.dumps(baseline),
        encoding="utf-8",
    )
    (corrected_dir / "benchmark_report.json").write_text(
        json.dumps(corrected),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_benchmark_correction.py",
            "--baseline",
            str(baseline_dir),
            "--corrected",
            str(corrected_dir),
            "--assert-agreement-tradeoff",
            "--min-coverage-gain",
            "0.01",
            "--max-bad-ratio-increase",
            "0.005",
        ],
    )
    rc = _MODULE.main()
    assert rc == 0
