import importlib.util
import json
import sys
from pathlib import Path

_MODULE_PATH = (
    Path(__file__).resolve().parents[3] / "tools" / "classify_alignment_failures.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "classify_alignment_failures_module", _MODULE_PATH
)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_classify_sparse_lexical_comparability() -> None:
    row = _MODULE._classify_song(
        {
            "artist": "A",
            "title": "Song",
            "metrics": {
                "agreement_coverage_ratio": 0.2,
                "agreement_eligibility_ratio": 0.3,
                "agreement_skip_reason_counts": {
                    "low_text_similarity": 2,
                },
            },
            "alignment_diagnostics": {},
        }
    )
    assert row["label"] == "sparse_lexical_comparability"


def test_classify_repetition_handling() -> None:
    row = _MODULE._classify_song(
        {
            "artist": "A",
            "title": "Song",
            "metrics": {
                "agreement_coverage_ratio": 0.6,
                "agreement_eligibility_ratio": 0.8,
                "agreement_match_ratio_within_eligible": 0.5,
                "agreement_skip_reason_counts": {
                    "anchor_outside_window": 2,
                },
            },
            "alignment_diagnostics": {"issue_tags": ["timing_delta_clamped"]},
        }
    )
    assert row["label"] == "repetition_handling"


def test_classify_sparse_precedes_repetition_when_lexical_signal_is_large() -> None:
    row = _MODULE._classify_song(
        {
            "artist": "A",
            "title": "Song",
            "metrics": {
                "agreement_coverage_ratio": 0.15,
                "agreement_eligibility_ratio": 0.9,
                "agreement_match_ratio_within_eligible": 0.2,
                "agreement_skip_reason_counts": {
                    "anchor_outside_window": 5,
                    "low_text_similarity": 4,
                },
            },
            "alignment_diagnostics": {"issue_tags": ["timing_delta_clamped"]},
        }
    )
    assert row["label"] == "sparse_lexical_comparability"


def test_classify_timing_drift_precedes_sparse_when_signal_is_strong() -> None:
    row = _MODULE._classify_song(
        {
            "artist": "A",
            "title": "Song",
            "metrics": {
                "agreement_coverage_ratio": 0.32,
                "agreement_eligibility_ratio": 0.82,
                "agreement_match_ratio_within_eligible": 0.58,
                "agreement_start_p95_abs_sec": 1.42,
                "dtw_line_coverage": 0.88,
                "agreement_skip_reason_counts": {
                    "low_text_similarity": 2,
                },
            },
            "alignment_diagnostics": {"issue_tags": []},
        }
    )
    assert row["label"] == "timing_drift"


def test_classify_downstream_timing_regression() -> None:
    row = _MODULE._classify_song(
        {
            "artist": "ROSALIA",
            "title": "DESPECHA",
            "metrics": {
                "agreement_coverage_ratio": 0.4107,
                "agreement_eligibility_ratio": 0.7857,
                "agreement_match_ratio_within_eligible": 0.5227,
                "agreement_start_p95_abs_sec": 1.126,
                "dtw_line_coverage": 0.804,
                "gold_start_mean_abs_sec": 0.7712,
                "gold_pre_whisper_start_mean_abs_sec": 0.5193,
                "gold_downstream_regression_line_count": 8,
                "gold_downstream_regression_mean_improvement_sec": 1.0363,
                "agreement_skip_reason_counts": {},
            },
            "alignment_diagnostics": {"issue_tags": []},
        }
    )
    assert row["label"] == "downstream_timing_regression"
    assert "downstream_regression_lines=8" in row["evidence"]


def test_classify_lexical_hook_variant_matching() -> None:
    row = _MODULE._classify_song(
        {
            "artist": "Mark Ronson",
            "title": "Uptown Funk",
            "metrics": {
                "agreement_coverage_ratio": 0.6571,
                "agreement_eligibility_ratio": 0.9,
                "agreement_match_ratio_within_eligible": 0.73,
                "agreement_start_p95_abs_sec": 0.728,
                "dtw_line_coverage": 0.724,
                "low_confidence_ratio": 0.0381,
                "gold_word_coverage_ratio": 1.0,
                "gold_start_mean_abs_sec": 0.3217,
                "agreement_skip_reason_counts": {},
            },
            "alignment_policy_hint": {
                "hint": "review_dtw_lexical_matching",
            },
            "lexical_mismatch_diagnostics": {
                "hook_boundary_variant_ratio": 0.3204,
                "truncation_pattern_ratio": 0.3689,
                "repetitive_phrase_line_ratio": 0.0291,
            },
            "alignment_diagnostics": {},
        }
    )
    assert row["label"] == "lexical_hook_boundary_variants"
    assert "alignment_hint=review_dtw_lexical_matching" in row["evidence"]
    assert "hook_boundary_ratio=0.320" in row["evidence"]


def test_classify_missing_metrics_as_mixed() -> None:
    row = _MODULE._classify_song(
        {
            "artist": "A",
            "title": "Song",
            "status": "failed",
            "metrics": {},
            "alignment_diagnostics": {},
        }
    )
    assert row["label"] == "mixed_or_unclear"
    assert "insufficient_agreement_metrics" in row["evidence"]


def test_main_writes_reports(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "benchmark_report.json").write_text(
        json.dumps({"songs": [{"artist": "A", "title": "Song", "metrics": {}}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "classify_alignment_failures.py",
            "--report",
            str(run_dir),
        ],
    )
    rc = _MODULE.main()
    assert rc == 0
    assert (run_dir / "failure_mode_report.json").exists()
    assert (run_dir / "failure_mode_report.md").exists()
