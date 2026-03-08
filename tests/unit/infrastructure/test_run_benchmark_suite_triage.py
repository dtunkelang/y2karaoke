"""Unit tests for benchmark triage ranking helpers."""

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


def test_build_triage_rankings_classifies_reference_and_pipeline():
    module = _load_module()
    succeeded = [
        {
            "artist": "A",
            "title": "Refy",
            "metrics": {
                "gold_available": False,
                "dtw_line_coverage": 0.45,
                "dtw_word_coverage": 0.35,
                "low_confidence_ratio": 0.03,
                "agreement_coverage_ratio": 0.12,
                "agreement_text_similarity_mean": 0.95,
                "agreement_start_p95_abs_sec": 0.9,
                "agreement_bad_ratio": 0.02,
            },
            "reference_divergence": {"suspected": True, "score": 3.0},
        },
        {
            "artist": "B",
            "title": "Pipey",
            "metrics": {
                "gold_available": True,
                "dtw_line_coverage": 0.2,
                "dtw_word_coverage": 0.1,
                "low_confidence_ratio": 0.25,
                "agreement_coverage_ratio": 0.2,
                "agreement_text_similarity_mean": 0.6,
                "agreement_start_p95_abs_sec": 2.2,
                "agreement_bad_ratio": 0.3,
            },
        },
    ]
    triage = module._build_triage_rankings(succeeded, top_n=5)
    assert triage["likely_reference_divergence"][0]["song"] == "A - Refy"
    assert triage["likely_pipeline_failure"][0]["song"] == "B - Pipey"


def test_build_triage_rankings_downweights_sparse_agreement_penalties():
    module = _load_module()
    succeeded = [
        {
            "artist": "A",
            "title": "SparseAnchors",
            "metrics": {
                "gold_available": False,
                "dtw_line_coverage": 0.98,
                "dtw_word_coverage": 0.84,
                "low_confidence_ratio": 0.06,
                "agreement_coverage_ratio": 0.2,
                "agreement_text_similarity_mean": 0.9,
                "agreement_start_p95_abs_sec": 1.9,
                "agreement_bad_ratio": 0.08,
            },
        },
    ]
    triage = module._build_triage_rankings(succeeded, top_n=5)
    assert triage["likely_pipeline_failure"] == []


def test_build_triage_rankings_uses_timing_quality_score_signal():
    module = _load_module()
    succeeded = [
        {
            "artist": "Q",
            "title": "LowScore",
            "metrics": {
                "gold_available": True,
                "dtw_line_coverage": 0.86,
                "dtw_word_coverage": 0.62,
                "low_confidence_ratio": 0.07,
                "agreement_coverage_ratio": 0.4,
                "agreement_text_similarity_mean": 0.86,
                "agreement_start_p95_abs_sec": 0.85,
                "agreement_bad_ratio": 0.08,
                "timing_quality_score": 0.3,
            },
        }
    ]
    triage = module._build_triage_rankings(succeeded, top_n=5)
    assert len(triage["likely_pipeline_failure"]) == 1
    reasons = triage["likely_pipeline_failure"][0]["reasons"]
    assert "low_timing_quality_score" in reasons


def test_aggregate_includes_triage_ranking_fields():
    module = _load_module()
    results = [
        {
            "artist": "A",
            "title": "T1",
            "status": "ok",
            "metrics": {
                "line_count": 10,
                "low_confidence_lines": 1,
                "low_confidence_ratio": 0.01,
                "dtw_line_coverage": 0.45,
                "dtw_word_coverage": 0.35,
                "agreement_count": 2,
                "agreement_coverage_ratio": 0.1,
                "agreement_text_similarity_mean": 0.95,
                "agreement_start_p95_abs_sec": 0.9,
                "agreement_bad_ratio": 0.02,
            },
            "reference_divergence": {"suspected": True, "score": 2.0},
        }
    ]
    agg = module._aggregate(results)
    assert agg["likely_reference_divergence_count"] >= 1
    assert agg["likely_pipeline_failure_count"] >= 0
    assert (
        "avg_abs_word_start_delta_sec_word_weighted_mean_excluding_reference_divergence"
        in agg
    )
    hotspots = agg["quality_hotspots"]
    assert "likely_reference_divergence" in hotspots
    assert "likely_pipeline_failure" in hotspots


def test_aggregate_reports_reference_excluded_primary_metric():
    module = _load_module()
    results = [
        {
            "artist": "A",
            "title": "Refy",
            "status": "ok",
            "metrics": {
                "line_count": 10,
                "avg_abs_word_start_delta_sec": 12.0,
                "gold_comparable_word_count": 100,
                "gold_word_count": 100,
            },
            "reference_divergence": {"suspected": True, "score": 3.0},
        },
        {
            "artist": "B",
            "title": "Clean",
            "status": "ok",
            "metrics": {
                "line_count": 10,
                "avg_abs_word_start_delta_sec": 2.0,
                "gold_comparable_word_count": 100,
                "gold_word_count": 100,
            },
            "reference_divergence": {"suspected": False, "score": 0.0},
        },
    ]
    agg = module._aggregate(results)
    assert agg["avg_abs_word_start_delta_sec_word_weighted_mean"] == 7.0
    assert (
        agg[
            "avg_abs_word_start_delta_sec_word_weighted_mean_excluding_reference_divergence"
        ]
        == 2.0
    )
    assert agg["curated_canary_song_count"] == 1
    assert agg["curated_canary_song_names"] == ["B - Clean"]
    assert agg["curated_canary_reference_watchlist_count"] == 1
    assert agg["curated_canary_reference_watchlist"] == ["A - Refy"]
    assert agg["curated_canary_avg_abs_word_start_delta_sec_word_weighted_mean"] == 2.0


def test_gold_metric_warnings_prefer_curated_canary_subset():
    module = _load_module()
    warnings = module._gold_metric_warnings(
        {
            "gold_metric_song_count": 3,
            "gold_metric_song_coverage_ratio": 1.0,
            "gold_word_coverage_ratio_total": 1.0,
            "avg_abs_word_start_delta_sec_word_weighted_mean": 7.2,
            "curated_canary_song_count": 2,
            "curated_canary_song_coverage_ratio": 1.0,
            "curated_canary_gold_word_coverage_ratio_total": 0.99,
            "curated_canary_avg_abs_word_start_delta_sec_word_weighted_mean": 1.02,
        }
    )
    assert any(
        "Curated-canary gold-set avg abs word-start delta is high" in w
        for w in warnings
    )
    assert all("Gold-set avg abs word-start delta is high" not in w for w in warnings)


def test_infer_reference_divergence_with_gold_strong_dtw_high_mismatch():
    module = _load_module()
    result = module._infer_reference_divergence_suspicion(
        {
            "gold_available": True,
            "line_count": 50,
            "gold_comparable_word_count": 300,
            "gold_word_coverage_ratio": 0.86,
            "gold_start_mean_abs_sec": 14.6,
            "gold_start_p95_abs_sec": 21.2,
            "dtw_line_coverage": 0.9,
            "dtw_word_coverage": 0.62,
            "agreement_coverage_ratio": 0.02,
            "agreement_text_similarity_mean": 0.84,
            "agreement_bad_ratio": 0.03,
            "low_confidence_ratio": 0.04,
        }
    )
    assert result["suspected"] is True
    assert "high_gold_mismatch_with_strong_dtw" in result["evidence"]


def test_infer_reference_divergence_with_gold_weak_dtw_stays_false():
    module = _load_module()
    result = module._infer_reference_divergence_suspicion(
        {
            "gold_available": True,
            "line_count": 60,
            "gold_comparable_word_count": 220,
            "gold_word_coverage_ratio": 0.8,
            "gold_start_mean_abs_sec": 14.0,
            "gold_start_p95_abs_sec": 22.0,
            "dtw_line_coverage": 0.62,
            "dtw_word_coverage": 0.4,
            "agreement_coverage_ratio": 0.03,
            "agreement_text_similarity_mean": 0.7,
            "agreement_bad_ratio": 0.2,
            "low_confidence_ratio": 0.18,
        }
    )
    assert result["suspected"] is False


def test_markdown_summary_includes_triage_rankings(tmp_path):
    module = _load_module()
    out_path = tmp_path / "summary.md"
    aggregate = module._aggregate(
        [
            {
                "artist": "A",
                "title": "Refy",
                "status": "ok",
                "metrics": {
                    "line_count": 10,
                    "low_confidence_lines": 0,
                    "low_confidence_ratio": 0.0,
                    "dtw_line_coverage": 0.45,
                    "dtw_word_coverage": 0.35,
                    "agreement_count": 2,
                    "agreement_coverage_ratio": 0.1,
                    "agreement_text_similarity_mean": 0.95,
                    "agreement_start_p95_abs_sec": 0.9,
                    "agreement_bad_ratio": 0.02,
                },
                "reference_divergence": {"suspected": True, "score": 2.0},
            },
            {
                "artist": "B",
                "title": "Pipey",
                "status": "ok",
                "metrics": {
                    "line_count": 12,
                    "low_confidence_lines": 4,
                    "low_confidence_ratio": 0.33,
                    "dtw_line_coverage": 0.2,
                    "dtw_word_coverage": 0.1,
                    "agreement_count": 3,
                    "agreement_coverage_ratio": 0.2,
                    "agreement_text_similarity_mean": 0.6,
                    "agreement_start_p95_abs_sec": 2.2,
                    "agreement_bad_ratio": 0.3,
                },
            },
        ]
    )
    module._write_markdown_summary(
        out_path,
        run_id="run1",
        manifest=Path("manifest.yaml"),
        aggregate=aggregate,
        songs=[],
    )
    content = out_path.read_text(encoding="utf-8")
    assert "Triage ranking: likely reference divergence" in content
    assert "Triage ranking: likely pipeline failure" in content


def test_classify_quality_diagnosis_flags_pipeline_ok_for_strong_signals():
    module = _load_module()
    diagnosis = module._classify_quality_diagnosis(
        {
            "line_count": 80,
            "dtw_line_coverage": 0.94,
            "dtw_word_coverage": 0.71,
            "low_confidence_ratio": 0.04,
            "agreement_coverage_ratio": 0.5,
            "agreement_start_p95_abs_sec": 0.62,
            "gold_available": True,
            "gold_comparable_word_count": 300,
            "gold_word_coverage_ratio": 0.9,
            "gold_start_mean_abs_sec": 0.44,
        }
    )
    assert diagnosis["verdict"] == "likely_pipeline_ok"
    assert diagnosis["confidence"] == "high"


def test_classify_quality_diagnosis_defers_to_reference_divergence():
    module = _load_module()
    diagnosis = module._classify_quality_diagnosis(
        {
            "line_count": 90,
            "dtw_line_coverage": 0.91,
            "dtw_word_coverage": 0.66,
            "low_confidence_ratio": 0.05,
        },
        reference_divergence={
            "suspected": True,
            "confidence": "medium",
            "evidence": ["severe_gold_timing_mismatch"],
        },
    )
    assert diagnosis["verdict"] == "likely_reference_divergence"
    assert diagnosis["confidence"] == "medium"
    assert "severe_gold_timing_mismatch" in diagnosis["reasons"]


def test_classify_quality_diagnosis_does_not_overcall_pipeline_when_gold_is_strong():
    module = _load_module()
    diagnosis = module._classify_quality_diagnosis(
        {
            "line_count": 41,
            "dtw_line_coverage": 0.805,
            "dtw_word_coverage": 0.616,
            "low_confidence_ratio": 0.0488,
            "agreement_coverage_ratio": 0.3902,
            "agreement_start_p95_abs_sec": 1.685,
            "gold_word_coverage_ratio": 1.0,
            "gold_start_mean_abs_sec": 0.6254,
            "gold_start_p95_abs_sec": 1.669,
            "gold_comparable_word_count": 271,
            "gold_available": True,
        }
    )
    assert diagnosis["verdict"] == "needs_manual_review"
    assert "high_agreement_p95" not in diagnosis["reasons"]


def test_aggregate_includes_quality_diagnosis_counts():
    module = _load_module()
    results = [
        {
            "artist": "A",
            "title": "Good",
            "status": "ok",
            "metrics": {"line_count": 10, "low_confidence_lines": 0},
            "quality_diagnosis": {"verdict": "likely_pipeline_ok"},
        },
        {
            "artist": "B",
            "title": "Needs",
            "status": "ok",
            "metrics": {"line_count": 10, "low_confidence_lines": 0},
            "quality_diagnosis": {"verdict": "needs_pipeline_work"},
        },
    ]
    agg = module._aggregate(results)
    assert agg["quality_diagnosis_counts"]["likely_pipeline_ok"] == 1
    assert agg["quality_diagnosis_counts"]["needs_pipeline_work"] == 1
    assert agg["quality_diagnosis_ratios"]["likely_pipeline_ok"] == 0.5


def test_agreement_text_normalization_folds_diacritics():
    module = _load_module()
    assert module._normalize_agreement_text("DÉSPÉCHA!") == "despecha"
    assert module._agreement_text_similarity("Déspecha", "Despecha") == 1.0


def test_lexical_token_helpers_fold_diacritics():
    module = _load_module()
    assert module._lexical_tokens_basic("Bésame más") == ["besame", "mas"]
    assert module._lexical_tokens_compact("Bésame más") == ["besame", "mas"]
