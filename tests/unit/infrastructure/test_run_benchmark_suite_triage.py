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
    hotspots = agg["quality_hotspots"]
    assert "likely_reference_divergence" in hotspots
    assert "likely_pipeline_failure" in hotspots


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
