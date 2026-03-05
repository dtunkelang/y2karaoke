"""Unit tests for main benchmark guardrails wrapper."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_module():
    module_path = (
        Path(__file__).resolve().parents[3] / "tools" / "main_benchmark_guardrails.py"
    )
    spec = importlib.util.spec_from_file_location(
        "main_benchmark_guardrails", module_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_check_thresholds_accepts_timing_quality_metric():
    module = _load_module()
    report = {"aggregate": {"timing_quality_score_line_weighted": 0.61}}
    issues = module._check_thresholds(
        report,
        {"min_timing_quality_score_line_weighted": 0.58},
    )
    assert issues == []


def test_check_thresholds_fails_low_timing_quality_metric():
    module = _load_module()
    report = {"aggregate": {"timing_quality_score_line_weighted": 0.5}}
    issues = module._check_thresholds(
        report,
        {"min_timing_quality_score_line_weighted": 0.58},
    )
    assert len(issues) == 1
    assert "timing_quality_score_line_weighted below threshold" in issues[0]


def test_metric_value_reference_divergence_fallback_for_legacy_reports():
    module = _load_module()
    report = {"aggregate": {}}
    assert module._metric_value(report, "reference_divergence_suspected_ratio") == 0.0
