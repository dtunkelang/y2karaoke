"""Tail-guardrail diagnostics tests for benchmark suite helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_module():
    module_path = (
        Path(__file__).resolve().parents[3] / "tools" / "run_benchmark_suite.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_benchmark_suite_tail_guardrail", module_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_extract_alignment_diagnostics_includes_tail_guardrail_fields():
    module = _load_module()
    report = {
        "alignment_method": "whisper_hybrid",
        "lyrics_source": "lyriq(provider)",
        "issues": [
            "Tail completeness guardrail flagged possible truncated lyric ending"
        ],
        "dtw_metrics": {
            "fallback_map_attempted": 1.0,
            "fallback_map_selected": 0.0,
            "fallback_map_rejected": 1.0,
            "fallback_map_decision_code": 2.0,
            "fallback_map_score_gain": 0.0123,
            "tail_guardrail_flagged": 1.0,
            "tail_guardrail_fallback_attempted": 1.0,
            "tail_guardrail_fallback_applied": 0.0,
            "tail_guardrail_target_coverage_ratio": 0.81,
            "tail_guardrail_target_shortfall_sec": 28.1,
        },
    }

    diag = module._extract_alignment_diagnostics(report)

    assert diag["fallback_map_attempted"] is True
    assert diag["fallback_map_selected"] is False
    assert diag["fallback_map_rejected"] is True
    assert diag["fallback_map_decision_reason"] == "rejected_insufficient_score_gain"
    assert diag["fallback_map_score_gain"] == 0.0123
    assert diag["tail_guardrail_flagged"] is True
    assert diag["tail_guardrail_fallback_attempted"] is True
    assert diag["tail_guardrail_fallback_applied"] is False
    assert diag["tail_guardrail_target_coverage_ratio"] == 0.81
    assert diag["tail_guardrail_target_shortfall_sec"] == 28.1
    assert "tail_completeness_guardrail" in diag["issue_tags"]
