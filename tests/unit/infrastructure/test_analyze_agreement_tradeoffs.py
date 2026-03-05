import importlib.util
import json
from pathlib import Path
import sys

_MODULE_PATH = (
    Path(__file__).resolve().parents[3] / "tools" / "analyze_agreement_tradeoffs.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "analyze_agreement_tradeoffs_module", _MODULE_PATH
)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_analyze_agreement_tradeoffs_sorts_by_score_and_guard(tmp_path, monkeypatch):
    baseline = tmp_path / "baseline"
    c1 = tmp_path / "c1"
    c2 = tmp_path / "c2"
    baseline.mkdir()
    c1.mkdir()
    c2.mkdir()
    baseline_doc = {
        "aggregate": {
            "agreement_coverage_ratio_total": 0.28,
            "agreement_start_p95_abs_sec_mean": 1.00,
            "agreement_bad_ratio_total": 0.05,
            "timing_quality_score_line_weighted_mean": 0.78,
        }
    }
    c1_doc = {
        "aggregate": {
            "agreement_coverage_ratio_total": 0.31,
            "agreement_start_p95_abs_sec_mean": 1.02,
            "agreement_bad_ratio_total": 0.051,
            "timing_quality_score_line_weighted_mean": 0.781,
        }
    }
    c2_doc = {
        "aggregate": {
            "agreement_coverage_ratio_total": 0.35,
            "agreement_start_p95_abs_sec_mean": 1.20,
            "agreement_bad_ratio_total": 0.09,
            "timing_quality_score_line_weighted_mean": 0.78,
        }
    }
    (baseline / "benchmark_report.json").write_text(
        json.dumps(baseline_doc), encoding="utf-8"
    )
    (c1 / "benchmark_report.json").write_text(json.dumps(c1_doc), encoding="utf-8")
    (c2 / "benchmark_report.json").write_text(json.dumps(c2_doc), encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_agreement_tradeoffs.py",
            "--baseline",
            f"base={baseline}",
            "--candidate",
            f"relax={c1}",
            "--candidate",
            f"too_risky={c2}",
        ],
    )
    rc = _MODULE.main()
    assert rc == 0
    out = json.loads((tmp_path / "agreement_tradeoff_analysis.json").read_text("utf-8"))
    assert out["baseline_label"] == "base"
    assert len(out["rows"]) == 2
    assert out["rows"][0]["label"] == "relax"
    assert out["rows"][0]["passes_tradeoff_guard"] is True
    assert out["rows"][1]["label"] == "too_risky"
    assert out["rows"][1]["passes_tradeoff_guard"] is False
