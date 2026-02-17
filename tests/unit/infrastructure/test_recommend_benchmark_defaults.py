import importlib.util
import json
import sys
from pathlib import Path

_MODULE_PATH = (
    Path(__file__).resolve().parents[3] / "tools" / "recommend_benchmark_defaults.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "recommend_benchmark_defaults_module", _MODULE_PATH
)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_score_row_prefers_lower_error_and_higher_coverage():
    a = {
        "agreement_start_p95_abs_sec_line_weighted_mean": 0.4,
        "agreement_start_mean_abs_sec_line_weighted_mean": 0.2,
        "low_confidence_ratio_line_weighted_mean": 0.1,
        "gold_start_abs_word_weighted_mean": 1.0,
        "dtw_line_coverage_line_weighted_mean": 0.8,
        "songs_succeeded": 2,
        "songs_total": 2,
    }
    b = {
        "agreement_start_p95_abs_sec_line_weighted_mean": 1.0,
        "agreement_start_mean_abs_sec_line_weighted_mean": 0.7,
        "low_confidence_ratio_line_weighted_mean": 0.4,
        "gold_start_abs_word_weighted_mean": 6.0,
        "dtw_line_coverage_line_weighted_mean": 0.4,
        "songs_succeeded": 1,
        "songs_total": 2,
    }
    assert _MODULE._score_row(a) > _MODULE._score_row(b)


def test_best_strategy_skips_failed_rows():
    rows = [
        {"strategy": "x", "status": "failed"},
        {
            "strategy": "y",
            "status": "finished",
            "agreement_start_p95_abs_sec_line_weighted_mean": 0.5,
            "agreement_start_mean_abs_sec_line_weighted_mean": 0.2,
            "low_confidence_ratio_line_weighted_mean": 0.1,
            "dtw_line_coverage_line_weighted_mean": 0.8,
            "songs_succeeded": 1,
            "songs_total": 1,
        },
    ]
    best = _MODULE._best_strategy(rows)
    assert best is not None
    assert best["strategy"] == "y"


def test_main_json_output(tmp_path, monkeypatch, capsys):
    matrix = tmp_path / "strategy_matrix_report.json"
    calib = tmp_path / "calibration.json"
    matrix.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "strategy": "hybrid_dtw",
                        "status": "finished",
                        "agreement_start_p95_abs_sec_line_weighted_mean": 0.4,
                        "agreement_start_mean_abs_sec_line_weighted_mean": 0.2,
                        "low_confidence_ratio_line_weighted_mean": 0.1,
                        "gold_start_abs_word_weighted_mean": 0.9,
                        "dtw_line_coverage_line_weighted_mean": 0.8,
                        "songs_succeeded": 1,
                        "songs_total": 1,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    calib.write_text(
        json.dumps(
            {
                "recommended": {
                    "min_detectability": 0.45,
                    "min_word_level_score": 0.15,
                    "min_line_confidence_mean": 0.3,
                    "min_word_confidence_mean": 0.25,
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "recommend_benchmark_defaults.py",
            "--matrix-report",
            str(matrix),
            "--calibration-report",
            str(calib),
            "--json",
        ],
    )
    assert _MODULE.main() == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["recommended_strategy"] == "hybrid_dtw"
    assert payload["recommended_bootstrap_thresholds"]["min_detectability"] == 0.45


def test_main_ignores_zero_calibrated_thresholds(tmp_path, monkeypatch, capsys):
    matrix = tmp_path / "strategy_matrix_report.json"
    calib = tmp_path / "calibration.json"
    matrix.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "strategy": "hybrid_dtw",
                        "status": "finished",
                        "dtw_line_coverage_line_weighted_mean": 0.8,
                        "songs_succeeded": 1,
                        "songs_total": 1,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    calib.write_text(
        json.dumps({"recommended": {"min_detectability": 0.0}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "recommend_benchmark_defaults.py",
            "--matrix-report",
            str(matrix),
            "--calibration-report",
            str(calib),
            "--json",
        ],
    )
    assert _MODULE.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["recommended_bootstrap_thresholds"] == {}
