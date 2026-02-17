import importlib.util
import json
import sys
from pathlib import Path

_MODULE_PATH = (
    Path(__file__).resolve().parents[3] / "tools" / "bootstrap_calibrate_thresholds.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "bootstrap_calibrate_thresholds_module", _MODULE_PATH
)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_collect_metrics_from_reports_and_outputs(tmp_path):
    gold1 = tmp_path / "song1.visual.gold.json"
    gold1.write_text(
        json.dumps(
            {
                "lines": [
                    {
                        "confidence": 0.7,
                        "words": [
                            {"text": "a", "confidence": 0.6},
                            {"text": "b", "confidence": 0.8},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    docs = [
        {
            "selected_visual_suitability": {
                "detectability_score": 0.65,
                "word_level_score": 0.22,
            },
            "output_path": str(gold1),
        }
    ]

    metrics = _MODULE._collect_metrics(docs)
    assert metrics["detectability"] == [0.65]
    assert metrics["word_level"] == [0.22]
    assert metrics["line_conf"] == [0.7]
    assert metrics["word_conf"] == [0.6, 0.8]


def test_pct_handles_small_inputs():
    assert _MODULE._pct([], 0.2) == 0.0
    assert _MODULE._pct([0.5], 0.2) == 0.5
    val = _MODULE._pct([0.1, 0.2, 0.8, 0.9], 0.5)
    assert 0.15 <= val <= 0.85


def test_main_json_output(tmp_path, monkeypatch, capsys):
    report = tmp_path / "benchmarks" / "x.bootstrap-report.json"
    gold = tmp_path / "benchmarks" / "x.visual.gold.json"
    report.parent.mkdir(parents=True, exist_ok=True)

    gold.write_text(
        json.dumps(
            {
                "lines": [
                    {
                        "confidence": 0.55,
                        "words": [
                            {"text": "a", "confidence": 0.45},
                            {"text": "b", "confidence": 0.65},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report.write_text(
        json.dumps(
            {
                "selected_visual_suitability": {
                    "detectability_score": 0.6,
                    "word_level_score": 0.2,
                },
                "output_path": str(gold),
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bootstrap_calibrate_thresholds.py",
            "--root",
            str(tmp_path),
            "--json",
        ],
    )

    assert _MODULE.main() == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["report_count"] == 1
    assert payload["recommended"]["min_detectability"] >= 0.0
