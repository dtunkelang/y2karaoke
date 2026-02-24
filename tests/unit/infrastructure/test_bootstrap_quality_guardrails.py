import importlib.util
import json
import sys
from pathlib import Path

_MODULE_PATH = (
    Path(__file__).resolve().parents[3] / "tools" / "bootstrap_quality_guardrails.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "bootstrap_quality_guardrails_module", _MODULE_PATH
)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_check_doc_passes_with_valid_confidence_and_suitability():
    doc = {
        "candidate_url": "https://youtube.com/watch?v=abc123def45",
        "visual_suitability": {
            "detectability_score": 0.7,
            "word_level_score": 0.25,
        },
        "lines": [
            {
                "confidence": 0.6,
                "words": [
                    {"text": "hello", "confidence": 0.7},
                    {"text": "world", "confidence": 0.5},
                ],
            }
        ],
    }

    issues = _MODULE._check_doc(
        doc,
        min_detectability=0.3,
        min_word_level_score=0.1,
        min_word_conf_mean=0.25,
        min_line_conf_mean=0.25,
    )
    assert issues == []


def test_check_doc_fails_when_confidence_missing():
    doc = {
        "candidate_url": "https://youtube.com/watch?v=abc123def45",
        "visual_suitability": {
            "detectability_score": 0.7,
            "word_level_score": 0.25,
        },
        "lines": [
            {
                "words": [
                    {"text": "hello"},
                    {"text": "world"},
                ],
            }
        ],
    }

    issues = _MODULE._check_doc(
        doc,
        min_detectability=0.3,
        min_word_level_score=0.1,
        min_word_conf_mean=0.25,
        min_line_conf_mean=0.25,
    )
    assert any("missing line confidence" in issue for issue in issues)
    assert any("missing word confidence" in issue for issue in issues)


def test_main_skips_non_visual_docs(tmp_path, monkeypatch):
    repo = tmp_path
    p = repo / "benchmarks" / "gold_set" / "song.gold.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"title": "x", "lines": []}), encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bootstrap_quality_guardrails.py",
            "--root",
            str(repo),
        ],
    )

    assert _MODULE.main() == 0


def test_check_visual_eval_summary_flags_low_f1():
    summary = {
        "songs": [
            {
                "index": 1,
                "artist": "A",
                "title": "B",
                "status": "ok",
                "strict": {"f1": 0.12},
                "repeat_capped": {"f1": 0.2},
            }
        ]
    }
    issues = _MODULE._check_visual_eval_summary(
        summary,
        min_strict_f1=0.2,
        min_repeat_capped_f1=0.1,
    )
    assert any("strict f1 too low" in issue for issue in issues)
    assert not any("repeat_capped f1 too low" in issue for issue in issues)


def test_main_fails_when_visual_eval_summary_below_threshold(tmp_path, monkeypatch):
    repo = tmp_path
    p = repo / "benchmarks" / "gold_set_karaoke_seed" / "01_demo.visual.gold.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(
            {
                "candidate_url": "https://youtube.com/watch?v=abc123def45",
                "visual_suitability": {
                    "detectability_score": 0.7,
                    "word_level_score": 0.25,
                },
                "lines": [
                    {
                        "confidence": 0.8,
                        "words": [{"text": "hello", "confidence": 0.9}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    eval_summary = repo / "benchmarks" / "results" / "visual_eval_summary.json"
    eval_summary.parent.mkdir(parents=True, exist_ok=True)
    eval_summary.write_text(
        json.dumps(
            {
                "songs": [
                    {
                        "index": 1,
                        "artist": "Demo",
                        "title": "Song",
                        "status": "ok",
                        "strict": {"f1": 0.05},
                        "repeat_capped": {"f1": 0.08},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bootstrap_quality_guardrails.py",
            "--root",
            str(repo),
            "--visual-eval-summary-json",
            str(eval_summary),
            "--min-visual-eval-strict-f1",
            "0.10",
        ],
    )

    assert _MODULE.main() == 1


def test_check_visual_eval_summary_flags_low_aggregate_metrics():
    summary = {
        "summary": {
            "strict_f1_mean": 0.11,
            "repeat_capped_f1_mean": 0.22,
            "strict_f1_median": 0.10,
            "repeat_capped_f1_median": 0.20,
        },
        "songs": [],
    }
    issues = _MODULE._check_visual_eval_summary(
        summary,
        min_strict_f1=None,
        min_repeat_capped_f1=None,
        min_strict_f1_mean=0.2,
        min_repeat_capped_f1_mean=0.2,
        min_strict_f1_median=0.15,
        min_repeat_capped_f1_median=0.15,
    )
    assert any("strict_f1_mean too low" in issue for issue in issues)
    assert any("strict_f1_median too low" in issue for issue in issues)
    assert not any("repeat_capped_f1_mean too low" in issue for issue in issues)
