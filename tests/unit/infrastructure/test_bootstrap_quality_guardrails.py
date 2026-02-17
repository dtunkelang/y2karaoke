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
