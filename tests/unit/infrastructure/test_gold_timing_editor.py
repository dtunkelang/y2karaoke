import importlib.util
import sys
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[3] / "tools" / "gold_timing_editor.py"
_SPEC = importlib.util.spec_from_file_location(
    "gold_timing_editor_module", _MODULE_PATH
)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

ValidationError = _MODULE.ValidationError
_from_timing_report = _MODULE._from_timing_report
_validate_and_normalize_gold = _MODULE._validate_and_normalize_gold


def test_from_timing_report_normalizes_and_snaps() -> None:
    report = {
        "title": "bad guy",
        "artist": "Billie Eilish",
        "lines": [
            {
                "words": [
                    {"text": "Hello", "start": 1.04, "end": 1.26},
                    {"text": "world", "start": 1.26, "end": 1.91},
                ]
            }
        ],
    }

    doc = _from_timing_report(report)
    words = doc["lines"][0]["words"]
    assert words[0]["start"] == 1.05
    assert words[0]["end"] == 1.25
    assert words[1]["start"] == 1.25
    assert words[1]["end"] == 1.90


def test_from_timing_report_clamps_overlap_forward() -> None:
    report = {
        "title": "x",
        "artist": "y",
        "lines": [
            {
                "words": [
                    {"text": "a", "start": 1.0, "end": 1.5},
                    {"text": "b", "start": 1.2, "end": 1.6},
                ]
            }
        ],
    }

    doc = _from_timing_report(report)
    words = doc["lines"][0]["words"]
    assert words[0]["end"] == 1.5
    assert words[1]["start"] == 1.5
    assert words[1]["end"] == 1.6


def test_validate_and_normalize_gold_rejects_overlap() -> None:
    doc = {
        "title": "x",
        "artist": "y",
        "lines": [
            {
                "words": [
                    {"text": "a", "start": 1.0, "end": 1.6},
                    {"text": "b", "start": 1.5, "end": 1.8},
                ]
            }
        ],
    }

    with pytest.raises(ValidationError):
        _validate_and_normalize_gold(doc)


def test_validate_and_normalize_gold_requires_numeric_times() -> None:
    doc = {
        "title": "x",
        "artist": "y",
        "lines": [{"words": [{"text": "a", "start": "nope", "end": 1.0}]}],
    }

    with pytest.raises(ValidationError):
        _validate_and_normalize_gold(doc)
