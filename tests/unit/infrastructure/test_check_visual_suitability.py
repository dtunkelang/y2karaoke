import importlib.util
import sys
from pathlib import Path

_MODULE_PATH = (
    Path(__file__).resolve().parents[3] / "tools" / "check_visual_suitability.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "check_visual_suitability_module", _MODULE_PATH
)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

calculate_visual_suitability = _MODULE.calculate_visual_suitability


def test_calculate_visual_suitability() -> None:
    # High evidence: mixed states in lines
    raw_high = [
        {
            "words": [
                {"text": "A", "color": "selected", "y": 100, "x": 0, "confidence": 0.9},
                {
                    "text": "B",
                    "color": "unselected",
                    "y": 100,
                    "x": 50,
                    "confidence": 0.8,
                },
            ]
        }
    ]
    res_high = calculate_visual_suitability(raw_high)
    assert res_high["word_level_score"] == 1.0
    assert res_high["has_word_level_highlighting"] is True
    import pytest

    assert res_high["avg_ocr_confidence"] == pytest.approx(0.85)
    assert res_high["detectability_score"] > 0.8

    # Low evidence: only full line highlights
    raw_low = [
        {
            "words": [
                {"text": "A", "color": "selected", "y": 100, "x": 0, "confidence": 0.9},
                {
                    "text": "B",
                    "color": "selected",
                    "y": 100,
                    "x": 50,
                    "confidence": 0.9,
                },
            ]
        }
    ]
    res_low = calculate_visual_suitability(raw_low)
    assert res_low["word_level_score"] == 0.0
    assert res_low["has_word_level_highlighting"] is False
    assert res_low["avg_ocr_confidence"] == 0.9
