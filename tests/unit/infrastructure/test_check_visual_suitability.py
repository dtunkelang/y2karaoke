import importlib.util
import sys
from pathlib import Path
import pytest

# Dynamic import of the tool
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


def test_calculate_visual_suitability_perfect_score() -> None:
    # Simulate frames with clear word-level highlighting
    raw_frames = [
        {
            "time": 1.0,
            "words": [
                {"text": "Hello", "color": "selected", "y": 100, "confidence": 1.0},
                {"text": "World", "color": "unselected", "y": 100, "confidence": 1.0},
            ],
        },
        {
            "time": 2.0,
            "words": [
                {"text": "Hello", "color": "selected", "y": 100, "confidence": 1.0},
                {"text": "World", "color": "selected", "y": 100, "confidence": 1.0},
            ],
        },
    ]
    metrics = calculate_visual_suitability(raw_frames)

    # 1 out of 2 active frames has mixed states (word-level evidence)
    assert metrics["word_level_score"] == 0.5
    assert metrics["avg_ocr_confidence"] == 1.0
    assert metrics["has_word_level_highlighting"] is True
    # Score = 1.0 * 0.7 + min(0.5 * 2.0, 1.0) * 0.3 = 0.7 + 0.3 = 1.0
    assert metrics["detectability_score"] == 1.0


def test_calculate_visual_suitability_no_word_level() -> None:
    # Simulate line-level only highlighting (all words change at once)
    raw_frames = [
        {
            "time": 1.0,
            "words": [
                {"text": "Line", "color": "unselected", "y": 100, "confidence": 0.8},
                {"text": "One", "color": "unselected", "y": 100, "confidence": 0.8},
            ],
        },
        {
            "time": 2.0,
            "words": [
                {"text": "Line", "color": "selected", "y": 100, "confidence": 0.8},
                {"text": "One", "color": "selected", "y": 100, "confidence": 0.8},
            ],
        },
    ]
    metrics = calculate_visual_suitability(raw_frames)

    assert metrics["word_level_score"] == 0.0
    assert metrics["avg_ocr_confidence"] == 0.8
    assert metrics["has_word_level_highlighting"] is False
    # Score = 0.8 * 0.7 + 0.0 * 0.3 = 0.56
    assert pytest.approx(metrics["detectability_score"]) == 0.56


def test_calculate_visual_suitability_empty_frames() -> None:
    metrics = calculate_visual_suitability([])
    assert metrics["word_level_score"] == 0.0
    assert metrics["avg_ocr_confidence"] == 0.0
    assert metrics["detectability_score"] == 0.0
