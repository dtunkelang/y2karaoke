import sys
from pathlib import Path

# Add project root to sys.path before other project imports
sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.y2karaoke.core.models import TargetLine  # noqa: E402
from src.y2karaoke.core.text_utils import (  # noqa: E402
    text_similarity,
    normalize_ocr_line,
    normalize_text_basic as normalize_text,
)


def test_text_similarity_is_case_and_punctuation_tolerant() -> None:
    s = text_similarity(
        "White shirt now red, my bloody nose", "white shirt now red my bloody nose"
    )
    assert s > 0.95


def test_normalize_text_handles_hyphens() -> None:
    # Ensure 'anti-hero' matches 'anti hero'
    assert normalize_text("anti-hero") == "anti hero"
    assert normalize_text("Anti-Hero!!!") == "anti hero"


def test_normalize_ocr_line_fixes_typos() -> None:
    assert normalize_ocr_line("the problei") == "the problem"
    assert normalize_ocr_line("have this thing") == "I have this thing"


def test_target_line_construction() -> None:
    line = TargetLine(
        line_index=1,
        start=10.0,
        end=15.0,
        text="Hello world",
        words=["Hello", "world"],
        y=100.0,
    )
    assert line.text == "Hello world"
    assert line.y == 100.0
