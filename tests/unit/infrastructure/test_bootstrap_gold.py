from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path before other project imports
sys.path.append(str(Path(__file__).resolve().parents[3]))

from tools.bootstrap_gold_from_karaoke import (  # noqa: E402
    _snap,
    _text_similarity,
    TargetLine,
)


def test_snap():
    assert _snap(1.234) == 1.25
    assert _snap(1.276) == 1.30
    assert _snap(1.31) == 1.3
    assert _snap(0.01) == 0.0


def test_text_similarity():
    assert _text_similarity("Hello", "hello") == 1.0
    assert _text_similarity("Hello world!", "hello world") > 0.9
    assert _text_similarity("Testing 123", "Test 123") > 0.8
    assert _text_similarity("abc", "xyz") == 0.0


def test_duration_balancing_logic():
    # We can't easily test the full build_gold function due to OCR dependencies,
    # but we can verify the logic if we extract it.
    # For now, we'll verify basic TargetLine construction.
    line = TargetLine(
        line_index=1,
        start=10.0,
        end=None,
        text="Test line",
        words=["Test", "line"],
        y=100.0,
    )
    assert line.text == "Test line"
    assert len(line.words) == 2
    assert line.start == 10.0
    assert line.y == 100.0
