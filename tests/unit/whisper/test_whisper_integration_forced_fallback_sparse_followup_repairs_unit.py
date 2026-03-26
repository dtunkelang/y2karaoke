import pytest

from y2karaoke.core.components.whisper import (
    whisper_forced_sparse_followup_repairs as _repairs,
)
from y2karaoke.core.models import Line, Word


def test_restore_sparse_forced_followup_lines_from_source_restores_collapsed_followup():
    baseline_lines = [
        Line(
            words=[
                Word(text="Help", start_time=1.1, end_time=1.6),
                Word(text="me", start_time=1.6, end_time=2.1),
                Word(text="make", start_time=2.1, end_time=2.7),
                Word(text="the", start_time=2.7, end_time=3.2),
                Word(text="most", start_time=3.2, end_time=3.75),
                Word(text="of", start_time=3.75, end_time=4.25),
                Word(text="freedom", start_time=4.25, end_time=5.25),
                Word(text="and", start_time=5.25, end_time=5.75),
                Word(text="of", start_time=5.75, end_time=6.45),
                Word(text="pleasure", start_time=6.45, end_time=7.35),
            ]
        ),
        Line(
            words=[
                Word(text="Nothing", start_time=7.2, end_time=8.2),
                Word(text="ever", start_time=8.2, end_time=8.9),
                Word(text="lasts", start_time=8.9, end_time=9.9),
                Word(text="forever", start_time=9.9, end_time=11.2),
            ]
        ),
    ]
    forced_lines = [
        Line(
            words=[
                Word(text="Help", start_time=1.265, end_time=1.746),
                Word(text="me", start_time=1.827, end_time=2.208),
                Word(text="make", start_time=2.348, end_time=2.789),
                Word(text="the", start_time=2.91, end_time=3.331),
                Word(text="most", start_time=3.451, end_time=4.053),
                Word(text="of", start_time=4.273, end_time=4.454),
                Word(text="freedom", start_time=4.534, end_time=5.316),
                Word(text="and", start_time=5.898, end_time=6.078),
                Word(text="of", start_time=6.419, end_time=6.68),
                Word(text="pleasure", start_time=6.7, end_time=7.422),
            ]
        ),
        Line(
            words=[
                Word(text="Nothing", start_time=6.825, end_time=7.367),
                Word(text="ever", start_time=7.407, end_time=7.728),
                Word(text="lasts", start_time=7.768, end_time=8.169),
                Word(text="forever", start_time=8.29, end_time=8.791),
            ]
        ),
    ]
    aligned_segments = [
        {"words": [{"score": 0.75}] * 10},
        {
            "words": [
                {"score": 0.495},
                {"score": 0.384},
                {"score": 0.576},
                {"score": 0.341},
            ]
        },
    ]

    repaired_lines, restored = (
        _repairs.restore_sparse_forced_followup_lines_from_source(
            baseline_lines,
            forced_lines,
            aligned_segments,
        )
    )

    assert restored == 1
    assert repaired_lines[1].start_time == pytest.approx(7.472)
    assert repaired_lines[1].end_time == pytest.approx(11.2)


def test_restore_sparse_forced_followup_lines_from_source_skips_supported_followup():
    baseline_lines = [
        Line(
            words=[
                Word(text="Help", start_time=1.1, end_time=1.6),
                Word(text="me", start_time=1.6, end_time=2.1),
                Word(text="make", start_time=2.1, end_time=2.7),
                Word(text="the", start_time=2.7, end_time=3.2),
                Word(text="most", start_time=3.2, end_time=3.75),
                Word(text="of", start_time=3.75, end_time=4.25),
                Word(text="freedom", start_time=4.25, end_time=5.25),
                Word(text="and", start_time=5.25, end_time=5.75),
                Word(text="of", start_time=5.75, end_time=6.45),
                Word(text="pleasure", start_time=6.45, end_time=7.35),
            ]
        ),
        Line(
            words=[
                Word(text="Nothing", start_time=7.2, end_time=8.2),
                Word(text="ever", start_time=8.2, end_time=8.9),
                Word(text="lasts", start_time=8.9, end_time=9.9),
                Word(text="forever", start_time=9.9, end_time=11.2),
            ]
        ),
    ]
    forced_lines = [
        baseline_lines[0],
        Line(
            words=[
                Word(text="Nothing", start_time=7.15, end_time=8.0),
                Word(text="ever", start_time=8.0, end_time=8.75),
                Word(text="lasts", start_time=8.75, end_time=9.8),
                Word(text="forever", start_time=9.8, end_time=10.95),
            ]
        ),
    ]
    aligned_segments = [
        {"words": [{"score": 0.75}] * 10},
        {"words": [{"score": 0.75}] * 4},
    ]

    repaired_lines, restored = (
        _repairs.restore_sparse_forced_followup_lines_from_source(
            baseline_lines,
            forced_lines,
            aligned_segments,
        )
    )

    assert restored == 0
    assert repaired_lines[1].start_time == forced_lines[1].start_time
    assert repaired_lines[1].end_time == forced_lines[1].end_time
