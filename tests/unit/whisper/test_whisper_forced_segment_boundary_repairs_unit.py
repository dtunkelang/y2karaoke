import pytest

from y2karaoke.core.components.alignment.timing_models import TranscriptionSegment
from y2karaoke.core.components.whisper.whisper_forced_segment_boundary_repairs import (
    restore_forced_exact_adjacent_segment_boundaries,
)
from y2karaoke.core.models import Line, Word


def _dur_multi_line(start: float, end: float, tokens: list[str]) -> Line:
    step = (end - start) / max(len(tokens), 1)
    words = [
        Word(
            text=token,
            start_time=start + step * idx,
            end_time=start + step * (idx + 1),
        )
        for idx, token in enumerate(tokens)
    ]
    return Line(words=words)


def test_restore_forced_exact_adjacent_segment_boundaries_repairs_hotline_boundary():
    forced_lines = [
        _dur_multi_line(
            0.757,
            3.184,
            ["You", "used", "to", "call", "me", "on", "my", "cell", "phone"],
        ),
        _dur_multi_line(
            5.005, 7.17, ["Late", "night", "when", "you", "need", "my", "love"]
        ),
        _dur_multi_line(7.19, 9.195, ["Call", "me", "on", "my", "cell", "phone"]),
    ]
    transcription = [
        TranscriptionSegment(
            start=0.4,
            end=2.84,
            text="You used to call me on my cell phone",
            words=[],
        ),
        TranscriptionSegment(
            start=2.84,
            end=7.74,
            text="Late night when you need my love",
            words=[],
        ),
        TranscriptionSegment(
            start=7.74,
            end=9.96,
            text="Call me on my cell phone",
            words=[],
        ),
    ]

    repaired, count = restore_forced_exact_adjacent_segment_boundaries(
        forced_lines,
        transcription,
    )

    assert count == 1
    assert repaired[1].start_time == pytest.approx(5.005)
    assert repaired[1].end_time == pytest.approx(7.69)
    assert repaired[2].start_time == pytest.approx(7.74)
    assert repaired[2].end_time == pytest.approx(9.96)


def test_restore_forced_exact_adjacent_segment_boundaries_skips_exact_control():
    forced_lines = [
        _dur_multi_line(1.011, 3.656, ["Please", "please", "please"]),
        _dur_multi_line(4.392, 6.442, ["Don't", "prove", "I'm", "right"]),
        _dur_multi_line(9.782, 12.736, ["And", "please", "please", "please"]),
    ]
    transcription = [
        TranscriptionSegment(
            start=1.011,
            end=3.656,
            text="Please please please",
            words=[],
        ),
        TranscriptionSegment(
            start=4.392,
            end=6.442,
            text="Don't prove I'm right",
            words=[],
        ),
        TranscriptionSegment(
            start=9.782,
            end=12.736,
            text="And please please please",
            words=[],
        ),
    ]

    repaired, count = restore_forced_exact_adjacent_segment_boundaries(
        forced_lines,
        transcription,
    )

    assert count == 0
    assert repaired == forced_lines
