from y2karaoke.core.models import Line, Word
from y2karaoke.core.components.alignment.timing_models import TranscriptionSegment
from y2karaoke.core.components.whisper.whisper_alignment_retime import (
    _retime_adjacent_lines_to_whisper_window,
)


def _line(text: str, start: float, end: float) -> Line:
    return Line(
        words=[Word(text=word, start_time=start, end_time=end) for word in text.split()]
    )


def test_retime_adjacent_lines_to_whisper_window_skips_short_repeated_refrain_pairs():
    lines = [
        _line("If you're lost you can look and you will find me", 1.1, 4.7),
        _line("Time after time", 6.0, 7.4),
        _line("If you fall I will catch you I'll be waiting", 8.75, 11.95),
        _line("Time after time", 13.3, 15.0),
    ]
    segments = [
        TranscriptionSegment(
            start=0.96,
            end=7.52,
            text="If you're lost you can look and you will find me time after time",
        )
    ]

    adjusted, fixes = _retime_adjacent_lines_to_whisper_window(lines, segments, "en")

    assert fixes == 0
    assert adjusted[0].start_time == lines[0].start_time
    assert adjusted[1].start_time == lines[1].start_time
