from y2karaoke.core.models import Line, Word
from y2karaoke.core.components.alignment.timing_models import TranscriptionSegment
from y2karaoke.core.components.whisper.whisper_alignment_pull_helpers import (
    line_neighbors,
    nearest_prior_segment_by_end,
    nearest_segment_by_start,
    reflow_two_lines_to_segment,
    reflow_words_to_window,
)


def test_nearest_segment_by_start_with_window():
    segments = [
        TranscriptionSegment(start=10.0, end=11.0, text="a", words=[]),
        TranscriptionSegment(start=20.0, end=21.0, text="b", words=[]),
    ]
    assert nearest_segment_by_start(10.4, segments, 1.0) is segments[0]
    assert nearest_segment_by_start(18.0, segments, 1.0) is None


def test_nearest_prior_segment_by_end():
    segments = [
        TranscriptionSegment(start=5.0, end=7.0, text="a", words=[]),
        TranscriptionSegment(start=10.0, end=12.0, text="b", words=[]),
    ]
    result = nearest_prior_segment_by_end(12.4, segments, 10.0)
    assert result is not None
    seg, late = result
    assert seg is segments[1]
    assert round(late, 2) == 0.4


def test_reflow_words_to_window_distributes_words():
    words = [
        Word(text="one", start_time=0.0, end_time=0.1),
        Word(text="two", start_time=0.2, end_time=0.3),
    ]
    out = reflow_words_to_window(words, 10.0, 12.0)
    assert out[0].start_time == 10.0
    assert out[1].start_time == 11.0
    assert round(out[1].end_time, 1) == 11.9


def test_reflow_two_lines_to_segment_respects_prev_end():
    line = Line(words=[Word(text="a", start_time=0.0, end_time=0.1)])
    next_line = Line(words=[Word(text="b", start_time=0.2, end_time=0.3)])
    segment = TranscriptionSegment(start=10.0, end=12.0, text="a b", words=[])

    reflowed = reflow_two_lines_to_segment(line, next_line, segment, prev_end=10.4)
    assert reflowed is not None
    left, right = reflowed
    assert left.start_time >= 10.41
    assert right.start_time > left.start_time


def test_line_neighbors():
    lines = [
        Line(words=[Word(text="a", start_time=1.0, end_time=2.0)]),
        Line(words=[Word(text="b", start_time=3.0, end_time=4.0)]),
        Line(words=[Word(text="c", start_time=5.0, end_time=6.0)]),
    ]
    prev_end, next_start = line_neighbors(lines, 1)
    assert prev_end == 2.0
    assert next_start == 5.0
