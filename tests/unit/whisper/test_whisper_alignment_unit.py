from y2karaoke.core.models import Line, Word
import y2karaoke.core.components.whisper.whisper_alignment as wa
from y2karaoke.core.components.alignment.timing_models import (
    TranscriptionWord,
    TranscriptionSegment,
)
import numpy as np


def test_enforce_monotonic_line_starts_handles_empty_lines():
    lines = [
        Line(words=[]),
        Line(words=[Word(text="hello", start_time=10.0, end_time=11.0)]),
    ]
    adjusted = wa._enforce_monotonic_line_starts(lines)
    assert len(adjusted) == 2
    assert not adjusted[0].words
    assert adjusted[1].start_time == 10.0


def test_enforce_monotonic_line_starts_fixes_order():
    lines = [
        Line(words=[Word(text="first", start_time=10.0, end_time=11.0)]),
        Line(words=[Word(text="second", start_time=9.0, end_time=9.5)]),
    ]
    adjusted = wa._enforce_monotonic_line_starts(lines, min_gap=0.1)
    assert adjusted[0].start_time == 10.0
    assert adjusted[1].start_time == 10.1
    assert adjusted[1].words[0].text == "second"


def test_scale_line_to_duration_simple():
    line = Line(
        words=[
            Word(text="one", start_time=10.0, end_time=11.0),
            Word(text="two", start_time=11.0, end_time=12.0),
        ]
    )
    # Total duration 2.0s -> scale to 1.0s
    scaled = wa._scale_line_to_duration(line, 1.0)
    assert scaled.start_time == 10.0
    assert scaled.end_time == 11.0
    assert scaled.words[0].end_time == 10.5
    assert scaled.words[1].start_time == 10.5


def test_scale_line_to_duration_edge_cases():
    # Empty words
    line = Line(words=[])
    assert wa._scale_line_to_duration(line, 1.0) == line

    # Zero duration line
    line = Line(words=[Word(text="a", start_time=10.0, end_time=10.0)])
    assert wa._scale_line_to_duration(line, 1.0) == line

    # Zero target duration
    line = Line(words=[Word(text="a", start_time=10.0, end_time=11.0)])
    assert wa._scale_line_to_duration(line, 0.0) == line


def test_enforce_non_overlapping_lines_shifts_and_scales():
    lines = [
        Line(words=[Word(text="one", start_time=10.0, end_time=12.0)]),
        Line(
            words=[Word(text="two", start_time=11.0, end_time=13.0)]
        ),  # Overlaps start
        Line(words=[Word(text="three", start_time=14.0, end_time=16.0)]),
    ]
    # min_gap=0.1
    adjusted = wa._enforce_non_overlapping_lines(lines, min_gap=0.1)

    # Line 1 should be pushed after Line 0
    assert adjusted[1].start_time >= adjusted[0].end_time + 0.1

    # Now create a squeeze case
    lines = [
        Line(words=[Word(text="one", start_time=10.0, end_time=12.0)]),
        Line(words=[Word(text="two", start_time=11.0, end_time=15.0)]),
        Line(words=[Word(text="three", start_time=14.0, end_time=15.0)]),
    ]
    # Line 2 starts at 14. Line 1 ends at 15. Line 1 must be scaled to fit between 12.1 and 13.9.
    adjusted = wa._enforce_non_overlapping_lines(lines, min_gap=0.1)
    assert adjusted[1].end_time <= adjusted[2].start_time - 0.1


def test_normalize_line_word_timings():
    # Word end < start
    line = Line(words=[Word(text="bad", start_time=10.0, end_time=9.0)])
    norm = wa._normalize_line_word_timings([line])[0]
    assert norm.words[0].end_time >= 10.05

    # Words out of order
    line = Line(
        words=[
            Word(text="first", start_time=10.0, end_time=11.0),
            Word(text="second", start_time=10.5, end_time=11.5),
        ]
    )
    norm = wa._normalize_line_word_timings([line])[0]
    assert norm.words[1].start_time >= 11.01


def test_interpolate_unmatched_lines():
    lines = [
        Line(words=[Word(text="matched1", start_time=10.0, end_time=11.0)]),
        Line(words=[Word(text="unmatched", start_time=20.0, end_time=21.0)]),
        Line(words=[Word(text="matched2", start_time=30.0, end_time=31.0)]),
    ]
    matched_indices = {0, 2}
    interpolated = wa._interpolate_unmatched_lines(lines, matched_indices)

    # Line 1 should be moved between 11.0 and 30.0
    assert 11.0 <= interpolated[1].start_time < 30.0
    # It should be shifted from its original 20.0 to somewhere reasonable
    assert interpolated[1].start_time == 11.0


def test_merge_first_two_lines_if_segment_matches(monkeypatch):
    lines = [
        Line(words=[Word(text="one", start_time=10.0, end_time=11.0)]),
        Line(words=[Word(text="two", start_time=12.0, end_time=13.0)]),
    ]
    segments = [
        TranscriptionSegment(start=10.0, end=13.5, text="one two", words=[]),
    ]

    import y2karaoke.core.components.whisper.whisper_alignment_segments as was

    monkeypatch.setattr(
        was, "_find_best_whisper_segment", lambda *args: (segments[0], 0.9, 0.0)
    )

    merged, success = wa._merge_first_two_lines_if_segment_matches(
        lines, segments, "fra-Latn"
    )
    assert success
    assert len(merged[0].words) == 2
    assert not merged[1].words


def test_drop_duplicate_lines(monkeypatch):
    lines = [
        Line(words=[Word(text="hello", start_time=10.0, end_time=11.0)]),
        Line(words=[Word(text="hello", start_time=12.0, end_time=13.0)]),
    ]
    segments = [
        TranscriptionSegment(start=10.0, end=13.0, text="hello", words=[]),
    ]

    import y2karaoke.core.components.whisper.whisper_alignment_segments as was

    # Both lines map to same segment
    monkeypatch.setattr(
        was, "_find_best_whisper_segment", lambda *args: (segments[0], 0.9, 0.0)
    )

    deduped, count = wa._drop_duplicate_lines(lines, segments, "fra-Latn")
    assert count == 1
    assert len(deduped) == 1


def test_drop_duplicate_lines_by_timing():
    lines = [
        Line(words=[Word(text="hello", start_time=10.0, end_time=11.0)]),
        Line(words=[Word(text="hello", start_time=11.1, end_time=12.0)]),
    ]
    # max_gap = 0.2. 11.1 - 11.0 = 0.1 < 0.2 -> drop
    deduped, count = wa._drop_duplicate_lines_by_timing(lines, max_gap=0.2)
    assert count == 1
    assert len(deduped) == 1


def test_pull_lines_forward_for_continuous_vocals():
    lines = [
        Line(words=[Word(text="one", start_time=10.0, end_time=11.0)]),
        Line(words=[Word(text="two", start_time=20.0, end_time=21.0)]),
    ]

    class MockAF:
        def __init__(self):
            self.onset_times = np.array([12.0, 15.0, 18.0])

    af = MockAF()

    import y2karaoke.core.components.whisper.whisper_alignment_refinement as war

    with war.use_alignment_refinement_hooks(
        check_vocal_activity_in_range_fn=lambda *args: 0.8,
        check_for_silence_in_range_fn=lambda *args, **kw: False,
    ):
        pulled, count = wa._pull_lines_forward_for_continuous_vocals(
            lines, af, max_gap=4.0
        )
    # gap is 9.0 > 4.0. Activity 0.8 > 0.6. No silence.
    # Should pull to first onset after prev end: 12.0
    assert count == 1
    assert pulled[1].start_time == 12.0


def test_retime_line_to_segment():
    line = Line(words=[Word(text="a", start_time=0, end_time=1)])
    seg = TranscriptionSegment(start=10.0, end=12.0, text="a", words=[])
    retimed = wa._retime_line_to_segment(line, seg)
    assert retimed.start_time == 10.0
    assert retimed.end_time == 11.8  # 10 + 0.9 * spacing


def test_retime_line_to_window():
    line = Line(words=[Word(text="a", start_time=0, end_time=1)])
    retimed = wa._retime_line_to_window(line, 20.0, 30.0)
    assert retimed.start_time == 20.0
    assert retimed.end_time == 29.0


def test_clamp_repeated_line_duration():
    lines = [
        Line(words=[Word(text="repeat", start_time=10.0, end_time=11.0)]),
        Line(words=[Word(text="repeat", start_time=12.0, end_time=15.0)]),
    ]
    # max_duration = 1.5
    clamped, count = wa._clamp_repeated_line_duration(lines, max_duration=1.5)
    assert count == 1
    assert clamped[1].end_time == 13.35  # 12 + 1.5 * 0.9


def test_fill_vocal_activity_gaps():
    words = [
        TranscriptionWord(text="hello", start=10.0, end=11.0, probability=1.0),
        TranscriptionWord(text="world", start=20.0, end=21.0, probability=1.0),
    ]
    # Mock audio features with high vocal activity in the gap
    # _check_vocal_activity_in_range is used
    # Threshold is 0.3

    class MockAudioFeatures:
        def __init__(self):
            self.vocal_start = 5.0
            self.vocal_end = 25.0
            self.onset_times = np.array([5.5, 15.0, 24.0])
            self.rms = np.ones(100)
            self.rms_times = np.linspace(0, 30, 100)
            self.silence_regions = []

    af = MockAudioFeatures()

    from unittest.mock import patch

    with patch(
        "y2karaoke.core.components.whisper.whisper_alignment_refinement._check_vocal_activity_in_range",
        return_value=0.8,
    ):
        filled_words, filled_segs = wa._fill_vocal_activity_gaps(
            words, af, min_gap=1.0, chunk_duration=0.5, segments=[]
        )

        # Gap before first word: 5.0 to 10.0 (5s) -> should have [VOCAL] words
        # Gap between: 11.0 to 20.0 (9s) -> should have [VOCAL] words
        # Gap after: 21.0 to 25.0 (4s) -> should have [VOCAL] words

        texts = [w.text for w in filled_words]
        assert "[VOCAL]" in texts
        assert "hello" in texts
        assert "world" in texts
        assert len(filled_segs) > 0
