import pytest

from y2karaoke.core.components.whisper import whisper_mapping as wm
from y2karaoke.core.models import Line, Word
from y2karaoke.core.components.alignment.timing_models import (
    TranscriptionSegment,
    TranscriptionWord,
)


def test_pull_late_lines_to_matching_segments_moves_late_line_earlier() -> None:
    lines = [
        Line(words=[Word(text="prev", start_time=74.0, end_time=74.5)]),
        Line(
            words=[
                Word(text="my", start_time=79.54, end_time=79.8),
                Word(text="father", start_time=79.8, end_time=80.2),
                Word(text="was", start_time=80.2, end_time=80.4),
                Word(text="a", start_time=80.4, end_time=80.5),
                Word(text="lord", start_time=80.5, end_time=81.0),
                Word(text="of", start_time=81.0, end_time=81.2),
                Word(text="land", start_time=81.2, end_time=82.48),
            ]
        ),
        Line(words=[Word(text="next", start_time=82.72, end_time=83.1)]),
    ]
    segments = [
        TranscriptionSegment(
            start=78.04,
            end=85.2,
            text="my father was a lord of land my daddy was a repo man",
            words=[],
        )
    ]

    adjusted = wm._pull_late_lines_to_matching_segments(
        lines, segments, language="eng-Latn"
    )

    assert adjusted[1].start_time < lines[1].start_time
    assert adjusted[1].start_time == 78.04
    assert adjusted[1].end_time < adjusted[2].start_time


def test_pull_late_lines_to_matching_segments_keeps_small_shift_unchanged() -> None:
    lines = [
        Line(words=[Word(text="my", start_time=78.5, end_time=79.0)]),
        Line(words=[Word(text="next", start_time=80.0, end_time=80.4)]),
    ]
    segments = [
        TranscriptionSegment(
            start=78.04,
            end=79.5,
            text="my",
            words=[],
        )
    ]

    adjusted = wm._pull_late_lines_to_matching_segments(
        lines, segments, language="eng-Latn"
    )

    assert adjusted[0].start_time == lines[0].start_time


def test_pull_late_lines_to_matching_segments_skips_repeated_phrase_handoff() -> None:
    lines = [
        Line(
            words=[
                Word(text="mother", start_time=68.8, end_time=69.4),
                Word(text="shelter", start_time=69.4, end_time=70.0),
            ]
        ),
        Line(
            words=[
                Word(text="mother", start_time=72.2, end_time=72.9),
                Word(text="shelter", start_time=72.9, end_time=73.6),
                Word(text="us", start_time=73.6, end_time=73.9),
            ]
        ),
    ]
    segments = [
        TranscriptionSegment(
            start=70.9,
            end=73.73,
            text="mother shelter",
            words=[],
        )
    ]

    adjusted = wm._pull_late_lines_to_matching_segments(
        lines, segments, language="eng-Latn"
    )

    assert adjusted[1].start_time == lines[1].start_time
    assert adjusted[1].end_time == lines[1].end_time


def test_pull_late_lines_to_matching_segments_allows_strong_containment_late_pull() -> (
    None
):
    lines = [
        Line(
            words=[
                Word(text="Hold", start_time=163.38, end_time=164.26),
                Word(text="her", start_time=164.26, end_time=164.44),
                Word(text="like", start_time=164.44, end_time=164.70),
                Word(text="a", start_time=164.70, end_time=164.90),
                Word(text="sword", start_time=164.90, end_time=165.18),
                Word(text="and", start_time=165.18, end_time=165.48),
                Word(text="shield", start_time=165.48, end_time=165.84),
            ]
        ),
        Line(
            words=[
                Word(text="Up", start_time=168.50, end_time=169.14),
                Word(text="against", start_time=169.14, end_time=169.50),
                Word(text="this", start_time=169.50, end_time=169.82),
                Word(text="lonely", start_time=169.82, end_time=170.24),
                Word(text="world", start_time=170.24, end_time=171.00),
            ]
        ),
    ]
    segments = [
        TranscriptionSegment(
            start=163.38,
            end=168.50,
            text="Hold her like a sword and shield Up against this lonely world",
            words=[],
        ),
    ]

    adjusted = wm._pull_late_lines_to_matching_segments(
        lines,
        segments,
        language="eng-Latn",
    )

    assert adjusted[1].start_time < lines[1].start_time
    assert adjusted[1].start_time == pytest.approx(165.89, abs=1e-3)


def test_pull_late_lines_prefers_containment_over_higher_overlap() -> None:
    lines = [
        Line(words=[Word(text="prev", start_time=160.0, end_time=162.0)]),
        Line(
            words=[
                Word(text="Up", start_time=168.50, end_time=169.14),
                Word(text="against", start_time=169.14, end_time=169.50),
                Word(text="this", start_time=169.50, end_time=169.82),
                Word(text="lonely", start_time=169.82, end_time=170.24),
                Word(text="world", start_time=170.24, end_time=171.00),
            ]
        ),
    ]
    segments = [
        TranscriptionSegment(
            start=163.38,
            end=168.50,
            text="hold her like a sword and shield up against this lonely world",
            words=[],
        ),
        TranscriptionSegment(
            start=168.50,
            end=173.18,
            text="up against the world its a lonely lonely world",
            words=[],
        ),
    ]

    adjusted = wm._pull_late_lines_to_matching_segments(
        lines,
        segments,
        language="eng-Latn",
    )

    assert adjusted[1].start_time < 168.0


def test_retime_short_interjection_lines_moves_to_matching_segment() -> None:
    lines = [
        Line(words=[Word(text="prev", start_time=71.26, end_time=72.8)]),
        Line(words=[Word(text="Oooh", start_time=73.78, end_time=74.58)]),
        Line(words=[Word(text="next", start_time=79.54, end_time=80.0)]),
    ]
    segments = [
        TranscriptionSegment(
            start=73.78, end=76.14, text="Mother, shelter us", words=[]
        ),
        TranscriptionSegment(start=77.46, end=78.26, text="Ooh", words=[]),
    ]

    adjusted = wm._retime_short_interjection_lines(lines, segments)

    assert adjusted[1].start_time == 77.46
    assert adjusted[1].end_time <= lines[2].start_time


def test_resolve_line_overlaps_skips_empty_lines_between_neighbors() -> None:
    lines = [
        Line(words=[Word(text="a", start_time=10.0, end_time=12.0)]),
        Line(words=[]),
        Line(words=[Word(text="b", start_time=11.5, end_time=13.0)]),
    ]

    adjusted = wm._resolve_line_overlaps(lines)

    assert adjusted[0].end_time <= adjusted[2].start_time


def test_snap_first_word_to_whisper_onset_shifts_late_line_forward() -> None:
    lines = [
        Line(words=[Word(text="prev", start_time=10.0, end_time=10.8)]),
        Line(
            words=[
                Word(text="Father", start_time=12.0, end_time=12.4),
                Word(text="shelter", start_time=12.4, end_time=12.9),
            ]
        ),
        Line(words=[Word(text="next", start_time=14.0, end_time=14.5)]),
    ]
    whisper_words = [
        TranscriptionWord(start=12.35, end=12.7, text="shelter", probability=0.9),
        TranscriptionWord(start=12.55, end=12.9, text="Father,", probability=0.9),
    ]

    adjusted = wm._snap_first_word_to_whisper_onset(lines, whisper_words)

    assert adjusted[1].start_time > lines[1].start_time
    assert adjusted[1].start_time == 12.55
    assert adjusted[1].end_time < lines[2].start_time


def test_snap_first_word_to_whisper_onset_respects_neighbors() -> None:
    lines = [
        Line(words=[Word(text="prev", start_time=10.0, end_time=11.95)]),
        Line(words=[Word(text="Father", start_time=12.0, end_time=12.4)]),
        Line(words=[Word(text="next", start_time=12.48, end_time=12.9)]),
    ]
    whisper_words = [
        TranscriptionWord(start=12.6, end=12.9, text="Father", probability=0.9),
    ]

    adjusted = wm._snap_first_word_to_whisper_onset(lines, whisper_words)

    assert adjusted[1].start_time == lines[1].start_time
    assert adjusted[1].end_time == lines[1].end_time


def test_extend_line_to_trailing_whisper_matches_extends_line_tail() -> None:
    lines = [
        Line(
            words=[
                Word(text="Mother", start_time=71.26, end_time=72.03),
                Word(text="shelter", start_time=72.03, end_time=72.72),
                Word(text="us", start_time=72.72, end_time=72.8),
            ]
        ),
        Line(words=[Word(text="Oooh", start_time=77.46, end_time=78.18)]),
    ]
    whisper_words = [
        TranscriptionWord(start=71.26, end=72.06, text="Mother,", probability=0.8),
        TranscriptionWord(start=72.10, end=72.80, text="shelter", probability=1.0),
        TranscriptionWord(start=75.14, end=75.56, text="shelter", probability=0.9),
        TranscriptionWord(start=75.56, end=76.06, text="us", probability=0.8),
    ]

    adjusted = wm._extend_line_to_trailing_whisper_matches(lines, whisper_words)

    assert adjusted[0].end_time >= 76.0
    assert adjusted[0].end_time < lines[1].start_time
