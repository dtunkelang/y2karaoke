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


def test_resolve_line_overlaps_rebalances_stacked_words_within_line() -> None:
    lines = [
        Line(
            words=[
                Word(text="Young", start_time=33.18, end_time=34.04),
                Word(text="man", start_time=33.98, end_time=34.04),
                Word(text="in", start_time=33.98, end_time=34.04),
                Word(text="America", start_time=33.98, end_time=34.04),
            ]
        ),
        Line(words=[Word(text="next", start_time=34.20, end_time=34.80)]),
    ]

    adjusted = wm._resolve_line_overlaps(lines)

    assert adjusted[0].start_time == lines[0].start_time
    assert adjusted[0].end_time == lines[0].end_time
    starts = [w.start_time for w in adjusted[0].words]
    assert all(starts[i] < starts[i + 1] for i in range(len(starts) - 1))


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


def test_snap_first_word_to_whisper_onset_shifts_repetitive_run_when_blocked() -> None:
    lines = [
        Line(words=[Word(text="prev", start_time=129.68, end_time=131.34)]),
        Line(words=[Word(text="Où", start_time=131.34, end_time=132.40)]),
        Line(words=[Word(text="Où", start_time=133.30, end_time=134.46)]),
        Line(words=[Word(text="Où", start_time=134.51, end_time=136.34)]),
        Line(words=[Word(text="next", start_time=172.32, end_time=174.47)]),
    ]
    whisper_words = [
        TranscriptionWord(start=133.20, end=133.60, text="Où", probability=0.9),
    ]

    adjusted = wm._snap_first_word_to_whisper_onset(lines, whisper_words)

    assert adjusted[1].start_time > lines[1].start_time
    assert adjusted[2].start_time > lines[2].start_time
    assert adjusted[3].start_time > lines[3].start_time


def test_snap_first_word_to_whisper_onset_prefers_run_shift_over_partial_single() -> (
    None
):
    lines = [
        Line(words=[Word(text="prev", start_time=129.68, end_time=131.34)]),
        Line(words=[Word(text="Où", start_time=131.34, end_time=132.40)]),
        Line(words=[Word(text="Où", start_time=132.45, end_time=133.51)]),
        Line(words=[Word(text="Où", start_time=133.56, end_time=134.62)]),
        Line(words=[Word(text="next", start_time=170.0, end_time=171.0)]),
    ]
    whisper_words = [
        TranscriptionWord(start=133.20, end=133.60, text="Où", probability=0.9),
    ]

    adjusted = wm._snap_first_word_to_whisper_onset(lines, whisper_words)

    # Single-line shift would cap near 132.40 due to immediate next line.
    # Run shift should allow landing near Whisper onset.
    assert adjusted[1].start_time >= 133.0
    assert adjusted[2].start_time >= 134.0


def test_snap_first_word_to_whisper_onset_shifts_packed_suffix_when_blocked() -> None:
    lines = [
        Line(words=[Word(text="a", start_time=1.0, end_time=1.5)]),
        Line(words=[Word(text="b", start_time=2.0, end_time=2.5)]),
        Line(words=[Word(text="c", start_time=3.0, end_time=3.5)]),
        Line(words=[Word(text="Dis-moi,", start_time=10.0, end_time=11.0)]),
        Line(words=[Word(text="où", start_time=11.0, end_time=12.0)]),
        Line(words=[Word(text="es-tu", start_time=12.0, end_time=13.0)]),
    ]
    whisper_words = [
        TranscriptionWord(start=12.4, end=12.8, text="dis", probability=0.9),
    ]

    adjusted = wm._snap_first_word_to_whisper_onset(lines, whisper_words, max_shift=2.5)

    assert adjusted[3].start_time >= 12.39
    assert adjusted[4].start_time > lines[4].start_time
    assert adjusted[5].start_time > lines[5].start_time


def test_snap_first_word_suffix_shift_does_not_trigger_for_question_line() -> None:
    lines = [
        Line(words=[Word(text="a", start_time=1.0, end_time=1.5)]),
        Line(words=[Word(text="b", start_time=2.0, end_time=2.5)]),
        Line(words=[Word(text="c", start_time=3.0, end_time=3.5)]),
        Line(
            words=[
                Word(text="Où", start_time=10.0, end_time=10.7),
                Word(text="est", start_time=10.7, end_time=11.4),
                Word(text="papa?", start_time=11.4, end_time=12.0),
            ]
        ),
        Line(words=[Word(text="x", start_time=12.0, end_time=12.6)]),
        Line(words=[Word(text="y", start_time=12.6, end_time=13.2)]),
    ]
    whisper_words = [
        TranscriptionWord(start=12.2, end=12.5, text="où", probability=0.9),
    ]

    adjusted = wm._snap_first_word_to_whisper_onset(lines, whisper_words, max_shift=2.5)

    assert adjusted[3].start_time == lines[3].start_time
    assert adjusted[4].start_time == lines[4].start_time
    assert adjusted[5].start_time == lines[5].start_time


def test_pull_late_lines_handles_code_switch_with_accented_spanish() -> None:
    lines = [
        Line(words=[Word(text="prev", start_time=46.0, end_time=46.4)]),
        Line(
            words=[
                Word(text="Dímelo", start_time=50.5, end_time=50.9),
                Word(text="baby", start_time=50.9, end_time=51.3),
            ]
        ),
        Line(words=[Word(text="next", start_time=52.2, end_time=52.6)]),
    ]
    segments = [
        TranscriptionSegment(
            start=48.0,
            end=51.8,
            text="dimelo baby",
            words=[],
        )
    ]

    adjusted = wm._pull_late_lines_to_matching_segments(
        lines, segments, language="spa-Latn"
    )

    assert adjusted[1].start_time == 48.0
