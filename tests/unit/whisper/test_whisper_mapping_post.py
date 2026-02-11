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


def test_pull_late_lines_pushes_early_refrain_line_toward_segment_start() -> None:
    lines = [
        Line(words=[Word(text="Où", start_time=10.0, end_time=10.4)]),
        Line(
            words=[
                Word(text="Où", start_time=11.0, end_time=11.5),
                Word(text="papa", start_time=11.5, end_time=12.0),
            ]
        ),
        Line(
            words=[
                Word(text="Où", start_time=12.3, end_time=12.8),
                Word(text="papa", start_time=12.8, end_time=13.3),
            ]
        ),
    ]
    segments = [
        TranscriptionSegment(
            start=12.0,
            end=13.0,
            text="où est papa où est-il",
            words=[],
        )
    ]

    adjusted = wm._pull_late_lines_to_matching_segments(
        lines, segments, language="fra-Latn"
    )

    assert adjusted[1].start_time > lines[1].start_time
    assert adjusted[1].start_time < adjusted[2].start_time


def test_pull_late_lines_allows_short_continuation_inside_same_segment() -> None:
    lines = [
        Line(
            words=[
                Word(text="Ça", start_time=52.5, end_time=53.0),
                Word(text="doit", start_time=53.0, end_time=53.4),
                Word(text="faire", start_time=53.4, end_time=53.8),
                Word(text="que", start_time=53.8, end_time=54.1),
                Word(text="j'ai", start_time=54.1, end_time=55.2),
            ]
        ),
        Line(
            words=[
                Word(text="Compté", start_time=58.0, end_time=58.4),
                Word(text="mes", start_time=58.4, end_time=58.8),
                Word(text="doigts", start_time=58.8, end_time=59.1),
                Word(text="hé", start_time=59.1, end_time=59.3),
            ]
        ),
    ]
    segments = [
        TranscriptionSegment(
            start=51.7,
            end=56.2,
            text="ça doit faire au moins mille fois que j'ai compté mes doigts",
            words=[],
        )
    ]

    adjusted = wm._pull_late_lines_to_matching_segments(
        lines, segments, language="fra-Latn"
    )

    assert adjusted[1].start_time < lines[1].start_time
    assert adjusted[1].start_time >= adjusted[0].end_time


def test_pull_late_lines_compresses_early_repetitive_line_into_segment_window() -> None:
    lines = [
        Line(
            words=[
                Word(text="où", start_time=63.6, end_time=64.1),
                Word(text="t'es", start_time=64.1, end_time=64.6),
                Word(text="papaoutai", start_time=64.6, end_time=65.1),
            ]
        ),
        Line(
            words=[
                Word(text="où", start_time=65.7, end_time=66.2),
                Word(text="t'es", start_time=66.2, end_time=66.7),
                Word(text="où", start_time=66.7, end_time=67.2),
                Word(text="t'es", start_time=67.2, end_time=67.7),
                Word(text="où", start_time=67.7, end_time=68.2),
                Word(text="papaoutai", start_time=68.2, end_time=68.7),
            ]
        ),
        Line(
            words=[
                Word(text="où", start_time=69.2, end_time=69.6),
                Word(text="t'es", start_time=69.6, end_time=70.0),
                Word(text="papaoutai", start_time=70.0, end_time=70.4),
            ]
        ),
    ]
    segments = [
        TranscriptionSegment(
            start=67.16,
            end=68.90,
            text="où est papa, où est-il",
            words=[],
        )
    ]

    adjusted = wm._pull_late_lines_to_matching_segments(
        lines, segments, language="fra-Latn"
    )

    assert adjusted[1].start_time < lines[1].start_time
    assert adjusted[1].end_time <= segments[0].end


def test_shift_repeated_lines_fallback_pulls_adjacent_duplicate_closer() -> None:
    lines = [
        Line(
            words=[
                Word(text="Où", start_time=57.5, end_time=58.0),
                Word(text="t'es", start_time=58.0, end_time=58.6),
                Word(text="papaoutai", start_time=58.6, end_time=59.2),
            ]
        ),
        Line(
            words=[
                Word(text="Où", start_time=61.7, end_time=62.2),
                Word(text="t'es", start_time=62.2, end_time=62.8),
                Word(text="papaoutai", start_time=62.8, end_time=63.2),
            ]
        ),
        Line(words=[Word(text="next", start_time=65.7, end_time=66.2)]),
    ]
    whisper_words = [
        TranscriptionWord(start=57.6, end=58.0, text="où", probability=0.9),
        TranscriptionWord(start=58.0, end=58.6, text="est", probability=0.9),
        TranscriptionWord(start=58.6, end=59.2, text="il", probability=0.9),
        TranscriptionWord(start=60.2, end=60.9, text="[VOCAL]", probability=1.0),
    ]

    adjusted = wm._shift_repeated_lines_to_next_whisper(lines, whisper_words)

    assert adjusted[1].start_time < lines[1].start_time
    assert adjusted[1].start_time > adjusted[0].end_time


def test_shift_repeated_lines_ignores_far_future_anchor_for_duplicates() -> None:
    lines = [
        Line(words=[Word(text="Où", start_time=57.5, end_time=58.0)]),
        Line(words=[Word(text="Où", start_time=61.7, end_time=62.1)]),
    ]
    whisper_words = [
        TranscriptionWord(start=57.6, end=58.0, text="où", probability=0.9),
        TranscriptionWord(start=66.9, end=67.2, text="où", probability=0.9),
    ]

    adjusted = wm._shift_repeated_lines_to_next_whisper(lines, whisper_words)

    # Should use cadence fallback near previous line, not the far-future anchor.
    assert adjusted[1].start_time < 61.0


def test_pull_late_lines_smooths_adjacent_duplicate_cadence() -> None:
    lines = [
        Line(
            words=[
                Word(text="Où", start_time=57.5, end_time=58.0),
                Word(text="t'es", start_time=58.0, end_time=58.6),
                Word(text="papaoutai", start_time=58.6, end_time=59.2),
            ]
        ),
        Line(
            words=[
                Word(text="Où", start_time=61.7, end_time=62.2),
                Word(text="t'es", start_time=62.2, end_time=62.8),
                Word(text="papaoutai", start_time=62.8, end_time=63.2),
            ]
        ),
        Line(words=[Word(text="next", start_time=65.7, end_time=66.2)]),
    ]
    segments = [
        TranscriptionSegment(
            start=57.6, end=61.9, text="où est papa, où est-il", words=[]
        )
    ]

    adjusted = wm._pull_late_lines_to_matching_segments(
        lines, segments, language="fra-Latn"
    )

    assert adjusted[1].start_time < lines[1].start_time
    assert adjusted[1].start_time > adjusted[0].end_time


def test_pull_late_lines_smooths_similar_refrain_cadence() -> None:
    lines = [
        Line(
            words=[
                Word(text="Où", start_time=61.2, end_time=61.7),
                Word(text="t'es", start_time=61.7, end_time=62.2),
                Word(text="papaoutai", start_time=62.2, end_time=62.7),
            ]
        ),
        Line(
            words=[
                Word(text="Où", start_time=65.7, end_time=66.2),
                Word(text="t'es", start_time=66.2, end_time=66.7),
                Word(text="où", start_time=66.7, end_time=67.2),
                Word(text="t'es", start_time=67.2, end_time=67.7),
                Word(text="où", start_time=67.7, end_time=68.2),
                Word(text="papaoutai", start_time=68.2, end_time=68.7),
            ]
        ),
        Line(words=[Word(text="next", start_time=69.2, end_time=69.7)]),
    ]
    segments = [
        TranscriptionSegment(start=67.16, end=68.9, text="random bridge", words=[])
    ]

    adjusted = wm._pull_late_lines_to_matching_segments(
        lines, segments, language="fra-Latn"
    )

    assert adjusted[1].start_time < lines[1].start_time
    assert adjusted[1].start_time <= adjusted[0].end_time + 0.55
    assert adjusted[1].end_time - adjusted[1].start_time <= 1.71


def test_pull_late_lines_rebalances_collapsed_short_question_pair() -> None:
    lines = [
        Line(
            words=[
                Word(text="Serons-nous", start_time=104.36, end_time=104.56),
                Word(text="détestables?", start_time=104.56, end_time=104.76),
            ]
        ),
        Line(
            words=[
                Word(text="Serons-nous", start_time=104.76, end_time=106.20),
                Word(text="admirables?", start_time=106.20, end_time=108.58),
            ]
        ),
        Line(words=[Word(text="Des", start_time=108.62, end_time=110.46)]),
    ]
    segments = [TranscriptionSegment(start=104.32, end=108.64, text="x", words=[])]

    adjusted = wm._pull_late_lines_to_matching_segments(
        lines, segments, language="fra-Latn"
    )

    assert adjusted[0].end_time - adjusted[0].start_time >= 1.4
    assert adjusted[1].end_time - adjusted[1].start_time <= 2.21
    assert adjusted[0].end_time <= adjusted[1].start_time


def test_pull_late_lines_ignores_short_interjection_segment_for_full_line() -> None:
    lines = [
        Line(words=[Word(text="Bouffé", start_time=129.68, end_time=131.34)]),
        Line(
            words=[
                Word(text="Où", start_time=134.80, end_time=135.20),
                Word(text="t'es,", start_time=135.20, end_time=135.60),
                Word(text="papaoutai?", start_time=135.60, end_time=136.00),
            ]
        ),
        Line(words=[Word(text="next", start_time=136.20, end_time=136.60)]),
    ]
    segments = [
        TranscriptionSegment(start=132.02, end=133.20, text="Hé !", words=[]),
        TranscriptionSegment(
            start=133.20, end=135.62, text="Où est papa, où est-il ?", words=[]
        ),
    ]

    adjusted = wm._pull_late_lines_to_matching_segments(
        lines, segments, language="fra-Latn"
    )

    assert adjusted[1].start_time < lines[1].start_time
    assert adjusted[1].start_time >= 133.15
    assert adjusted[1].start_time <= 133.30


def test_pull_late_lines_realigns_early_repetitive_run_to_segments() -> None:
    lines = [
        Line(words=[Word(text="Bouffé", start_time=129.68, end_time=131.34)]),
        Line(
            words=[
                Word(text="Où", start_time=131.34, end_time=132.40),
                Word(text="t'es,", start_time=132.40, end_time=132.80),
            ]
        ),
        Line(
            words=[
                Word(text="Où", start_time=133.30, end_time=134.46),
                Word(text="t'es,", start_time=134.46, end_time=134.86),
            ]
        ),
        Line(
            words=[
                Word(text="Où", start_time=134.86, end_time=136.00),
                Word(text="t'es,", start_time=136.00, end_time=136.30),
            ]
        ),
    ]
    segments = [
        TranscriptionSegment(start=132.02, end=133.20, text="Hé !", words=[]),
        TranscriptionSegment(
            start=133.20, end=135.62, text="Où est papa, où est-il ?", words=[]
        ),
        TranscriptionSegment(
            start=135.62, end=138.14, text="Où est papa, où est-il ?", words=[]
        ),
        TranscriptionSegment(
            start=138.14, end=140.08, text="Où est papa, où est-il ?", words=[]
        ),
    ]

    adjusted = wm._pull_late_lines_to_matching_segments(
        lines, segments, language="fra-Latn"
    )

    assert adjusted[1].start_time >= 132.00
    assert adjusted[2].start_time >= 134.00
    assert adjusted[3].start_time >= 135.00


def test_pull_late_lines_repetitive_phrase_advances_to_next_segment() -> None:
    lines = [
        Line(words=[Word(text="Où?", start_time=132.97, end_time=134.03)]),
        Line(words=[Word(text="Où?", start_time=134.14, end_time=135.30)]),
        Line(words=[Word(text="next", start_time=136.40, end_time=137.20)]),
    ]
    segments = [
        TranscriptionSegment(
            start=133.20, end=135.62, text="Où est papa, où est-il ?", words=[]
        ),
        TranscriptionSegment(
            start=135.62, end=138.14, text="Où est papa, où est-il ?", words=[]
        ),
    ]

    adjusted = wm._pull_late_lines_to_matching_segments(
        lines, segments, language="fra-Latn"
    )

    assert adjusted[1].start_time >= 135.60


def test_pull_late_lines_pulls_similar_line_forward_across_long_gap() -> None:
    lines = [
        Line(
            words=[
                Word(text="Où", start_time=153.46, end_time=154.20),
                Word(text="est", start_time=154.20, end_time=154.70),
                Word(text="ton", start_time=154.70, end_time=155.00),
                Word(text="papa?", start_time=155.00, end_time=155.46),
            ]
        ),
        Line(
            words=[
                Word(text="Dis-moi,", start_time=172.32, end_time=173.10),
                Word(text="où", start_time=173.10, end_time=173.45),
                Word(text="est", start_time=173.45, end_time=173.80),
                Word(text="ton", start_time=173.80, end_time=174.10),
                Word(text="papa?", start_time=174.10, end_time=174.47),
            ]
        ),
    ]
    segments = [TranscriptionSegment(start=170.0, end=176.0, text="x", words=[])]

    adjusted = wm._pull_late_lines_to_matching_segments(
        lines, segments, language="fra-Latn"
    )

    assert adjusted[0].start_time > lines[0].start_time
    assert adjusted[0].end_time <= adjusted[1].start_time


def test_pull_late_lines_ignores_weak_short_line_match_in_repeated_segment() -> None:
    lines = [
        Line(
            words=[
                Word(text="Il", start_time=197.34, end_time=197.80),
                Word(text="sait", start_time=197.80, end_time=198.20),
                Word(text="ce", start_time=198.20, end_time=198.60),
                Word(text="qui", start_time=198.60, end_time=198.95),
                Word(text="ne", start_time=198.95, end_time=199.10),
                Word(text="va", start_time=199.10, end_time=199.24),
                Word(text="pas", start_time=199.24, end_time=199.50),
            ]
        ),
        Line(
            words=[
                Word(text="Ah,", start_time=201.00, end_time=201.25),
                Word(text="sacré", start_time=201.25, end_time=201.55),
                Word(text="papa", start_time=201.55, end_time=201.95),
            ]
        ),
        Line(
            words=[
                Word(text="Dis-moi,", start_time=202.90, end_time=203.45),
                Word(text="où", start_time=203.45, end_time=203.75),
                Word(text="es-tu", start_time=203.75, end_time=204.20),
                Word(text="caché?", start_time=204.20, end_time=204.70),
            ]
        ),
    ]
    segments = [
        TranscriptionSegment(
            start=199.40,
            end=204.22,
            text="Où est papa, où est-il ?",
            words=[],
        )
    ]

    adjusted = wm._pull_late_lines_to_matching_segments(
        lines, segments, language="fra-Latn"
    )

    assert adjusted[1].start_time >= lines[1].start_time - 0.01
