import pytest

from y2karaoke.core.components.alignment.timing_models import TranscriptionWord
from y2karaoke.core.components.whisper import whisper_integration_align as wialign
from y2karaoke.core.models import Line, Word


def test_restore_weak_evidence_large_start_shifts_restores_to_baseline():
    mapped = [
        Line(words=[Word(text="alpha", start_time=20.0, end_time=21.0)]),
        Line(words=[Word(text="beta", start_time=30.0, end_time=31.0)]),
    ]
    baseline = [
        Line(words=[Word(text="alpha", start_time=10.0, end_time=11.0)]),
        Line(words=[Word(text="beta", start_time=30.0, end_time=31.0)]),
    ]
    whisper_words = [
        TranscriptionWord(text="[VOCAL]", start=20.1, end=20.4, probability=0.9),
        TranscriptionWord(text="[VOCAL]", start=20.6, end=20.9, probability=0.9),
        TranscriptionWord(text="beta", start=30.0, end=30.5, probability=0.9),
    ]

    repaired, restored = wialign._restore_weak_evidence_large_start_shifts(
        mapped,
        baseline,
        whisper_words,
        min_shift_sec=1.1,
        min_support_words=2,
        support_window_sec=1.0,
    )

    assert restored == 1
    assert repaired[0].start_time == pytest.approx(10.0)
    assert repaired[1].start_time == pytest.approx(30.0)


def test_restore_weak_evidence_large_start_shifts_keeps_supported_shift():
    mapped = [
        Line(words=[Word(text="alpha", start_time=20.0, end_time=21.0)]),
    ]
    baseline = [
        Line(words=[Word(text="alpha", start_time=10.0, end_time=11.0)]),
    ]
    whisper_words = [
        TranscriptionWord(text="i", start=19.4, end=19.5, probability=0.8),
        TranscriptionWord(text="hear", start=19.8, end=20.0, probability=0.8),
        TranscriptionWord(text="words", start=20.2, end=20.4, probability=0.8),
        TranscriptionWord(text="[VOCAL]", start=20.6, end=20.9, probability=0.8),
    ]

    repaired, restored = wialign._restore_weak_evidence_large_start_shifts(
        mapped,
        baseline,
        whisper_words,
        min_shift_sec=1.1,
        min_support_words=3,
        support_window_sec=1.0,
    )

    assert restored == 0
    assert repaired[0].start_time == pytest.approx(20.0)
