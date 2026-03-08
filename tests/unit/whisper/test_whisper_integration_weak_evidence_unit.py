from y2karaoke.core.models import Line, Word
from y2karaoke.core.components.alignment.timing_models import TranscriptionWord
from y2karaoke.core.components.whisper.whisper_integration_weak_evidence import (
    restore_weak_evidence_large_start_shifts,
)


def _line(start: float) -> Line:
    return Line(words=[Word(text="x", start_time=start, end_time=start + 0.4)])


def test_restore_weak_evidence_large_start_shifts_restores_without_support():
    mapped = [_line(10.0)]
    baseline = [_line(8.0)]
    whisper_words = [
        TranscriptionWord(text="[VOCAL]", start=10.0, end=10.1, probability=1.0)
    ]
    repaired, restored = restore_weak_evidence_large_start_shifts(
        mapped, baseline, whisper_words
    )
    assert restored == 1
    assert repaired[0].start_time == 8.0


def test_restore_weak_evidence_large_start_shifts_keeps_supported_shift():
    mapped = [_line(10.0)]
    baseline = [_line(8.0)]
    whisper_words = [
        TranscriptionWord(text="a", start=9.7, end=9.8, probability=1.0),
        TranscriptionWord(text="b", start=10.0, end=10.1, probability=1.0),
        TranscriptionWord(text="c", start=10.2, end=10.3, probability=1.0),
    ]
    repaired, restored = restore_weak_evidence_large_start_shifts(
        mapped, baseline, whisper_words
    )
    assert restored == 0
    assert repaired[0].start_time == 10.0


def test_restore_weak_evidence_large_start_shifts_restores_low_confidence_window():
    mapped = [_line(10.0), _line(12.0)]
    baseline = [_line(8.0), _line(12.0)]
    whisper_words = [
        TranscriptionWord(text="a", start=9.1, end=9.2, probability=0.3),
        TranscriptionWord(text="b", start=9.5, end=9.6, probability=0.4),
        TranscriptionWord(text="c", start=9.9, end=10.0, probability=0.45),
        TranscriptionWord(text="tail", start=12.0, end=12.2, probability=0.9),
    ]
    repaired, restored = restore_weak_evidence_large_start_shifts(
        mapped, baseline, whisper_words
    )
    assert restored == 1
    assert repaired[0].start_time == 8.0


def test_restore_weak_evidence_large_start_shifts_restores_without_lexical_onset_support():
    mapped = [
        Line(
            words=[
                Word(text="So", start_time=10.0, end_time=10.3),
                Word(text="hit", start_time=10.3, end_time=10.8),
            ]
        )
    ]
    baseline = [
        Line(
            words=[
                Word(text="So", start_time=8.0, end_time=8.3),
                Word(text="hit", start_time=8.3, end_time=8.8),
            ]
        )
    ]
    whisper_words = [
        TranscriptionWord(text="want", start=10.0, end=10.1, probability=0.99),
        TranscriptionWord(text="to", start=10.1, end=10.2, probability=0.99),
        TranscriptionWord(text="bless", start=10.2, end=10.3, probability=0.99),
    ]

    repaired, restored = restore_weak_evidence_large_start_shifts(
        mapped, baseline, whisper_words
    )

    assert restored == 1
    assert repaired[0].start_time == 8.0


def test_restore_weak_evidence_large_start_shifts_keeps_lexically_supported_shift():
    mapped = [
        Line(
            words=[
                Word(text="So", start_time=10.0, end_time=10.3),
                Word(text="hit", start_time=10.3, end_time=10.8),
            ]
        )
    ]
    baseline = [
        Line(
            words=[
                Word(text="So", start_time=8.0, end_time=8.3),
                Word(text="hit", start_time=8.3, end_time=8.8),
            ]
        )
    ]
    whisper_words = [
        TranscriptionWord(text="so", start=9.95, end=10.05, probability=0.99),
        TranscriptionWord(text="road", start=10.1, end=10.3, probability=0.99),
        TranscriptionWord(text="hit", start=10.35, end=10.55, probability=0.99),
    ]

    repaired, restored = restore_weak_evidence_large_start_shifts(
        mapped, baseline, whisper_words
    )

    assert restored == 0
    assert repaired[0].start_time == 10.0
