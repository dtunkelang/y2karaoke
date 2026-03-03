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
