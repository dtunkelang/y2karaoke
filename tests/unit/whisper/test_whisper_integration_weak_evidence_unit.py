from y2karaoke.core.models import Line, Word
from y2karaoke.core.components.alignment.timing_models import TranscriptionWord
from y2karaoke.core.components.whisper.whisper_integration_weak_evidence import (
    restore_adjacent_near_threshold_late_shifts,
    restore_unsupported_early_duplicate_shifts,
    restore_weak_evidence_large_start_shifts,
)


def _line(start: float) -> Line:
    return Line(words=[Word(text="x", start_time=start, end_time=start + 0.4)])


def _long_line(start: float, end: float) -> Line:
    return Line(
        words=[
            Word(text="x", start_time=start, end_time=(start + end) / 2),
            Word(text="y", start_time=(start + end) / 2, end_time=end),
        ]
    )


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
    mapped = [
        Line(words=[Word(text="alpha", start_time=10.0, end_time=10.4)]),
    ]
    baseline = [
        Line(words=[Word(text="alpha", start_time=8.0, end_time=8.4)]),
    ]
    whisper_words = [
        TranscriptionWord(text="alpha", start=9.7, end=9.8, probability=1.0),
        TranscriptionWord(text="beta", start=10.0, end=10.1, probability=1.0),
        TranscriptionWord(text="gamma", start=10.2, end=10.3, probability=1.0),
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


def test_restore_weak_evidence_large_start_shifts_uses_first_substantive_token():
    mapped = [
        Line(
            words=[
                Word(text="I", start_time=12.8, end_time=13.0),
                Word(text="said", start_time=13.0, end_time=13.2),
                Word(text="ooh", start_time=13.2, end_time=13.4),
            ]
        )
    ]
    baseline = [
        Line(
            words=[
                Word(text="I", start_time=8.0, end_time=8.2),
                Word(text="said", start_time=8.2, end_time=8.4),
                Word(text="ooh", start_time=8.4, end_time=8.6),
            ]
        )
    ]
    whisper_words = [
        TranscriptionWord(text="drowning", start=12.4, end=12.8, probability=0.99),
        TranscriptionWord(text="in", start=12.8, end=13.0, probability=0.99),
        TranscriptionWord(text="the", start=13.0, end=13.2, probability=0.99),
    ]

    repaired, restored = restore_weak_evidence_large_start_shifts(
        mapped,
        baseline,
        whisper_words,
    )

    assert restored == 1
    assert repaired[0].start_time == 8.0


def test_restore_weak_evidence_large_start_shifts_restores_unsupported_early_shift():
    mapped = [
        Line(
            words=[
                Word(text="No", start_time=6.0, end_time=6.2),
                Word(text="sleep", start_time=6.2, end_time=6.7),
            ]
        ),
        _line(12.0),
    ]
    baseline = [
        Line(
            words=[
                Word(text="No", start_time=10.0, end_time=10.2),
                Word(text="sleep", start_time=10.2, end_time=10.7),
            ]
        ),
        _line(12.0),
    ]
    whisper_words = [
        TranscriptionWord(text="oh", start=6.4, end=6.6, probability=0.02),
        TranscriptionWord(text="baby", start=7.4, end=7.8, probability=0.05),
        TranscriptionWord(text="beta", start=12.0, end=12.4, probability=0.9),
    ]

    repaired, restored = restore_weak_evidence_large_start_shifts(
        mapped, baseline, whisper_words
    )

    assert restored == 1
    assert repaired[0].start_time == 10.0


def test_restore_weak_evidence_large_start_shifts_keeps_supported_early_shift():
    mapped = [
        Line(
            words=[
                Word(text="No", start_time=6.0, end_time=6.2),
                Word(text="sleep", start_time=6.2, end_time=6.7),
            ]
        )
    ]
    baseline = [
        Line(
            words=[
                Word(text="No", start_time=10.0, end_time=10.2),
                Word(text="sleep", start_time=10.2, end_time=10.7),
            ]
        )
    ]
    whisper_words = [
        TranscriptionWord(text="No", start=5.9, end=6.1, probability=0.99),
        TranscriptionWord(text="sleep", start=6.18, end=6.55, probability=0.99),
        TranscriptionWord(text="touch", start=6.6, end=6.9, probability=0.99),
    ]

    repaired, restored = restore_weak_evidence_large_start_shifts(
        mapped, baseline, whisper_words
    )

    assert restored == 0
    assert repaired[0].start_time == 6.0


def test_restore_weak_evidence_large_start_shifts_keeps_single_word_early_pull():
    mapped = [
        Line(words=[Word(text="I'm", start_time=98.0, end_time=99.0)]),
    ]
    baseline = [
        Line(words=[Word(text="I'm", start_time=120.0, end_time=121.0)]),
    ]
    whisper_words = [
        TranscriptionWord(text="im", start=120.2, end=120.8, probability=0.9),
    ]

    repaired, restored = restore_weak_evidence_large_start_shifts(
        mapped, baseline, whisper_words
    )

    assert restored == 0
    assert repaired[0].start_time == 98.0


def test_restore_adjacent_near_threshold_late_shifts_restores_sandwiched_run():
    mapped = [
        _line(10.0),
        _long_line(12.0, 13.8),
        _long_line(13.8, 15.6),
        _line(14.6),
    ]
    baseline = [
        _line(10.0),
        _long_line(11.05, 12.9),
        _long_line(12.9, 14.7),
        _line(13.7),
    ]
    whisper_words = [
        TranscriptionWord(text="a", start=11.1, end=11.2, probability=0.99),
        TranscriptionWord(text="b", start=11.5, end=11.6, probability=0.99),
        TranscriptionWord(text="c", start=12.9, end=13.0, probability=0.99),
        TranscriptionWord(text="d", start=13.2, end=13.3, probability=0.99),
    ]

    repaired, restored = restore_adjacent_near_threshold_late_shifts(
        mapped,
        baseline,
        whisper_words,
    )

    assert restored == 2
    assert repaired[1].start_time == 11.05
    assert repaired[2].start_time == 12.9


def test_restore_adjacent_near_threshold_late_shifts_skips_without_baseline_support():
    mapped = [
        _line(10.0),
        _long_line(12.0, 13.8),
        _long_line(13.8, 15.6),
        _line(14.6),
    ]
    baseline = [
        _line(10.0),
        _long_line(11.05, 12.9),
        _long_line(12.9, 14.7),
        _line(13.7),
    ]
    whisper_words = [
        TranscriptionWord(text="late", start=12.1, end=12.2, probability=0.99),
        TranscriptionWord(text="shift", start=12.4, end=12.5, probability=0.99),
    ]

    repaired, restored = restore_adjacent_near_threshold_late_shifts(
        mapped,
        baseline,
        whisper_words,
    )

    assert restored == 0
    assert repaired[1].start_time == 12.0
    assert repaired[2].start_time == 13.8


def test_restore_unsupported_early_duplicate_shifts_restores_unsupported_repeat():
    mapped = [
        Line(words=[Word(text="(Hey,", start_time=160.79, end_time=161.16)]),
        Line(words=[Word(text="(Hey,", start_time=169.91, end_time=170.19)]),
    ]
    baseline = [
        Line(words=[Word(text="(Hey,", start_time=160.79, end_time=161.16)]),
        Line(words=[Word(text="(Hey,", start_time=171.94, end_time=172.22)]),
    ]
    whisper_words = [
        TranscriptionWord(text="[VOCAL]", start=169.9, end=170.0, probability=1.0),
    ]

    repaired, restored = restore_unsupported_early_duplicate_shifts(
        mapped,
        baseline,
        whisper_words,
    )

    assert restored == 1
    assert repaired[1].start_time == 171.94


def test_restore_unsupported_early_duplicate_shifts_keeps_supported_repeat():
    mapped = [
        Line(words=[Word(text="(Hey,", start_time=160.79, end_time=161.16)]),
        Line(words=[Word(text="(Hey,", start_time=169.91, end_time=170.19)]),
    ]
    baseline = [
        Line(words=[Word(text="(Hey,", start_time=160.79, end_time=161.16)]),
        Line(words=[Word(text="(Hey,", start_time=171.94, end_time=172.22)]),
    ]
    whisper_words = [
        TranscriptionWord(text="hey", start=170.0, end=170.1, probability=0.9),
    ]

    repaired, restored = restore_unsupported_early_duplicate_shifts(
        mapped,
        baseline,
        whisper_words,
    )

    assert restored == 0
    assert repaired[1].start_time == 169.91
