import pytest

from y2karaoke.core.components.alignment.timing_models import TranscriptionWord
from y2karaoke.core.components.whisper.whisper_integration_align_corrections import (
    _extend_final_line_last_word_to_baseline_end,
    _restore_alternating_middle_hook_from_phrase_window,
    _restore_compact_exact_phrase_late_starts,
    _restore_leading_alternating_hook_start_to_baseline,
)
from y2karaoke.core.models import Line, Word


def _line(texts: list[str], start: float, end: float) -> Line:
    spacing = (end - start) / len(texts)
    words = []
    for idx, text in enumerate(texts):
        word_start = start + idx * spacing
        word_end = end if idx == len(texts) - 1 else word_start + spacing * 0.9
        words.append(
            Word(
                text=text,
                start_time=word_start,
                end_time=word_end,
            )
        )
    return Line(words=words)


def test_restore_alternating_middle_hook_from_phrase_window_extends_middle_line_end():
    baseline_lines = [
        _line(["Take", "on", "me"], 1.2, 5.2),
        _line(["Take", "me", "on"], 6.8, 10.6),
        _line(["I'll", "be", "gone"], 12.1, 15.5),
    ]
    mapped_lines = [
        baseline_lines[0],
        _line(["Take", "me", "on"], 6.451, 8.931),
        _line(["I'll", "be", "gone"], 11.912, 15.494),
    ]
    whisper_words = [
        TranscriptionWord(text="take", start=0.64, end=1.2, probability=1.0),
        TranscriptionWord(text="on", start=1.2, end=3.08, probability=1.0),
        TranscriptionWord(text="me", start=3.08, end=4.22, probability=1.0),
        TranscriptionWord(text="take", start=4.22, end=5.48, probability=1.0),
        TranscriptionWord(text="me", start=5.48, end=8.08, probability=1.0),
        TranscriptionWord(text="on", start=8.08, end=9.7, probability=1.0),
        TranscriptionWord(text="i'll", start=12.34, end=13.18, probability=1.0),
        TranscriptionWord(text="be", start=13.18, end=13.84, probability=1.0),
        TranscriptionWord(text="gone", start=13.84, end=15.5, probability=1.0),
    ]

    repaired, restored = _restore_alternating_middle_hook_from_phrase_window(
        mapped_lines,
        baseline_lines,
        whisper_words,
    )

    assert restored == 1
    assert repaired[1].start_time == pytest.approx(6.451)
    assert repaired[1].end_time == pytest.approx(10.124333333333333)
    assert repaired[1].end_time > mapped_lines[1].end_time + 1.1
    assert repaired[2].start_time == pytest.approx(11.912)


def test_alternating_middle_hook_restore_skips_non_alternating_pair():
    baseline_lines = [
        _line(["Take", "on", "me"], 1.2, 5.2),
        _line(["Take", "on", "me"], 6.8, 10.6),
        _line(["I'll", "be", "gone"], 12.1, 15.5),
    ]
    mapped_lines = [
        baseline_lines[0],
        _line(["Take", "on", "me"], 6.451, 8.931),
        _line(["I'll", "be", "gone"], 11.912, 15.494),
    ]
    whisper_words = [
        TranscriptionWord(text="take", start=4.22, end=5.48, probability=1.0),
        TranscriptionWord(text="on", start=5.48, end=8.08, probability=1.0),
        TranscriptionWord(text="me", start=8.08, end=9.7, probability=1.0),
    ]

    repaired, restored = _restore_alternating_middle_hook_from_phrase_window(
        mapped_lines,
        baseline_lines,
        whisper_words,
    )

    assert restored == 0
    assert repaired[1].start_time == mapped_lines[1].start_time
    assert repaired[1].end_time == mapped_lines[1].end_time


def test_restore_compact_exact_phrase_late_starts_reanchors_late_start_only():
    baseline_lines = [
        _line(["Take", "on", "me"], 1.2, 5.2),
        _line(["Take", "me", "on"], 6.8, 10.6),
        _line(["I'll", "be", "gone"], 12.35, 15.5),
        _line(["In", "a", "day", "or", "two"], 17.4, 22.0),
    ]
    mapped_lines = [
        baseline_lines[0],
        _line(["Take", "me", "on"], 6.451, 10.124333333333333),
        _line(["I'll", "be", "gone"], 11.912, 15.494),
        _line(["In", "a", "day", "or", "two"], 17.373, 20.873),
    ]
    whisper_words = [
        TranscriptionWord(text="take", start=0.64, end=1.2, probability=1.0),
        TranscriptionWord(text="on", start=1.2, end=3.08, probability=1.0),
        TranscriptionWord(text="me", start=3.08, end=4.22, probability=1.0),
        TranscriptionWord(text="take", start=4.22, end=5.48, probability=1.0),
        TranscriptionWord(text="me", start=5.48, end=8.08, probability=1.0),
        TranscriptionWord(text="on", start=8.08, end=9.7, probability=1.0),
        TranscriptionWord(text="i'll", start=12.34, end=13.18, probability=1.0),
        TranscriptionWord(text="be", start=13.18, end=13.84, probability=1.0),
        TranscriptionWord(text="gone", start=13.84, end=15.5, probability=1.0),
    ]

    repaired, restored = _restore_compact_exact_phrase_late_starts(
        mapped_lines,
        baseline_lines,
        whisper_words,
    )

    assert restored == 1
    assert repaired[2].start_time == pytest.approx(12.34)
    assert repaired[2].end_time == pytest.approx(15.394666666666668)
    assert repaired[2].start_time > mapped_lines[2].start_time + 0.4
    assert repaired[1].start_time == pytest.approx(mapped_lines[1].start_time)


def test_restore_compact_exact_phrase_late_starts_skips_when_end_would_cross_next():
    baseline_lines = [
        _line(["I'll", "be", "gone"], 12.35, 15.5),
        _line(["In", "a", "day"], 15.55, 17.0),
    ]
    mapped_lines = [
        _line(["I'll", "be", "gone"], 11.912, 15.494),
        _line(["In", "a", "day"], 15.56, 16.8),
    ]
    whisper_words = [
        TranscriptionWord(text="i'll", start=12.34, end=13.18, probability=1.0),
        TranscriptionWord(text="be", start=13.18, end=13.84, probability=1.0),
        TranscriptionWord(text="gone", start=13.84, end=15.54, probability=1.0),
    ]

    repaired, restored = _restore_compact_exact_phrase_late_starts(
        mapped_lines,
        baseline_lines,
        whisper_words,
    )

    assert restored == 0
    assert repaired[0].start_time == pytest.approx(mapped_lines[0].start_time)


def test_extend_final_line_last_word_to_baseline_end_only_extends_tail():
    baseline_lines = [
        _line(["Take", "me", "on"], 6.8, 10.6),
        _line(["I'll", "be", "gone"], 12.35, 15.5),
        _line(["In", "a", "day", "or", "two"], 17.05, 21.42),
    ]
    mapped_lines = [
        _line(["Take", "me", "on"], 6.451, 9.91),
        _line(["I'll", "be", "gone"], 12.34, 15.394666666666668),
        _line(["In", "a", "day", "or", "two"], 17.37, 20.87),
    ]
    whisper_words = [
        TranscriptionWord(text="take", start=4.22, end=5.48, probability=1.0),
        TranscriptionWord(text="me", start=5.48, end=8.08, probability=1.0),
        TranscriptionWord(text="on", start=8.08, end=9.7, probability=1.0),
    ]

    repaired, restored = _extend_final_line_last_word_to_baseline_end(
        mapped_lines,
        baseline_lines,
        whisper_words,
    )

    assert restored == 1
    assert repaired[-1].start_time == pytest.approx(mapped_lines[-1].start_time)
    assert repaired[-1].words[-1].start_time == pytest.approx(
        mapped_lines[-1].words[-1].start_time
    )
    assert repaired[-1].end_time == pytest.approx(21.42)


def test_extend_final_line_last_word_to_baseline_end_skips_when_later_phrase_exists():
    baseline_lines = [
        _line(["In", "a", "day", "or", "two"], 17.05, 21.42),
    ]
    mapped_lines = [
        _line(["In", "a", "day", "or", "two"], 17.37, 20.87),
    ]
    whisper_words = [
        TranscriptionWord(text="in", start=17.4, end=17.8, probability=1.0),
        TranscriptionWord(text="a", start=17.8, end=18.1, probability=1.0),
        TranscriptionWord(text="day", start=18.1, end=18.8, probability=1.0),
        TranscriptionWord(text="or", start=18.8, end=19.2, probability=1.0),
        TranscriptionWord(text="two", start=21.4, end=22.0, probability=1.0),
    ]

    repaired, restored = _extend_final_line_last_word_to_baseline_end(
        mapped_lines,
        baseline_lines,
        whisper_words,
    )

    assert restored == 0
    assert repaired[-1].end_time == pytest.approx(mapped_lines[-1].end_time)


def test_restore_leading_alternating_hook_start_to_baseline_reanchors_start_only():
    baseline_lines = [
        _line(["Take", "on", "me"], 1.12, 4.57),
        _line(["Take", "me", "on"], 6.84, 10.42),
        _line(["I'll", "be", "gone"], 12.23, 15.81),
    ]
    mapped_lines = [
        _line(["Take", "on", "me"], 0.64, 4.57),
        _line(["Take", "me", "on"], 6.45, 9.91),
        _line(["I'll", "be", "gone"], 12.34, 15.39),
    ]
    whisper_words = [
        TranscriptionWord(text="take", start=0.64, end=1.2, probability=1.0),
        TranscriptionWord(text="on", start=1.2, end=3.08, probability=1.0),
        TranscriptionWord(text="me", start=3.08, end=4.22, probability=1.0),
        TranscriptionWord(text="take", start=4.22, end=5.48, probability=1.0),
        TranscriptionWord(text="me", start=5.48, end=8.08, probability=1.0),
        TranscriptionWord(text="on", start=8.08, end=9.7, probability=1.0),
    ]

    repaired, restored = _restore_leading_alternating_hook_start_to_baseline(
        mapped_lines,
        baseline_lines,
        whisper_words,
    )

    assert restored == 1
    assert repaired[0].start_time == pytest.approx(1.12)
    assert repaired[0].end_time == pytest.approx(4.455)


def test_restore_leading_alternating_hook_start_to_baseline_skips_non_alternating():
    baseline_lines = [
        _line(["Take", "on", "me"], 1.12, 4.57),
        _line(["Take", "on", "me"], 6.84, 10.42),
    ]
    mapped_lines = [
        _line(["Take", "on", "me"], 0.64, 4.57),
        _line(["Take", "on", "me"], 6.45, 9.91),
    ]
    whisper_words = [
        TranscriptionWord(text="take", start=0.64, end=1.2, probability=1.0),
        TranscriptionWord(text="on", start=1.2, end=3.08, probability=1.0),
        TranscriptionWord(text="me", start=3.08, end=4.22, probability=1.0),
    ]

    repaired, restored = _restore_leading_alternating_hook_start_to_baseline(
        mapped_lines,
        baseline_lines,
        whisper_words,
    )

    assert restored == 0
    assert repaired[0].start_time == pytest.approx(mapped_lines[0].start_time)
