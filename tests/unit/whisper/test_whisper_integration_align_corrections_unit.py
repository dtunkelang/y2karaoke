import pytest

from y2karaoke.core.components.alignment.timing_models import TranscriptionWord
from y2karaoke.core.components.whisper.whisper_integration_align_corrections import (
    _restore_alternating_middle_hook_from_phrase_window,
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
