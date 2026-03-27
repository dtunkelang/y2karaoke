import pytest

from y2karaoke.core.components.alignment.timing_models import TranscriptionWord
from y2karaoke.core.components.whisper.whisper_integration_forced_fallback import (
    attempt_whisperx_forced_alignment,
)
from y2karaoke.core.components.whisper import (
    whisper_integration_forced_fallback as _forced,
)
from y2karaoke.core.models import Line, Word


class _Logger:
    def debug(self, *_args, **_kwargs):
        return None

    def info(self, *_args, **_kwargs):
        return None

    def warning(self, *_args, **_kwargs):
        return None


def _dur_line(start: float, end: float, text: str = "x") -> Line:
    return Line(words=[Word(text=text, start_time=start, end_time=end)])


def _dur_multi_line(start: float, end: float, tokens: list[str]) -> Line:
    step = (end - start) / max(len(tokens), 1)
    words = [
        Word(
            text=token,
            start_time=start + step * idx,
            end_time=start + step * (idx + 1),
        )
        for idx, token in enumerate(tokens)
    ]
    return Line(words=words)


def test_attempt_whisperx_forced_alignment_restores_sustained_line_compression():
    baseline_lines = [
        _dur_line(1.0, 5.3, "Take on me"),
        _dur_line(6.85, 10.55, "Take me on"),
        _dur_line(12.35, 16.5, "I'll be gone"),
        _dur_line(17.05, 22.05, "In a day or two"),
    ]
    forced_lines = [
        _dur_line(1.19, 2.42, "Take on me"),
        _dur_line(6.84, 9.59, "Take me on"),
        _dur_line(10.97, 12.31, "I'll be gone"),
        _dur_line(19.8, 21.68, "In a day or two"),
    ]

    result = attempt_whisperx_forced_alignment(
        lines=baseline_lines,
        baseline_lines=baseline_lines,
        vocals_path="vocals.wav",
        language="en",
        detected_lang="en",
        logger=_Logger(),
        used_model="base",
        reason="test",
        align_lines_with_whisperx_fn=lambda *_a, **_k: (
            forced_lines,
            {"forced_word_coverage": 1.0, "forced_line_coverage": 1.0},
        ),
        should_rollback_short_line_degradation_fn=lambda *_a, **_k: (False, 0, 0),
        restore_implausibly_short_lines_fn=lambda *_a, **_k: (forced_lines, 0),
        normalize_line_word_timings_fn=lambda lines: lines,
        enforce_monotonic_line_starts_fn=lambda lines: lines,
        enforce_non_overlapping_lines_fn=lambda lines: lines,
    )

    assert result is not None
    repaired_lines, _corrections, payload = result
    assert repaired_lines[0].start_time == pytest.approx(1.19)
    assert repaired_lines[0].end_time == pytest.approx(5.49)
    assert repaired_lines[2].end_time > forced_lines[2].end_time
    assert payload["whisperx_forced"] == 1.0


def test_attempt_whisperx_forced_alignment_restores_sustained_line_durations():
    baseline_lines = [
        _dur_line(1.0, 5.3, "Take on me"),
        _dur_line(6.85, 10.55, "Take me on"),
        _dur_line(12.35, 16.5, "I'll be gone"),
        _dur_line(17.05, 22.05, "In a day or two"),
    ]
    forced_lines = [
        _dur_line(1.19, 2.42, "Take on me"),
        _dur_line(6.84, 9.59, "Take me on"),
        _dur_line(10.97, 12.31, "I'll be gone"),
        _dur_line(19.8, 21.68, "In a day or two"),
    ]

    result = attempt_whisperx_forced_alignment(
        lines=baseline_lines,
        baseline_lines=baseline_lines,
        vocals_path="vocals.wav",
        language="en",
        detected_lang="en",
        logger=_Logger(),
        used_model="base",
        reason="test",
        align_lines_with_whisperx_fn=lambda *_a, **_k: (
            forced_lines,
            {"forced_word_coverage": 1.0, "forced_line_coverage": 1.0},
        ),
        should_rollback_short_line_degradation_fn=lambda *_a, **_k: (False, 0, 0),
        restore_implausibly_short_lines_fn=lambda *_a, **_k: (forced_lines, 0),
        normalize_line_word_timings_fn=lambda lines: lines,
        enforce_monotonic_line_starts_fn=lambda lines: lines,
        enforce_non_overlapping_lines_fn=lambda lines: lines,
    )

    assert result is not None
    repaired_lines, _corrections, payload = result
    assert repaired_lines[0].start_time == pytest.approx(1.19)
    assert repaired_lines[0].end_time == pytest.approx(5.49)
    assert repaired_lines[2].start_time == pytest.approx(10.97)
    assert repaired_lines[2].end_time > forced_lines[2].end_time
    assert payload["whisperx_forced"] == 1.0


def test_restore_sustained_durations_restore_baseline_start_on_extreme_collapse():
    baseline_lines = [
        _dur_line(1.3, 5.7, "Ah, ha, ha, ha, stayin' alive, stayin' alive"),
        _dur_line(5.85, 16.0, "Ah, ha, ha, ha, stayin' alive"),
    ]
    forced_lines = [
        _dur_line(1.39, 5.4, "Ah, ha, ha, ha, stayin' alive, stayin' alive"),
        _dur_line(8.73, 9.29, "Ah, ha, ha, ha, stayin' alive"),
    ]

    repaired_lines, restored_count = (
        _forced._restore_sustained_line_durations_from_source(
            baseline_lines,
            forced_lines,
        )
    )

    assert restored_count == 1
    assert repaired_lines[0].start_time == pytest.approx(1.39)
    assert repaired_lines[1].start_time == pytest.approx(5.85)
    assert repaired_lines[1].end_time == pytest.approx(16.0)


def test_restore_sustained_durations_shift_compact_recovered_lines_later():
    baseline_lines = [
        _dur_multi_line(1.0, 5.58, ["Take", "on", "me"]),
        _dur_multi_line(6.13, 10.71, ["Take", "me", "on"]),
        _dur_multi_line(11.26, 15.84, ["I'll", "be", "gone"]),
        _dur_multi_line(16.38, 21.0, ["In", "a", "day", "or", "two"]),
    ]
    forced_lines = [
        _dur_multi_line(1.12, 4.15, ["Take", "on", "me"]),
        _dur_multi_line(6.48, 7.08, ["Take", "me", "on"]),
        _dur_multi_line(11.76, 13.79, ["I'll", "be", "gone"]),
        _dur_multi_line(17.2, 18.71, ["In", "a", "day", "or", "two"]),
    ]

    repaired_lines, restored_count = (
        _forced._restore_sustained_line_durations_from_source(
            baseline_lines,
            forced_lines,
        )
    )

    assert restored_count == 3
    assert repaired_lines[1].start_time > forced_lines[1].start_time
    assert repaired_lines[2].start_time > forced_lines[2].start_time
    assert repaired_lines[3].start_time == pytest.approx(forced_lines[3].start_time)


def test_attempt_whisperx_forced_alignment_redistributes_sparse_sustained_words():
    baseline_lines = [
        _dur_multi_line(0.99, 4.57, ["Take", "on", "me"]),
        _dur_multi_line(6.45, 10.03, ["Take", "me", "on"]),
        _dur_multi_line(11.91, 15.49, ["I'll", "be", "gone"]),
        _dur_multi_line(17.37, 21.42, ["In", "a", "day", "or", "two"]),
    ]
    forced_lines = [
        _dur_multi_line(1.12, 4.15, ["Take", "on", "me"]),
        _dur_multi_line(6.84, 10.42, ["Take", "me", "on"]),
        _dur_multi_line(12.23, 15.81, ["I'll", "be", "gone"]),
        _dur_multi_line(17.2, 21.25, ["In", "a", "day", "or", "two"]),
    ]
    whisper_words = [
        TranscriptionWord(text="Okay", start=0.05, end=0.63, probability=0.3),
        TranscriptionWord(text="take", start=0.63, end=1.55, probability=0.8),
        TranscriptionWord(text="off", start=1.55, end=2.77, probability=0.5),
    ]

    result = attempt_whisperx_forced_alignment(
        lines=baseline_lines,
        baseline_lines=baseline_lines,
        vocals_path="vocals.wav",
        language="en",
        detected_lang="en",
        logger=_Logger(),
        used_model="base",
        reason="test",
        align_lines_with_whisperx_fn=lambda *_a, **_k: (
            forced_lines,
            {"forced_word_coverage": 1.0, "forced_line_coverage": 1.0},
        ),
        should_rollback_short_line_degradation_fn=lambda *_a, **_k: (False, 0, 0),
        restore_implausibly_short_lines_fn=lambda *_a, **_k: (forced_lines, 0),
        whisper_words=whisper_words,
        normalize_line_word_timings_fn=lambda lines: lines,
        enforce_monotonic_line_starts_fn=lambda lines: lines,
        enforce_non_overlapping_lines_fn=lambda lines: lines,
    )

    assert result is not None
    repaired_lines, _corrections, _payload = result
    assert repaired_lines[3].words[-1].start_time < forced_lines[3].words[-1].start_time
