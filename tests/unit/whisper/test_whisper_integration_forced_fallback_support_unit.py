import numpy as np
import pytest

from y2karaoke.core.components.alignment.timing_models import (
    AudioFeatures,
    TranscriptionWord,
)
from y2karaoke.core.components.whisper.whisper_integration_forced_fallback import (
    attempt_whisperx_forced_alignment,
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


def test_attempt_whisperx_forced_alignment_restores_sparse_support_line_starts():
    baseline_lines = [
        _dur_line(0.99, 4.57, "Take on me"),
        _dur_line(6.45, 10.03, "Take me on"),
        _dur_line(11.91, 15.49, "I'll be gone"),
        _dur_line(17.37, 21.42, "In a day or two"),
    ]
    forced_lines = [
        _dur_line(1.12, 4.15, "Take on me"),
        _dur_line(6.48, 10.06, "Take me on"),
        _dur_line(11.76, 15.34, "I'll be gone"),
        _dur_line(17.2, 21.25, "In a day or two"),
    ]
    whisper_words = [
        TranscriptionWord(text="Okay", start=0.05, end=0.63, probability=0.3),
        TranscriptionWord(text="take", start=0.63, end=1.55, probability=0.8),
        TranscriptionWord(text="off", start=1.55, end=2.77, probability=0.5),
    ]
    audio_features = AudioFeatures(
        onset_times=np.array([1.12, 6.44, 11.9, 17.35], dtype=float),
        silence_regions=[],
        vocal_start=0.0,
        vocal_end=22.0,
        duration=22.0,
        energy_envelope=np.array([], dtype=float),
        energy_times=np.array([], dtype=float),
    )

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
        audio_features=audio_features,
        normalize_line_word_timings_fn=lambda lines: lines,
        enforce_monotonic_line_starts_fn=lambda lines: lines,
        enforce_non_overlapping_lines_fn=lambda lines: lines,
    )

    assert result is not None
    repaired_lines, _corrections, _payload = result
    assert repaired_lines[2].start_time == pytest.approx(11.91)
    assert repaired_lines[3].start_time == pytest.approx(17.37)
