from y2karaoke.core.models import Line, Word
from y2karaoke.core.components.alignment.timing_models import TranscriptionWord
from y2karaoke.core.components.whisper.whisper_integration_forced_fallback import (
    attempt_whisperx_forced_alignment,
)


class _Logger:
    def warning(self, *_args, **_kwargs):
        return None

    def info(self, *_args, **_kwargs):
        return None


def _line(start: float) -> Line:
    return Line(words=[Word(text="x", start_time=start, end_time=start + 0.2)])


def test_attempt_whisperx_forced_alignment_returns_none_when_under_coverage():
    lines = [_line(1.0)]
    result = attempt_whisperx_forced_alignment(
        lines=lines,
        baseline_lines=lines,
        vocals_path="vocals.wav",
        language="en",
        detected_lang=None,
        logger=_Logger(),
        used_model="base",
        reason="test",
        align_lines_with_whisperx_fn=lambda *_a, **_k: (
            lines,
            {"forced_word_coverage": 0.1, "forced_line_coverage": 1.0},
        ),
        should_rollback_short_line_degradation_fn=lambda *_a, **_k: (False, 0, 0),
        restore_implausibly_short_lines_fn=lambda *_a, **_k: (lines, 0),
    )
    assert result is None


def test_attempt_whisperx_forced_alignment_returns_payload_on_success():
    lines = [_line(1.0), _line(2.0)]
    result = attempt_whisperx_forced_alignment(
        lines=lines,
        baseline_lines=lines,
        vocals_path="vocals.wav",
        language="en",
        detected_lang=None,
        logger=_Logger(),
        used_model="base",
        reason="sparse",
        align_lines_with_whisperx_fn=lambda *_a, **_k: (
            lines,
            {"forced_word_coverage": 0.8, "forced_line_coverage": 0.9},
        ),
        should_rollback_short_line_degradation_fn=lambda *_a, **_k: (False, 0, 0),
        restore_implausibly_short_lines_fn=lambda *_a, **_k: (lines, 0),
    )
    assert result is not None
    out_lines, corrections, payload = result
    assert len(out_lines) == 2
    assert "sparse" in corrections[0]
    assert payload["whisperx_forced"] == 1.0


def test_attempt_whisperx_forced_alignment_uses_detected_lang_fallback():
    lines = [_line(1.0)]
    observed: list[str | None] = []

    def _align(_lines, _vocals_path, language, _logger):
        observed.append(language)
        return (
            lines,
            {"forced_word_coverage": 0.9, "forced_line_coverage": 1.0},
        )

    result = attempt_whisperx_forced_alignment(
        lines=lines,
        baseline_lines=lines,
        vocals_path="vocals.wav",
        language=None,
        detected_lang="fr",
        logger=_Logger(),
        used_model="base",
        reason="tail shortfall",
        align_lines_with_whisperx_fn=_align,
        should_rollback_short_line_degradation_fn=lambda *_a, **_k: (False, 0, 0),
        restore_implausibly_short_lines_fn=lambda *_a, **_k: (lines, 0),
    )
    assert result is not None
    assert observed == ["fr"]


def test_attempt_whisperx_forced_alignment_reanchors_leading_article_to_content_word():
    line = Line(
        words=[
            Word(text="The", start_time=19.7, end_time=20.0),
            Word(text="needle", start_time=20.1, end_time=20.5),
            Word(text="tears", start_time=20.5, end_time=20.7),
            Word(text="a", start_time=22.3, end_time=22.4),
            Word(text="hole", start_time=22.4, end_time=22.8),
        ]
    )
    whisper_words = [
        TranscriptionWord(text="real", start=19.84, end=20.18, probability=1.0),
        TranscriptionWord(text="The", start=21.06, end=22.38, probability=1.0),
        TranscriptionWord(text="needle", start=22.38, end=23.04, probability=1.0),
        TranscriptionWord(text="tears", start=23.04, end=24.18, probability=1.0),
    ]

    result = attempt_whisperx_forced_alignment(
        lines=[line],
        baseline_lines=[line],
        vocals_path="vocals.wav",
        language="en",
        detected_lang="en",
        logger=_Logger(),
        used_model="base",
        reason="test",
        align_lines_with_whisperx_fn=lambda *_a, **_k: (
            [line],
            {"forced_word_coverage": 1.0, "forced_line_coverage": 1.0},
        ),
        should_rollback_short_line_degradation_fn=lambda *_a, **_k: (False, 0, 0),
        restore_implausibly_short_lines_fn=lambda *_a, **_k: ([line], 0),
        whisper_words=whisper_words,
        normalize_line_word_timings_fn=lambda lines: lines,
        enforce_monotonic_line_starts_fn=lambda lines: lines,
        enforce_non_overlapping_lines_fn=lambda lines: lines,
    )

    assert result is not None
    forced_lines, _corrections, _payload = result
    assert forced_lines[0].words[0].start_time >= 21.95
    assert abs(forced_lines[0].words[1].start_time - 22.38) < 0.05
