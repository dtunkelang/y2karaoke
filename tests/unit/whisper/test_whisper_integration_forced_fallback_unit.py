from y2karaoke.core.models import Line, Word
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
