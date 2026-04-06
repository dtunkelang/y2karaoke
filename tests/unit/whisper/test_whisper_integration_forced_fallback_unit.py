import numpy as np
import pytest

from y2karaoke.core.components.alignment.timing_models import (
    AudioFeatures,
    TranscriptionWord,
)
from y2karaoke.core.components.whisper.whisper_integration_forced_fallback import (
    attempt_whisperx_forced_alignment,
)
from y2karaoke.core.components.whisper import (
    whisper_alignment as _wa,
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


def _line(start: float) -> Line:
    return Line(
        words=[Word(text="x", start_time=start, end_time=start + 0.4)],
    )


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


def test_forced_alignment_restores_start_for_unsupported_leading_prefix() -> None:
    baseline_line = _dur_multi_line(
        0.944,
        2.433,
        ["Uh,", "summa-lumma,", "dooma-lumma,", "you", "assumin'", "I'm", "a", "human"],
    )
    forced_line = _dur_multi_line(
        0.554,
        2.951,
        ["Uh,", "summa-lumma,", "dooma-lumma,", "you", "assumin'", "I'm", "a", "human"],
    )
    whisper_words = [
        TranscriptionWord(text="I'm", start=0.98, end=1.14, probability=0.6),
        TranscriptionWord(text="a", start=1.14, end=1.24, probability=0.8),
        TranscriptionWord(text="human", start=1.24, end=1.48, probability=0.7),
        TranscriptionWord(text="what", start=1.48, end=1.7, probability=0.7),
    ]
    audio_features = AudioFeatures(
        onset_times=np.array([0.98, 1.24, 2.99], dtype=float),
        silence_regions=[],
        vocal_start=0.98,
        vocal_end=4.6,
        duration=4.6,
        energy_envelope=np.array([], dtype=float),
        energy_times=np.array([], dtype=float),
    )

    result = attempt_whisperx_forced_alignment(
        lines=[baseline_line, _dur_line(2.8, 4.45, "What I gotta do")],
        baseline_lines=[baseline_line, _dur_line(2.8, 4.45, "What I gotta do")],
        vocals_path="vocals.wav",
        language="en",
        detected_lang="en",
        logger=_Logger(),
        used_model="base",
        reason="test",
        align_lines_with_whisperx_fn=lambda *_a, **_k: (
            [forced_line, _dur_line(2.987, 4.596, "What I gotta do")],
            {"forced_word_coverage": 1.0, "forced_line_coverage": 1.0},
        ),
        should_rollback_short_line_degradation_fn=lambda *_a, **_k: (False, 0, 0),
        restore_implausibly_short_lines_fn=lambda *_a, **_k: ([forced_line], 0),
        whisper_words=whisper_words,
        audio_features=audio_features,
        normalize_line_word_timings_fn=lambda lines: lines,
        enforce_monotonic_line_starts_fn=lambda lines: lines,
        enforce_non_overlapping_lines_fn=lambda lines: lines,
    )

    assert result is not None
    forced_lines, _corrections, _payload = result
    assert forced_lines[0].start_time == pytest.approx(0.944)
    assert forced_lines[0].end_time == pytest.approx(2.926, abs=0.01)


def test_attempt_whisperx_forced_alignment_keeps_earlier_supported_prefix_start():
    baseline_line = _dur_multi_line(
        1.0,
        5.635,
        ["I've", "never", "seen", "a", "diamond", "in", "the", "flesh"],
    )
    forced_line = _dur_multi_line(
        0.55,
        3.462,
        ["I've", "never", "seen", "a", "diamond", "in", "the", "flesh"],
    )
    whisper_words = [
        TranscriptionWord(text="diamond", start=0.952, end=1.594, probability=0.85),
        TranscriptionWord(text="in", start=1.896, end=1.996, probability=0.91),
        TranscriptionWord(text="the", start=2.036, end=2.117, probability=0.57),
        TranscriptionWord(text="flesh", start=2.217, end=3.462, probability=0.75),
    ]
    audio_features = AudioFeatures(
        onset_times=np.array([0.56, 0.79, 0.95, 5.78], dtype=float),
        silence_regions=[],
        vocal_start=0.56,
        vocal_end=10.8,
        duration=22.0,
        energy_envelope=np.array([], dtype=float),
        energy_times=np.array([], dtype=float),
    )

    result = attempt_whisperx_forced_alignment(
        lines=[
            baseline_line,
            _dur_line(5.784, 10.758, "I cut my teeth on wedding rings in the movies"),
        ],
        baseline_lines=[
            baseline_line,
            _dur_line(5.784, 10.758, "I cut my teeth on wedding rings in the movies"),
        ],
        vocals_path="vocals.wav",
        language="en",
        detected_lang="en",
        logger=_Logger(),
        used_model="base",
        reason="test",
        align_lines_with_whisperx_fn=lambda *_a, **_k: (
            [
                forced_line,
                _dur_line(
                    5.784, 10.758, "I cut my teeth on wedding rings in the movies"
                ),
            ],
            {"forced_word_coverage": 1.0, "forced_line_coverage": 1.0},
        ),
        should_rollback_short_line_degradation_fn=lambda *_a, **_k: (False, 0, 0),
        restore_implausibly_short_lines_fn=lambda *_a, **_k: ([forced_line], 0),
        whisper_words=whisper_words,
        audio_features=audio_features,
        normalize_line_word_timings_fn=lambda lines: lines,
        enforce_monotonic_line_starts_fn=lambda lines: lines,
        enforce_non_overlapping_lines_fn=lambda lines: lines,
    )

    assert result is not None
    forced_lines, _corrections, _payload = result
    assert forced_lines[0].start_time == pytest.approx(0.55, abs=0.01)
    assert forced_lines[0].end_time == pytest.approx(3.462, abs=0.01)


def test_attempt_whisperx_forced_alignment_rejects_compact_line_drift():
    baseline_lines = [
        _dur_line(0.0, 1.98, "Guess who's back, back again?"),
        _dur_line(2.2, 4.18, "Shady's back, tell a friend"),
        _dur_line(4.4, 6.48, "Guess who's back? Guess who's back?"),
        _dur_line(6.7, 8.78, "Guess who's back? Guess who's back?"),
        _dur_line(9.0, 11.08, "Guess who's back? Guess who's back?"),
        _dur_line(11.3, 13.27, "Guess who's back?"),
    ]
    forced_lines = [
        _dur_line(0.5, 2.69, "Guess who's back, back again?"),
        _dur_line(2.7, 4.89, "Shady's back, tell a friend"),
        _dur_line(8.64, 10.74, "Guess who's back? Guess who's back?"),
        _dur_line(11.14, 12.77, "Guess who's back? Guess who's back?"),
        _dur_line(12.95, 13.85, "Guess who's back? Guess who's back?"),
        _dur_line(12.96, 14.93, "Guess who's back?"),
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

    assert result is None


def test_attempt_whisperx_forced_alignment_shifts_refrains_after_long_lines():
    baseline_lines = [
        Line(
            words=[
                Word(text="If", start_time=1.1, end_time=1.35),
                Word(text="you're", start_time=1.35, end_time=1.65),
                Word(text="lost,", start_time=1.65, end_time=2.05),
                Word(text="you", start_time=2.05, end_time=2.3),
                Word(text="can", start_time=2.3, end_time=2.55),
                Word(text="look", start_time=2.55, end_time=2.95),
                Word(text="and", start_time=2.95, end_time=3.2),
                Word(text="you", start_time=3.2, end_time=3.45),
                Word(text="will", start_time=3.45, end_time=3.7),
                Word(text="find", start_time=3.7, end_time=4.1),
                Word(text="me", start_time=4.1, end_time=4.7),
            ]
        ),
        Line(
            words=[
                Word(text="Time", start_time=4.521, end_time=6.567),
                Word(text="after", start_time=6.808, end_time=7.229),
                Word(text="time", start_time=7.269, end_time=7.791),
            ]
        ),
        Line(
            words=[
                Word(text="If", start_time=8.732, end_time=8.812),
                Word(text="you", start_time=8.872, end_time=9.073),
                Word(text="fall,", start_time=9.113, end_time=9.574),
                Word(text="I", start_time=9.654, end_time=9.754),
                Word(text="will", start_time=9.794, end_time=9.975),
                Word(text="catch", start_time=10.035, end_time=10.436),
                Word(text="you,", start_time=10.476, end_time=10.657),
                Word(text="I'll", start_time=10.797, end_time=10.957),
                Word(text="be", start_time=11.017, end_time=11.358),
                Word(text="waiting", start_time=11.398, end_time=11.779),
            ]
        ),
        Line(
            words=[
                Word(text="Time", start_time=10.989, end_time=11.651),
                Word(text="after", start_time=11.711, end_time=12.293),
                Word(text="time", start_time=12.333, end_time=12.674),
            ]
        ),
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
            baseline_lines,
            {"forced_word_coverage": 1.0, "forced_line_coverage": 1.0},
        ),
        should_rollback_short_line_degradation_fn=lambda *_a, **_k: (False, 0, 0),
        restore_implausibly_short_lines_fn=lambda *_a, **_k: (baseline_lines, 0),
        normalize_line_word_timings_fn=lambda lines: lines,
        enforce_monotonic_line_starts_fn=lambda lines: lines,
        enforce_non_overlapping_lines_fn=lambda lines: lines,
    )

    assert result is not None


def test_finalize_forced_line_timing_can_disable_post_normalize(monkeypatch):
    lines = [_dur_multi_line(0.0, 2.0, ["a", "b", "c"])]
    calls: list[str] = []

    monkeypatch.setenv("Y2K_FORCE_FINALIZE_ENABLE_POST_NORMALIZE", "0")
    monkeypatch.setattr(
        _forced,
        "_reanchor_forced_lines_to_local_content_words",
        lambda forced_lines, _words: (forced_lines, 0),
    )
    monkeypatch.setattr(
        _forced,
        "_reanchor_medium_lines_to_earlier_exact_prefixes_impl",
        lambda forced_lines, _words, **_kwargs: (forced_lines, 0),
    )
    monkeypatch.setattr(
        _forced,
        "_retime_three_word_lines_from_suffix_matches",
        lambda forced_lines, _words: (forced_lines, 0),
    )
    monkeypatch.setattr(
        _forced,
        "_post_normalize_sparse_support_repairs",
        lambda **_kwargs: calls.append("post_normalize"),
    )

    result = _forced._finalize_forced_line_timing(
        forced_lines=lines,
        baseline_lines=lines,
        whisper_words=[],
        audio_features=None,
        logger=_Logger(),
        normalize_line_word_timings_fn=lambda current: current,
        enforce_monotonic_line_starts_fn=lambda current: current,
        enforce_non_overlapping_lines_fn=lambda current: current,
    )

    assert result == lines
    assert calls == []


def test_finalize_forced_line_timing_can_disable_non_overlap(monkeypatch):
    lines = [_dur_multi_line(0.0, 2.0, ["a", "b", "c"])]
    calls: list[str] = []

    monkeypatch.setenv("Y2K_FORCE_FINALIZE_ENABLE_NON_OVERLAP", "0")
    monkeypatch.setattr(
        _forced,
        "_reanchor_forced_lines_to_local_content_words",
        lambda forced_lines, _words: (forced_lines, 0),
    )
    monkeypatch.setattr(
        _forced,
        "_reanchor_medium_lines_to_earlier_exact_prefixes_impl",
        lambda forced_lines, _words, **_kwargs: (forced_lines, 0),
    )
    monkeypatch.setattr(
        _forced,
        "_retime_three_word_lines_from_suffix_matches",
        lambda forced_lines, _words: (forced_lines, 0),
    )
    monkeypatch.setattr(
        _forced,
        "_post_normalize_sparse_support_repairs",
        lambda **_kwargs: lines,
    )

    result = _forced._finalize_forced_line_timing(
        forced_lines=lines,
        baseline_lines=lines,
        whisper_words=[],
        audio_features=None,
        logger=_Logger(),
        normalize_line_word_timings_fn=lambda current: current,
        enforce_monotonic_line_starts_fn=lambda current: current,
        enforce_non_overlapping_lines_fn=lambda current: calls.append("non_overlap"),
    )

    assert result == lines
    assert calls == []


def test_attempt_whisperx_forced_alignment_shifts_refrains_before_non_overlap():
    baseline_lines = [
        Line(
            words=[
                Word(text="If", start_time=1.092, end_time=1.253),
                Word(text="you're", start_time=1.273, end_time=1.635),
                Word(text="lost,", start_time=1.655, end_time=2.137),
                Word(text="you", start_time=2.177, end_time=2.378),
                Word(text="can", start_time=2.399, end_time=2.66),
                Word(text="look", start_time=2.7, end_time=3.001),
                Word(text="and", start_time=3.182, end_time=3.303),
                Word(text="you", start_time=3.343, end_time=3.564),
                Word(text="will", start_time=3.604, end_time=3.885),
                Word(text="find", start_time=4.026, end_time=4.488),
                Word(text="me", start_time=4.528, end_time=4.749),
            ]
        ),
        Line(
            words=[
                Word(text="Time", start_time=4.501, end_time=4.882),
                Word(text="after", start_time=4.902, end_time=5.103),
                Word(text="time", start_time=6.086, end_time=6.547),
            ]
        ),
        Line(
            words=[
                Word(text="If", start_time=8.732, end_time=8.812),
                Word(text="you", start_time=8.852, end_time=9.073),
                Word(text="fall,", start_time=9.113, end_time=9.574),
                Word(text="I", start_time=9.654, end_time=9.734),
                Word(text="will", start_time=9.774, end_time=9.995),
                Word(text="catch", start_time=10.035, end_time=10.416),
                Word(text="you,", start_time=10.456, end_time=10.677),
                Word(text="I'll", start_time=10.797, end_time=10.957),
                Word(text="be", start_time=10.997, end_time=11.318),
                Word(text="waiting", start_time=11.398, end_time=11.799),
            ]
        ),
        Line(
            words=[
                Word(text="Time", start_time=10.989, end_time=11.31),
                Word(text="after", start_time=11.651, end_time=12.514),
                Word(text="time", start_time=13.456, end_time=13.897),
            ]
        ),
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
            baseline_lines,
            {"forced_word_coverage": 1.0, "forced_line_coverage": 1.0},
        ),
        should_rollback_short_line_degradation_fn=lambda *_a, **_k: (False, 0, 0),
        restore_implausibly_short_lines_fn=lambda *_a, **_k: (baseline_lines, 0),
        normalize_line_word_timings_fn=_wa._normalize_line_word_timings,
        enforce_monotonic_line_starts_fn=_wa._enforce_monotonic_line_starts,
        enforce_non_overlapping_lines_fn=_wa._enforce_non_overlapping_lines,
    )

    assert result is not None
    repaired_lines, _corrections, payload = result
    assert payload["shifted_refrain_followup_gaps"] == pytest.approx(2.0)
    assert repaired_lines[1].start_time == pytest.approx(6.099, abs=0.01)
    assert repaired_lines[2].end_time == pytest.approx(11.799, abs=0.01)
    assert repaired_lines[3].start_time == pytest.approx(13.149, abs=0.01)


def test_attempt_whisperx_forced_alignment_skips_long_parenthetical_refrain_gap_shift():
    forced_lines = [
        Line(
            words=[
                Word(text="(Turn", start_time=0.55, end_time=1.02),
                Word(text="around,", start_time=1.04, end_time=2.02),
                Word(text="bright", start_time=2.18, end_time=3.18),
                Word(text="eyes)", start_time=3.62, end_time=4.29),
            ]
        ),
        Line(
            words=[
                Word(text="Every", start_time=4.272, end_time=4.6),
                Word(text="now", start_time=4.6, end_time=4.93),
                Word(text="and", start_time=4.93, end_time=5.16),
                Word(text="then,", start_time=5.16, end_time=5.46),
                Word(text="I", start_time=5.46, end_time=5.63),
                Word(text="fall", start_time=5.63, end_time=6.08),
                Word(text="apart", start_time=6.08, end_time=6.54),
            ]
        ),
        Line(
            words=[
                Word(text="(Turn", start_time=6.529, end_time=7.01),
                Word(text="around,", start_time=7.03, end_time=7.95),
                Word(text="bright", start_time=8.06, end_time=8.95),
                Word(text="eyes)", start_time=9.28, end_time=9.867),
            ]
        ),
        Line(
            words=[
                Word(text="Every", start_time=9.789, end_time=10.12),
                Word(text="now", start_time=10.12, end_time=10.39),
                Word(text="and", start_time=10.39, end_time=10.64),
                Word(text="then,", start_time=10.64, end_time=10.89),
                Word(text="I", start_time=10.89, end_time=11.03),
                Word(text="fall", start_time=11.03, end_time=11.29),
                Word(text="apart", start_time=11.29, end_time=11.575),
            ]
        ),
    ]

    result = attempt_whisperx_forced_alignment(
        lines=forced_lines,
        baseline_lines=forced_lines,
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
    assert payload["shifted_refrain_followup_gaps"] == pytest.approx(0.0)
    assert repaired_lines[2].start_time == pytest.approx(6.529, abs=0.01)


def test_attempt_whisperx_forced_alignment_rejects_compact_line_duration_collapse():
    baseline_lines = [
        _dur_line(2.4, 3.92, "Guess who's back"),
        _dur_line(4.49, 5.49, "Back again"),
        _dur_line(6.06, 7.06, "Shady's back"),
        _dur_line(7.63, 9.15, "Tell a friend"),
    ]
    forced_lines = [
        _dur_line(2.75, 3.57, "Guess who's back"),
        _dur_line(4.91, 5.66, "Back again"),
        _dur_line(7.03, 7.23, "Shady's back"),
        _dur_line(7.25, 7.63, "Tell a friend"),
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

    assert result is None


def test_attempt_whisperx_forced_alignment_rejects_post_normalize_compact_collapse():
    baseline_lines = [
        _dur_line(2.4, 3.92, "Guess who's back"),
        _dur_line(4.49, 5.49, "Back again"),
        _dur_line(6.06, 7.06, "Shady's back"),
        _dur_line(7.63, 9.15, "Tell a friend"),
    ]
    raw_forced_lines = [
        _dur_line(2.75, 3.9, "Guess who's back"),
        _dur_line(4.91, 5.86, "Back again"),
        _dur_line(6.9, 7.8, "Shady's back"),
        _dur_line(8.0, 9.2, "Tell a friend"),
    ]
    normalized_lines = [
        _dur_line(2.75, 3.57, "Guess who's back"),
        _dur_line(4.91, 5.66, "Back again"),
        _dur_line(7.03, 7.23, "Shady's back"),
        _dur_line(7.25, 7.63, "Tell a friend"),
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
            raw_forced_lines,
            {"forced_word_coverage": 1.0, "forced_line_coverage": 1.0},
        ),
        should_rollback_short_line_degradation_fn=lambda *_a, **_k: (False, 0, 0),
        restore_implausibly_short_lines_fn=lambda *_a, **_k: (raw_forced_lines, 0),
        normalize_line_word_timings_fn=lambda _lines: normalized_lines,
        enforce_monotonic_line_starts_fn=lambda lines: lines,
        enforce_non_overlapping_lines_fn=lambda lines: lines,
    )

    assert result is None


def test_post_normalize_sparse_repairs_restore_two_word_lines_from_source():
    baseline_lines = [
        _dur_multi_line(2.4, 3.92, ["Guess", "who's", "back"]),
        _dur_multi_line(4.69, 5.78, ["Back", "again"]),
        _dur_multi_line(6.41, 7.5, ["Shady's", "back"]),
        _dur_multi_line(8.13, 9.79, ["Tell", "a", "friend"]),
    ]
    forced_lines = [
        _dur_multi_line(2.75, 3.58, ["Guess", "who's", "back"]),
        _dur_multi_line(4.4, 5.23, ["Back", "again"]),
        _dur_multi_line(7.01, 7.86, ["Shady's", "back"]),
        _dur_multi_line(8.77, 9.99, ["Tell", "a", "friend"]),
    ]

    repaired = _forced._post_normalize_sparse_support_repairs(
        baseline_lines=baseline_lines,
        forced_lines=forced_lines,
        whisper_words=[],
        audio_features=None,
        logger=_Logger(),
        normalize_line_word_timings_fn=lambda lines: lines,
    )

    assert repaired[1].start_time == pytest.approx(4.69)
    assert repaired[1].end_time == pytest.approx(5.78)
    assert repaired[3].start_time == pytest.approx(8.77)


def test_attempt_whisperx_forced_alignment_restores_sparse_durations_post_normalize():
    baseline_lines = [
        _dur_line(0.99, 4.21, "Take on me"),
        _dur_multi_line(6.51, 9.73, ["Take", "me", "on", "right", "now"]),
        _dur_line(12.03, 15.25, "I'll be gone"),
        _dur_line(17.56, 21.42, "In a day or two"),
    ]
    raw_forced_lines = [
        _dur_line(1.12, 4.15, "Take on me"),
        _dur_multi_line(6.48, 9.7, ["Take", "me", "on", "right", "now"]),
        _dur_line(11.72, 14.94, "I'll be gone"),
        _dur_line(17.19, 21.05, "In a day or two"),
    ]
    normalized_lines = [
        _dur_line(1.12, 4.15, "Take on me"),
        _dur_multi_line(6.48, 7.02, ["Take", "me", "on", "right", "now"]),
        _dur_line(11.72, 14.23, "I'll be gone"),
        _dur_line(17.19, 17.73, "In a day or two"),
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
            raw_forced_lines,
            {"forced_word_coverage": 1.0, "forced_line_coverage": 1.0},
        ),
        should_rollback_short_line_degradation_fn=lambda *_a, **_k: (False, 0, 0),
        restore_implausibly_short_lines_fn=lambda *_a, **_k: (raw_forced_lines, 0),
        whisper_words=whisper_words,
        normalize_line_word_timings_fn=lambda _lines: normalized_lines,
        enforce_monotonic_line_starts_fn=lambda lines: lines,
        enforce_non_overlapping_lines_fn=lambda lines: lines,
    )

    assert result is not None
    repaired_lines, _corrections, _payload = result
    assert repaired_lines[1].start_time == pytest.approx(6.48)
    assert repaired_lines[1].end_time == pytest.approx(9.7)
    assert repaired_lines[3].end_time == pytest.approx(21.03, abs=0.05)


def test_attempt_whisperx_forced_alignment_rejects_sparse_forced_on_better_baseline():
    baseline_lines = [
        _dur_line(0.99, 4.21, "Take on me"),
        _dur_line(6.51, 9.73, "Take me on"),
        _dur_line(12.03, 15.25, "I'll be gone"),
        _dur_line(17.56, 21.42, "In a day or two"),
    ]
    forced_lines = [
        _dur_line(1.12, 4.15, "Take on me"),
        _dur_line(6.32, 9.54, "Take me on"),
        _dur_line(11.94, 14.45, "I'll be gone"),
        _dur_line(16.79, 20.65, "In a day or two"),
    ]
    whisper_words = [
        TranscriptionWord(text="Okay", start=0.05, end=0.63, probability=0.3),
        TranscriptionWord(text="take", start=0.63, end=1.55, probability=0.8),
        TranscriptionWord(text="off", start=1.55, end=2.77, probability=0.5),
    ]
    audio_features = AudioFeatures(
        onset_times=np.array([0.42, 1.0, 6.43, 6.68, 12.14, 16.21, 17.55], dtype=float),
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

    assert result is None


def test_attempt_whisperx_forced_alignment_shifts_sparse_support_line_to_better_onset():
    baseline_lines = [
        _dur_line(0.99, 4.21, "Take on me"),
        _dur_line(6.51, 9.73, "Take me on"),
        _dur_line(12.03, 15.25, "I'll be gone"),
        _dur_line(17.56, 21.42, "In a day or two"),
    ]
    forced_lines = [
        _dur_line(1.12, 4.15, "Take on me"),
        _dur_line(6.48, 9.7, "Take me on"),
        _dur_line(11.72, 14.23, "I'll be gone"),
        _dur_line(17.19, 21.05, "In a day or two"),
    ]
    whisper_words = [
        TranscriptionWord(text="Okay", start=0.05, end=0.63, probability=0.3),
        TranscriptionWord(text="take", start=0.63, end=1.55, probability=0.8),
        TranscriptionWord(text="off", start=1.55, end=2.77, probability=0.5),
    ]
    audio_features = AudioFeatures(
        onset_times=np.array(
            [1.0, 6.48, 12.14, 17.19],
            dtype=float,
        ),
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
    assert repaired_lines[2].start_time == pytest.approx(12.14)
    assert repaired_lines[1].start_time == pytest.approx(6.48)


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
