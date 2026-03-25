from y2karaoke.core.models import Line, Word
from y2karaoke.core.components.alignment.timing_models import (
    AudioFeatures,
    TranscriptionWord,
)
import pytest
import numpy as np

from y2karaoke.core.components.whisper.whisper_integration_forced_fallback import (
    attempt_whisperx_forced_alignment,
)
from y2karaoke.core.components.whisper import (
    whisper_integration_forced_fallback as _forced,
)


class _Logger:
    def warning(self, *_args, **_kwargs):
        return None

    def info(self, *_args, **_kwargs):
        return None


def _line(start: float) -> Line:
    return Line(words=[Word(text="x", start_time=start, end_time=start + 0.2)])


def _dur_line(start: float, end: float, text: str = "x") -> Line:
    return Line(words=[Word(text=text, start_time=start, end_time=end)])


def _dur_multi_line(start: float, end: float, words: list[str]) -> Line:
    step = (end - start) / max(len(words), 1)
    built: list[Word] = []
    cursor = start
    for idx, text in enumerate(words):
        next_end = end if idx == len(words) - 1 else cursor + step * 0.9
        built.append(Word(text=text, start_time=cursor, end_time=next_end))
        cursor += step
    return Line(words=built)


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


def test_retime_three_word_lines_from_suffix_matches_rebuilds_late_suffix_window():
    forced_lines = [
        _dur_multi_line(7.01, 7.86, ["Shady's", "back"]),
        Line(
            words=[
                Word(text="Tell", start_time=8.122, end_time=9.487),
                Word(text="a", start_time=9.547, end_time=9.668),
                Word(text="friend", start_time=9.708, end_time=9.989),
            ]
        ),
    ]
    whisper_words = [
        TranscriptionWord(text="he's", start=7.33, end=7.51, probability=0.995),
        TranscriptionWord(text="back", start=7.51, end=7.79, probability=0.998),
        TranscriptionWord(text="still", start=7.87, end=9.11, probability=0.415),
        TranscriptionWord(text="a", start=9.11, end=9.49, probability=0.778),
        TranscriptionWord(text="friend", start=9.49, end=9.79, probability=0.921),
    ]

    repaired_lines, restored_count = (
        _forced._retime_three_word_lines_from_suffix_matches(
            forced_lines,
            whisper_words,
        )
    )

    assert restored_count == 1
    assert repaired_lines[1].start_time == pytest.approx(8.77, abs=0.03)
    assert repaired_lines[1].words[1].start_time == pytest.approx(9.11, abs=0.01)
    assert repaired_lines[1].words[2].start_time >= 9.70
    assert repaired_lines[1].words[2].end_time == pytest.approx(9.989, abs=0.01)


def test_retime_three_word_lines_from_suffix_matches_skips_balanced_refrain_lines():
    forced_lines = [
        Line(
            words=[
                Word(text="Time", start_time=5.831, end_time=6.296),
                Word(text="after", start_time=6.348, end_time=6.813),
                Word(text="time", start_time=6.864, end_time=7.381),
            ]
        )
    ]
    whisper_words = [
        TranscriptionWord(text="Time", start=5.72, end=6.08, probability=0.8),
        TranscriptionWord(text="after", start=6.12, end=6.89, probability=0.8),
        TranscriptionWord(text="time", start=7.059, end=7.52, probability=0.8),
    ]

    repaired_lines, restored_count = (
        _forced._retime_three_word_lines_from_suffix_matches(
            forced_lines,
            whisper_words,
        )
    )

    assert restored_count == 0
    assert repaired_lines[0].start_time == pytest.approx(5.831)


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


def test_restore_sustained_line_durations_from_source_restores_exact_baseline_start_for_extreme_collapse():
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


def test_restore_sustained_line_durations_from_source_shifts_compact_recovered_lines_later():
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


def test_attempt_whisperx_forced_alignment_shifts_repeated_short_refrains_after_long_lines():
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
    repaired_lines, _corrections, payload = result
    assert payload["shifted_refrain_followup_gaps"] == pytest.approx(2.0)
    assert repaired_lines[1].start_time > 6.0
    assert repaired_lines[3].start_time > 12.0


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


def test_post_normalize_sparse_support_repairs_restore_compact_two_word_lines_from_source():
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


def test_attempt_whisperx_forced_alignment_restores_sparse_support_durations_post_normalize():
    baseline_lines = [
        _dur_line(0.99, 4.21, "Take on me"),
        _dur_line(6.51, 9.73, "Take me on"),
        _dur_line(12.03, 15.25, "I'll be gone"),
        _dur_line(17.56, 21.42, "In a day or two"),
    ]
    raw_forced_lines = [
        _dur_line(1.12, 4.15, "Take on me"),
        _dur_line(6.48, 9.7, "Take me on"),
        _dur_line(11.72, 14.94, "I'll be gone"),
        _dur_line(17.19, 21.05, "In a day or two"),
    ]
    normalized_lines = [
        _dur_line(1.12, 4.15, "Take on me"),
        _dur_line(6.48, 7.02, "Take me on"),
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


def test_attempt_whisperx_forced_alignment_rejects_sparse_forced_when_baseline_onsets_are_better():
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
