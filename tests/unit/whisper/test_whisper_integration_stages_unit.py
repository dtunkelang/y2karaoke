import numpy as np
import pytest

from y2karaoke.core.components.whisper import whisper_integration_stages as stages
from y2karaoke.core.models import Line, Word
from y2karaoke.core.components.alignment.timing_models import (
    AudioFeatures,
    TranscriptionWord,
)


def test_normalize_repeated_line_durations_noop_when_flag_disabled(monkeypatch):
    monkeypatch.delenv("Y2K_REPEAT_DURATION_NORMALIZE", raising=False)
    lines = [
        Line(
            words=[
                Word(text="I", start_time=10.0, end_time=10.2),
                Word(text="said", start_time=10.2, end_time=10.7),
            ]
        ),
        Line(
            words=[
                Word(text="I", start_time=20.0, end_time=20.2),
                Word(text="said", start_time=20.2, end_time=21.4),
            ]
        ),
    ]

    out = stages._normalize_repeated_line_durations(lines)

    assert out == lines


def test_normalize_repeated_line_durations_scales_repeated_lines_when_flag_enabled(
    monkeypatch,
):
    monkeypatch.setenv("Y2K_REPEAT_DURATION_NORMALIZE", "1")
    lines = [
        Line(
            words=[
                Word(text="I", start_time=10.0, end_time=10.2),
                Word(text="said", start_time=10.2, end_time=10.6),
            ]
        ),
        Line(
            words=[
                Word(text="I", start_time=20.0, end_time=20.3),
                Word(text="said", start_time=20.3, end_time=21.5),
            ]
        ),
        Line(
            words=[
                Word(text="next", start_time=21.9, end_time=22.4),
            ]
        ),
    ]

    out = stages._normalize_repeated_line_durations(lines)

    assert out[0].start_time == pytest.approx(10.0)
    assert out[0].end_time == pytest.approx(10.9)
    assert out[1].start_time == pytest.approx(20.0)
    assert out[1].end_time == pytest.approx(21.14)
    assert (
        abs(
            (out[0].end_time - out[0].start_time)
            - (out[1].end_time - out[1].start_time)
        )
        < 0.3
    )
    assert out[1].end_time < lines[2].start_time - 0.05
    assert out[2] == lines[2]


def test_enforce_mapped_line_stage_invariants_runs_two_cycles():
    lines = [Line(words=[Word(text="a", start_time=1.0, end_time=1.4)])]
    whisper_words = [TranscriptionWord(text="a", start=1.0, end=1.3, probability=0.9)]
    calls: list[str] = []

    def _monotonic(in_lines, _words):
        calls.append("monotonic")
        return in_lines

    def _resolve(in_lines):
        calls.append("resolve")
        return in_lines

    out = stages._enforce_mapped_line_stage_invariants(
        lines,
        whisper_words,
        enforce_monotonic_line_starts_whisper_fn=_monotonic,
        resolve_line_overlaps_fn=_resolve,
    )

    assert out == lines
    assert calls == ["monotonic", "resolve", "monotonic", "resolve"]


def test_run_mapped_line_postpasses_handles_snap_fallback_and_audio_corrections():
    lines = [
        Line(words=[Word(text="one", start_time=10.0, end_time=10.5)]),
        Line(words=[Word(text="two", start_time=11.0, end_time=11.5)]),
    ]
    whisper_words = [
        TranscriptionWord(text="one", start=10.0, end=10.5, probability=0.9),
        TranscriptionWord(text="two", start=11.0, end=11.5, probability=0.9),
    ]
    audio_features = AudioFeatures(
        onset_times=np.array([], dtype=float),
        silence_regions=[],
        vocal_start=0.0,
        vocal_end=20.0,
        duration=20.0,
        energy_envelope=np.array([], dtype=float),
        energy_times=np.array([], dtype=float),
    )
    call_log: list[str] = []
    pull_counts = [1, 2]

    def _mark(tag):
        def _fn(*args, **kwargs):
            call_log.append(tag)
            return args[0]

        return _fn

    def _pull_lines_forward(in_lines, _audio_features):
        call_log.append("pull_lines_forward")
        count = pull_counts.pop(0) if pull_counts else 0
        return in_lines, count

    def _snap(lines_in, all_words_in):
        call_log.append("snap_no_kwargs")
        return lines_in

    out, corrections = stages._run_mapped_line_postpasses(
        mapped_lines=lines,
        mapped_lines_set={0, 1},
        all_words=whisper_words,
        transcription=[],
        audio_features=audio_features,
        vocals_path="vocals.wav",
        epitran_lang="eng-Latn",
        corrections=[],
        interpolate_unmatched_lines_fn=_mark("interpolate"),
        refine_unmatched_lines_with_onsets_fn=_mark("refine"),
        shift_repeated_lines_to_next_whisper_fn=_mark("shift_repeat"),
        extend_line_to_trailing_whisper_matches_fn=_mark("extend"),
        pull_late_lines_to_matching_segments_fn=_mark("pull_late"),
        retime_short_interjection_lines_fn=_mark("retime_interjection"),
        snap_first_word_to_whisper_onset_fn=_snap,
        pull_lines_forward_for_continuous_vocals_fn=_pull_lines_forward,
        enforce_monotonic_line_starts_whisper_fn=_mark("monotonic"),
        resolve_line_overlaps_fn=_mark("resolve"),
    )

    assert out == lines
    assert "interpolate" in call_log
    assert "shift_repeat" in call_log
    assert call_log.count("pull_lines_forward") == 2
    assert call_log.count("snap_no_kwargs") >= 3
    assert any(
        "Pulled 1 line(s) forward for continuous vocals" in c for c in corrections
    )
    assert any(
        "Applied 2 late audio onset/silence adjustment(s)" in c for c in corrections
    )


def test_run_mapped_line_postpasses_no_audio_calls_snap_with_max_shift():
    lines = [Line(words=[Word(text="one", start_time=10.0, end_time=10.5)])]
    whisper_words = [
        TranscriptionWord(text="one", start=10.0, end=10.5, probability=0.9)
    ]
    snap_kwargs: list[dict[str, float]] = []
    call_log: list[str] = []

    def _mark(tag):
        def _fn(*args, **kwargs):
            call_log.append(tag)
            return args[0]

        return _fn

    def _snap(lines_in, all_words_in, **kwargs):
        assert all_words_in == whisper_words
        snap_kwargs.append(dict(kwargs))
        return lines_in

    out, corrections = stages._run_mapped_line_postpasses(
        mapped_lines=lines,
        mapped_lines_set={0},
        all_words=whisper_words,
        transcription=[],
        audio_features=None,
        vocals_path="vocals.wav",
        epitran_lang="eng-Latn",
        corrections=[],
        interpolate_unmatched_lines_fn=_mark("interpolate"),
        refine_unmatched_lines_with_onsets_fn=_mark("refine"),
        shift_repeated_lines_to_next_whisper_fn=_mark("shift_repeat"),
        extend_line_to_trailing_whisper_matches_fn=_mark("extend"),
        pull_late_lines_to_matching_segments_fn=_mark("pull_late"),
        retime_short_interjection_lines_fn=_mark("retime_interjection"),
        snap_first_word_to_whisper_onset_fn=_snap,
        pull_lines_forward_for_continuous_vocals_fn=lambda ml, _af: (ml, 0),
        enforce_monotonic_line_starts_whisper_fn=_mark("monotonic"),
        resolve_line_overlaps_fn=_mark("resolve"),
    )

    assert out == lines
    assert corrections == []
    assert call_log.count("pull_late") == 2
    assert snap_kwargs == [{}, {"max_shift": 2.5}]


def test_shift_weak_opening_lines_past_phrase_carryover_moves_line_when_gap_is_tight():
    lines = [
        Line(
            words=[
                Word(text="lights", start_time=150.59, end_time=156.41),
            ]
        ),
        Line(
            words=[
                Word(text="No,", start_time=156.57, end_time=156.9),
                Word(text="I", start_time=156.9, end_time=157.2),
                Word(text="can't", start_time=157.2, end_time=157.5),
                Word(text="sleep", start_time=157.5, end_time=157.8),
                Word(text="until", start_time=157.8, end_time=158.2),
                Word(text="I", start_time=158.2, end_time=158.5),
                Word(text="feel", start_time=158.5, end_time=158.8),
                Word(text="your", start_time=158.8, end_time=159.1),
                Word(text="touch", start_time=159.1, end_time=159.56),
            ]
        ),
        Line(words=[Word(text="next", start_time=160.79, end_time=161.2)]),
    ]
    audio_features = AudioFeatures(
        onset_times=np.array([156.69, 157.11, 157.29, 157.43], dtype=float),
        silence_regions=[],
        vocal_start=0.0,
        vocal_end=200.0,
        duration=200.0,
        energy_envelope=np.array([], dtype=float),
        energy_times=np.array([], dtype=float),
    )

    adjusted, count = stages._shift_weak_opening_lines_past_phrase_carryover(
        lines, audio_features
    )

    assert count == 1
    assert adjusted[1].start_time == 157.11
    assert adjusted[1].end_time == pytest.approx(160.10, abs=1e-2)


def test_shift_weak_opening_lines_past_phrase_carryover_keeps_supported_line():
    lines = [
        Line(words=[Word(text="lights", start_time=120.59, end_time=121.97)]),
        Line(
            words=[
                Word(text="No,", start_time=122.84, end_time=123.17),
                Word(text="I", start_time=123.17, end_time=123.5),
                Word(text="can't", start_time=123.5, end_time=123.85),
                Word(text="sleep", start_time=123.85, end_time=124.2),
                Word(text="until", start_time=124.2, end_time=124.6),
                Word(text="I", start_time=124.6, end_time=124.9),
                Word(text="feel", start_time=124.9, end_time=125.2),
                Word(text="your", start_time=125.2, end_time=125.5),
                Word(text="touch", start_time=125.5, end_time=126.19),
            ]
        ),
        Line(words=[Word(text="next", start_time=128.09, end_time=128.8)]),
    ]
    whisper_words = [
        TranscriptionWord(text="No", start=122.92, end=123.10, probability=0.99),
        TranscriptionWord(text="sleep", start=123.75, end=124.15, probability=0.99),
        TranscriptionWord(text="feel", start=124.85, end=125.10, probability=0.99),
        TranscriptionWord(text="touch", start=125.45, end=126.15, probability=0.99),
    ]
    audio_features = AudioFeatures(
        onset_times=np.array([123.54, 123.89, 124.32], dtype=float),
        silence_regions=[],
        vocal_start=0.0,
        vocal_end=200.0,
        duration=200.0,
        energy_envelope=np.array([], dtype=float),
        energy_times=np.array([], dtype=float),
    )

    adjusted, count = stages._shift_weak_opening_lines_past_phrase_carryover(
        lines,
        audio_features,
        whisper_words,
    )

    assert count == 0
    assert adjusted[1].start_time == pytest.approx(122.84, abs=1e-2)
