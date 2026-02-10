import numpy as np
import pytest

from y2karaoke.core.timing_models import AudioFeatures
from y2karaoke.core.audio_analysis import (
    _find_silence_regions,
    _find_vocal_end,
    _find_vocal_start,
    _get_audio_features_cache_path,
    _load_audio_features_cache,
    _save_audio_features_cache,
)
import y2karaoke.core.timing_evaluator_correction as te_corr
import y2karaoke.core.timing_evaluator_core as te_core
from y2karaoke.core.timing_evaluator import (
    correct_line_timestamps,
    _find_best_onset_during_silence,
    _find_best_onset_for_phrase_end,
    _find_best_onset_proximity,
    _find_phrase_end,
    _check_pause_alignment,
    _calculate_pause_score,
)
from y2karaoke.core.models import Line, Word
import y2karaoke.core.timing_evaluator as te


def test_get_audio_features_cache_path(tmp_path):
    missing = tmp_path / "missing.wav"
    assert _get_audio_features_cache_path(str(missing)) is None

    vocals = tmp_path / "vocals.wav"
    vocals.write_bytes(b"data")
    cache_path = _get_audio_features_cache_path(str(vocals))
    assert cache_path is not None
    assert cache_path.endswith("vocals_audio_features.npz")


def test_save_and_load_audio_features_cache(tmp_path):
    cache_path = tmp_path / "features.npz"
    features = AudioFeatures(
        onset_times=np.array([0.1, 0.5, 1.0]),
        silence_regions=[(0.0, 0.2), (1.2, 1.5)],
        vocal_start=0.1,
        vocal_end=1.6,
        duration=2.0,
        energy_envelope=np.array([0.0, 0.2, 0.1]),
        energy_times=np.array([0.0, 0.1, 0.2]),
    )
    _save_audio_features_cache(str(cache_path), features)

    loaded = _load_audio_features_cache(str(cache_path))
    assert loaded is not None
    assert loaded.vocal_start == 0.1
    assert loaded.vocal_end == 1.6
    loaded_regions = [tuple(region) for region in loaded.silence_regions]
    assert loaded_regions == [(0.0, 0.2), (1.2, 1.5)]
    assert np.allclose(loaded.onset_times, features.onset_times)


def test_load_audio_features_cache_missing_file(tmp_path):
    missing = tmp_path / "missing.npz"
    assert _load_audio_features_cache(str(missing)) is None


def test_load_audio_features_cache_handles_exception(tmp_path, monkeypatch):
    cache_path = tmp_path / "features.npz"
    cache_path.write_bytes(b"invalid")

    def raise_error(*_args, **_kwargs):
        raise ValueError("bad npz")

    monkeypatch.setattr("numpy.load", raise_error)
    assert _load_audio_features_cache(str(cache_path)) is None


def test_save_audio_features_cache_handles_exception(tmp_path, monkeypatch):
    cache_path = tmp_path / "features.npz"
    features = AudioFeatures(
        onset_times=np.array([0.1]),
        silence_regions=[],
        vocal_start=0.1,
        vocal_end=0.2,
        duration=1.0,
        energy_envelope=np.array([0.1]),
        energy_times=np.array([0.0]),
    )

    def raise_error(*_args, **_kwargs):
        raise ValueError("bad save")

    monkeypatch.setattr("numpy.savez", raise_error)
    _save_audio_features_cache(str(cache_path), features)
    assert not cache_path.exists()


def test_find_silence_regions_handles_trailing_silence():
    is_silent = np.array([False, True, True, False, True, True])
    times = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    regions = _find_silence_regions(is_silent, times, min_duration=0.4)
    assert regions == [(0.5, 1.5), (2.0, 2.5)]


def test_find_silence_regions_skips_short_silence():
    is_silent = np.array([False, True, False])
    times = np.array([0.0, 0.2, 0.4])
    regions = _find_silence_regions(is_silent, times, min_duration=0.5)
    assert regions == []


def test_find_vocal_start_prefers_sustained_energy():
    onset_times = np.array([0.5, 1.0])
    rms_times = np.array([0.4, 0.5, 0.6, 1.0, 1.1, 1.2])
    rms = np.array([0.1, 0.1, 0.1, 0.6, 0.7, 0.6])
    start = _find_vocal_start(
        onset_times, rms, rms_times, threshold=0.5, min_duration=0.2
    )
    assert start == 1.0


def test_find_vocal_start_handles_onset_out_of_range():
    onset_times = np.array([5.0, 6.0])
    rms_times = np.array([0.0, 1.0, 2.0])
    rms = np.array([0.1, 0.2, 0.3])
    start = _find_vocal_start(
        onset_times, rms, rms_times, threshold=0.5, min_duration=0.2
    )
    assert start == 5.0


def test_find_vocal_start_handles_single_rms_frame():
    onset_times = np.array([0.1])
    rms_times = np.array([0.0])
    rms = np.array([1.0])
    start = _find_vocal_start(
        onset_times, rms, rms_times, threshold=0.5, min_duration=0.2
    )
    assert start == 0.1


def test_find_vocal_start_no_onsets_returns_zero():
    onset_times = np.array([])
    rms_times = np.array([0.0, 1.0])
    rms = np.array([0.2, 0.3])
    start = _find_vocal_start(
        onset_times, rms, rms_times, threshold=0.5, min_duration=0.2
    )
    assert start == 0.0


def _line(text: str, start: float, end: float) -> Line:
    return Line(words=[Word(text=text, start_time=start, end_time=end)])


def test_check_pause_alignment_spurious_gap_detected():
    lines = [_line("a", 0.0, 1.0), _line("b", 3.0, 4.0)]
    features = AudioFeatures(
        onset_times=np.array([]),
        silence_regions=[],
        vocal_start=0.0,
        vocal_end=4.0,
        duration=4.0,
        energy_envelope=np.ones(9),
        energy_times=np.linspace(0.0, 4.0, 9),
    )

    issues = _check_pause_alignment(lines, features)

    assert any(issue.issue_type == "spurious_gap" for issue in issues)
    assert any(issue.severity == "severe" for issue in issues)


def test_check_pause_alignment_split_phrase_detected():
    lines = [_line("a", 0.0, 1.0), _line("b", 4.0, 5.0)]
    features = AudioFeatures(
        onset_times=np.array([]),
        silence_regions=[],
        vocal_start=0.0,
        vocal_end=6.0,
        duration=6.0,
        energy_envelope=np.ones(13),
        energy_times=np.linspace(0.0, 6.0, 13),
    )

    issues = _check_pause_alignment(lines, features)

    assert any(issue.issue_type == "split_phrase" for issue in issues)


def test_check_pause_alignment_line_spans_silence_detected():
    lines = [_line("a", 0.0, 5.0)]
    features = AudioFeatures(
        onset_times=np.array([]),
        silence_regions=[(2.0, 3.5)],
        vocal_start=0.0,
        vocal_end=6.0,
        duration=6.0,
        energy_envelope=np.ones(13),
        energy_times=np.linspace(0.0, 6.0, 13),
    )

    issues = _check_pause_alignment(lines, features)

    assert any(issue.issue_type == "line_spans_silence" for issue in issues)


def test_check_pause_alignment_unexpected_pause_detected():
    lines = [_line("a", 0.0, 1.0)]
    features = AudioFeatures(
        onset_times=np.array([]),
        silence_regions=[(5.0, 7.5)],
        vocal_start=0.0,
        vocal_end=8.0,
        duration=8.0,
        energy_envelope=np.array([0.1]),
        energy_times=np.array([0.0]),
    )

    issues = _check_pause_alignment(lines, features)

    assert any(issue.issue_type == "unexpected_pause" for issue in issues)


def test_calculate_pause_score_matches_silence_with_gap():
    lines = [_line("a", 0.0, 1.5), _line("b", 5.2, 6.0)]
    features = AudioFeatures(
        onset_times=np.array([]),
        silence_regions=[(2.0, 5.0)],
        vocal_start=0.0,
        vocal_end=6.0,
        duration=6.0,
        energy_envelope=np.array([0.1]),
        energy_times=np.array([0.0]),
    )

    score = _calculate_pause_score(lines, features)

    assert score == 100.0


def test_calculate_pause_score_no_gap_for_silence():
    lines = [_line("a", 0.0, 4.0), _line("b", 4.1, 5.0)]
    features = AudioFeatures(
        onset_times=np.array([]),
        silence_regions=[(2.0, 5.0)],
        vocal_start=0.0,
        vocal_end=5.0,
        duration=5.0,
        energy_envelope=np.array([0.1]),
        energy_times=np.array([0.0]),
    )

    score = _calculate_pause_score(lines, features)

    assert score == 0.0


def test_fix_spurious_gaps_merges_large_gap_with_activity(monkeypatch):
    features = AudioFeatures(
        onset_times=np.array([]),
        silence_regions=[],
        vocal_start=0.0,
        vocal_end=10.0,
        duration=10.0,
        energy_envelope=np.ones(11),
        energy_times=np.linspace(0.0, 10.0, 11),
    )
    lines = [
        Line(words=[Word(text="a", start_time=0.0, end_time=1.0)]),
        Line(words=[Word(text="b", start_time=6.0, end_time=7.0)]),
    ]

    monkeypatch.setattr(
        te_corr, "_check_vocal_activity_in_range", lambda *_a, **_k: 0.9
    )
    monkeypatch.setattr(te_corr, "_check_for_silence_in_range", lambda *_a, **_k: False)

    merged, fixes = te.fix_spurious_gaps(lines, features)

    assert len(merged) == 1
    assert fixes


def test_find_best_onset_for_phrase_end_returns_none_when_not_silent(monkeypatch):
    features = AudioFeatures(
        onset_times=np.array([1.0, 2.0]),
        silence_regions=[],
        vocal_start=0.0,
        vocal_end=3.0,
        duration=3.0,
        energy_envelope=np.array([1.0, 1.0, 1.0]),
        energy_times=np.array([0.0, 1.0, 2.0]),
    )
    monkeypatch.setattr(
        te_corr, "_check_vocal_activity_in_range", lambda *_a, **_k: 0.9
    )

    onset = _find_best_onset_for_phrase_end(
        onset_times=np.array([1.0, 2.0]),
        line_start=2.5,
        prev_line_audio_end=0.0,
        audio_features=features,
    )

    assert onset is None


def test_find_best_onset_during_silence_requires_silence(monkeypatch):
    features = AudioFeatures(
        onset_times=np.array([1.5]),
        silence_regions=[],
        vocal_start=0.0,
        vocal_end=3.0,
        duration=3.0,
        energy_envelope=np.array([1.0, 1.0, 1.0]),
        energy_times=np.array([0.0, 1.0, 2.0]),
    )
    monkeypatch.setattr(
        te_corr, "_check_vocal_activity_in_range", lambda *_a, **_k: 0.9
    )

    onset = _find_best_onset_during_silence(
        onset_times=np.array([1.5]),
        line_start=2.0,
        prev_line_audio_end=0.0,
        max_correction=2.0,
        audio_features=features,
    )

    assert onset is None


def test_find_vocal_end_uses_last_activity():
    rms_times = np.array([0.0, 0.5, 1.0])
    rms = np.array([0.1, 0.6, 0.2])
    end = _find_vocal_end(rms, rms_times, threshold=0.5, min_silence=0.3)
    assert end == 0.5


def test_find_vocal_end_falls_back_to_end_time():
    rms_times = np.array([0.0, 0.5, 1.0])
    rms = np.array([0.1, 0.2, 0.2])
    end = _find_vocal_end(rms, rms_times, threshold=0.5, min_silence=0.3)
    assert end == 1.0


def test_find_vocal_end_empty_rms():
    rms_times = np.array([])
    rms = np.array([])
    end = _find_vocal_end(rms, rms_times, threshold=0.5, min_silence=0.3)
    assert end == 0.0


def test_find_best_onset_for_phrase_end_returns_onset():
    features = AudioFeatures(
        onset_times=np.array([]),
        silence_regions=[],
        vocal_start=0.0,
        vocal_end=4.0,
        duration=4.0,
        energy_envelope=np.array([0.0, 0.0, 0.0, 0.0]),
        energy_times=np.array([0.0, 1.0, 2.0, 3.0]),
    )
    onset_times = np.array([1.0, 2.0, 3.0])

    onset = _find_best_onset_for_phrase_end(
        onset_times, line_start=2.5, prev_line_audio_end=0.5, audio_features=features
    )

    assert onset == 1.0


def test_find_best_onset_proximity_prefers_closest():
    features = AudioFeatures(
        onset_times=np.array([]),
        silence_regions=[],
        vocal_start=0.0,
        vocal_end=4.0,
        duration=4.0,
        energy_envelope=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        energy_times=np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
    )
    onset_times = np.array([1.0, 2.0, 3.0])

    onset = _find_best_onset_proximity(
        onset_times, line_start=2.4, max_correction=1.5, audio_features=features
    )

    assert onset == 2.0


def test_find_best_onset_during_silence_returns_none_when_no_candidates():
    features = AudioFeatures(
        onset_times=np.array([]),
        silence_regions=[],
        vocal_start=0.0,
        vocal_end=4.0,
        duration=4.0,
        energy_envelope=np.array([0.0, 0.0, 0.0, 0.0]),
        energy_times=np.array([0.0, 1.0, 2.0, 3.0]),
    )
    onset_times = np.array([0.5])

    onset = _find_best_onset_during_silence(
        onset_times,
        line_start=3.0,
        prev_line_audio_end=2.8,
        max_correction=0.1,
        audio_features=features,
    )

    assert onset is None


def test_correct_line_timestamps_applies_offset(monkeypatch):
    features = AudioFeatures(
        onset_times=np.array([2.0]),
        silence_regions=[],
        vocal_start=0.0,
        vocal_end=4.0,
        duration=4.0,
        energy_envelope=np.array([1.0, 1.0, 1.0, 1.0]),
        energy_times=np.array([0.0, 1.0, 2.0, 3.0]),
    )
    line = Line(words=[Word(text="hi", start_time=1.0, end_time=1.2)])

    responses = iter([1.0, 0.0, 1.0])

    def fake_activity(*_args, **_kwargs):
        return next(responses, 1.0)

    monkeypatch.setattr(te_corr, "_check_vocal_activity_in_range", fake_activity)
    monkeypatch.setattr(
        te_corr, "_find_best_onset_for_phrase_end", lambda *_a, **_k: 2.0
    )
    monkeypatch.setattr(te_corr, "_find_phrase_end", lambda *_a, **_k: 5.0)

    corrected, corrections = correct_line_timestamps(
        [line], features, max_correction=3.0
    )

    assert corrected[0].words[0].start_time == pytest.approx(2.0)
    assert corrections


def test_correct_line_timestamps_handles_empty_lines():
    features = AudioFeatures(
        onset_times=np.array([]),
        silence_regions=[],
        vocal_start=0.0,
        vocal_end=0.0,
        duration=0.0,
        energy_envelope=np.array([]),
        energy_times=np.array([]),
    )

    corrected, corrections = correct_line_timestamps([], features)

    assert corrected == []
    assert corrections == []


def test_correct_line_timestamps_skips_empty_word_line():
    features = AudioFeatures(
        onset_times=np.array([]),
        silence_regions=[],
        vocal_start=0.0,
        vocal_end=1.0,
        duration=1.0,
        energy_envelope=np.array([1.0]),
        energy_times=np.array([0.0]),
    )
    line = Line(words=[])

    corrected, corrections = correct_line_timestamps([line], features)

    assert corrected[0] is line
    assert corrections == []


def test_correct_line_timestamps_no_singing_uses_silence_path(monkeypatch):
    features = AudioFeatures(
        onset_times=np.array([1.0]),
        silence_regions=[],
        vocal_start=0.0,
        vocal_end=2.0,
        duration=2.0,
        energy_envelope=np.array([0.0, 0.0]),
        energy_times=np.array([0.0, 1.0]),
    )
    line = Line(words=[Word(text="hi", start_time=1.0, end_time=1.2)])

    monkeypatch.setattr(
        te_corr, "_check_vocal_activity_in_range", lambda *_a, **_k: 0.0
    )
    monkeypatch.setattr(
        te_corr, "_find_best_onset_during_silence", lambda *_a, **_k: None
    )
    monkeypatch.setattr(te_corr, "_find_phrase_end", lambda *_a, **_k: 2.0)

    corrected, corrections = correct_line_timestamps(
        [line], features, max_correction=1.0
    )

    assert corrected[0] is line
    assert corrections == []


def test_correct_line_timestamps_skips_small_offset(monkeypatch):
    features = AudioFeatures(
        onset_times=np.array([1.1]),
        silence_regions=[],
        vocal_start=0.0,
        vocal_end=2.0,
        duration=2.0,
        energy_envelope=np.array([1.0, 1.0]),
        energy_times=np.array([0.0, 1.0]),
    )
    line = Line(words=[Word(text="hi", start_time=1.0, end_time=1.2)])

    responses = iter([1.0, 1.0, 1.0])

    monkeypatch.setattr(
        te, "_check_vocal_activity_in_range", lambda *_a, **_k: next(responses, 1.0)
    )
    monkeypatch.setattr(te_corr, "_find_best_onset_proximity", lambda *_a, **_k: 1.1)
    monkeypatch.setattr(te_corr, "_find_phrase_end", lambda *_a, **_k: 2.0)

    corrected, corrections = correct_line_timestamps(
        [line], features, max_correction=1.0
    )

    assert corrected[0] is line
    assert corrections == []


def test_find_phrase_end_returns_silence_start():
    features = AudioFeatures(
        onset_times=np.array([]),
        silence_regions=[],
        vocal_start=0.0,
        vocal_end=4.0,
        duration=4.0,
        energy_envelope=np.array([1.0, 0.0, 0.0, 1.0, 1.0]),
        energy_times=np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
    )

    end = _find_phrase_end(0.0, 4.0, features, min_silence_duration=1.0)

    assert end == 1.0


def test_find_phrase_end_uses_max_when_no_silence():
    features = AudioFeatures(
        onset_times=np.array([]),
        silence_regions=[],
        vocal_start=0.0,
        vocal_end=4.0,
        duration=4.0,
        energy_envelope=np.array([1.0, 1.0, 1.0, 1.0]),
        energy_times=np.array([0.0, 1.0, 2.0, 3.0]),
    )

    end = _find_phrase_end(0.0, 3.0, features, min_silence_duration=1.0)

    assert end == 3.0


def test_find_phrase_end_handles_trailing_silence():
    features = AudioFeatures(
        onset_times=np.array([]),
        silence_regions=[],
        vocal_start=0.0,
        vocal_end=3.0,
        duration=3.0,
        energy_envelope=np.array([1.0, 0.0, 0.0, 0.0]),
        energy_times=np.array([0.0, 1.0, 2.0, 3.0]),
    )

    end = _find_phrase_end(0.0, 3.0, features, min_silence_duration=1.0)

    assert end == 1.0


def test_find_phrase_end_stops_at_max_end_time():
    features = AudioFeatures(
        onset_times=np.array([]),
        silence_regions=[],
        vocal_start=0.0,
        vocal_end=2.0,
        duration=2.0,
        energy_envelope=np.array([1.0, 1.0, 1.0]),
        energy_times=np.array([0.0, 1.0, 2.0]),
    )

    end = _find_phrase_end(0.0, 0.5, features, min_silence_duration=1.0)

    assert end == 0.5


def test_find_phrase_end_ignores_short_silence():
    features = AudioFeatures(
        onset_times=np.array([]),
        silence_regions=[],
        vocal_start=0.0,
        vocal_end=3.0,
        duration=3.0,
        energy_envelope=np.array([1.0, 0.0, 1.0, 1.0]),
        energy_times=np.array([0.0, 1.0, 2.0, 3.0]),
    )

    end = _find_phrase_end(0.0, 3.0, features, min_silence_duration=2.0)

    assert end == 3.0


def test_find_phrase_end_trailing_silence_too_short_returns_max():
    features = AudioFeatures(
        onset_times=np.array([]),
        silence_regions=[],
        vocal_start=0.0,
        vocal_end=2.0,
        duration=2.0,
        energy_envelope=np.array([1.0, 0.0, 0.0]),
        energy_times=np.array([0.0, 1.0, 2.0]),
    )

    end = _find_phrase_end(0.0, 2.0, features, min_silence_duration=2.0)

    assert end == 2.0


def test_find_best_onset_proximity_applies_early_penalty():
    features = AudioFeatures(
        onset_times=np.array([]),
        silence_regions=[],
        vocal_start=0.0,
        vocal_end=6.0,
        duration=6.0,
        energy_envelope=np.array([0.0, 0.0, 0.0, 0.0]),
        energy_times=np.array([0.0, 1.0, 2.0, 3.0]),
    )
    onset = _find_best_onset_proximity(
        onset_times=np.array([2.0, 4.0]),
        line_start=4.0,
        max_correction=3.0,
        audio_features=features,
    )

    assert onset == 4.0


def test_correct_line_timestamps_prefers_phrase_end_when_silence_after(monkeypatch):
    features = AudioFeatures(
        onset_times=np.array([2.0]),
        silence_regions=[],
        vocal_start=0.0,
        vocal_end=4.0,
        duration=4.0,
        energy_envelope=np.array([1.0, 1.0, 1.0, 1.0]),
        energy_times=np.array([0.0, 1.0, 2.0, 3.0]),
    )
    line = Line(words=[Word(text="hi", start_time=1.0, end_time=1.2)])

    responses = iter([1.0, 0.0, 1.0])

    monkeypatch.setattr(
        te_corr, "_check_vocal_activity_in_range", lambda *_a, **_k: next(responses)
    )

    called = {"phrase_end": 0}

    def fake_phrase_end(*_args, **_kwargs):
        called["phrase_end"] += 1
        return 2.0

    monkeypatch.setattr(
        te_corr, "_find_best_onset_for_phrase_end", lambda *_a, **_k: 2.0
    )
    monkeypatch.setattr(te_corr, "_find_best_onset_proximity", lambda *_a, **_k: None)
    monkeypatch.setattr(te_corr, "_find_phrase_end", fake_phrase_end)

    corrected, corrections = correct_line_timestamps([line], features)

    assert corrected[0].words[0].start_time == 2.0
    assert corrections
    assert called["phrase_end"] > 0


def test_correct_line_timestamps_uses_proximity_when_silence_before(monkeypatch):
    features = AudioFeatures(
        onset_times=np.array([1.5]),
        silence_regions=[],
        vocal_start=0.0,
        vocal_end=4.0,
        duration=4.0,
        energy_envelope=np.array([1.0, 1.0, 1.0, 1.0]),
        energy_times=np.array([0.0, 1.0, 2.0, 3.0]),
    )
    line = Line(words=[Word(text="hi", start_time=1.0, end_time=1.2)])

    responses = iter([1.0, 1.0, 0.0])

    monkeypatch.setattr(
        te_corr, "_check_vocal_activity_in_range", lambda *_a, **_k: next(responses)
    )
    monkeypatch.setattr(
        te_corr, "_find_best_onset_for_phrase_end", lambda *_a, **_k: None
    )
    monkeypatch.setattr(te_corr, "_find_best_onset_proximity", lambda *_a, **_k: 1.5)
    monkeypatch.setattr(te_corr, "_find_phrase_end", lambda *_a, **_k: 2.0)

    corrected, corrections = correct_line_timestamps([line], features)

    assert corrected[0].words[0].start_time == 1.5
    assert corrections


def test_correct_line_timestamps_applies_global_shift():
    features = AudioFeatures(
        onset_times=np.array([]),
        silence_regions=[],
        vocal_start=9.0,
        vocal_end=20.0,
        duration=20.0,
        energy_envelope=np.array([1.0]),
        energy_times=np.array([0.0]),
    )
    lines = [
        Line(words=[Word(text="first", start_time=0.0, end_time=1.0)]),
        Line(words=[Word(text="second", start_time=5.0, end_time=6.0)]),
    ]

    corrected, corrections = correct_line_timestamps(lines, features)

    assert corrected[0].words[0].start_time == pytest.approx(9.0)
    assert corrected[1].words[0].start_time == pytest.approx(14.0)
    assert any("Global shift" in note for note in corrections)
