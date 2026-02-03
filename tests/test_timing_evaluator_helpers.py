import numpy as np
import pytest

from y2karaoke.core.timing_evaluator import (
    AudioFeatures,
    correct_line_timestamps,
    _find_best_onset_during_silence,
    _find_best_onset_for_phrase_end,
    _find_best_onset_proximity,
    _find_phrase_end,
    _find_silence_regions,
    _find_vocal_end,
    _find_vocal_start,
    _get_audio_features_cache_path,
    _load_audio_features_cache,
    _save_audio_features_cache,
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

    monkeypatch.setattr(te, "_check_vocal_activity_in_range", fake_activity)
    monkeypatch.setattr(te, "_find_best_onset_for_phrase_end", lambda *_a, **_k: 2.0)
    monkeypatch.setattr(te, "_find_phrase_end", lambda *_a, **_k: 5.0)

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

    monkeypatch.setattr(te, "_check_vocal_activity_in_range", lambda *_a, **_k: 0.0)
    monkeypatch.setattr(te, "_find_best_onset_during_silence", lambda *_a, **_k: None)
    monkeypatch.setattr(te, "_find_phrase_end", lambda *_a, **_k: 2.0)

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
    monkeypatch.setattr(te, "_find_best_onset_proximity", lambda *_a, **_k: 1.1)
    monkeypatch.setattr(te, "_find_phrase_end", lambda *_a, **_k: 2.0)

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
