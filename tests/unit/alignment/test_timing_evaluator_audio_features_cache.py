import numpy as np

from y2karaoke.core.components.alignment.timing_models import AudioFeatures
from y2karaoke.core.audio_analysis import (
    _find_silence_regions,
    _find_vocal_start,
    _get_audio_features_cache_path,
    _load_audio_features_cache,
    _save_audio_features_cache,
)


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
