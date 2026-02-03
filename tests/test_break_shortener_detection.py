import types

import numpy as np

import y2karaoke.core.break_shortener as bs


def test_detect_instrumental_breaks_finds_silent_regions(monkeypatch):
    fake_librosa = types.SimpleNamespace()

    def fake_load(path, sr=22050):
        return np.zeros(100), sr

    def fake_rms(y, frame_length, hop_length):
        return np.array([[0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0]])

    def fake_frames_to_time(frames, sr=22050, hop_length=512):
        return np.array([0, 1, 2, 3, 4, 5, 6], dtype=float)

    fake_librosa.load = fake_load
    fake_librosa.feature = types.SimpleNamespace(rms=fake_rms)
    fake_librosa.frames_to_time = fake_frames_to_time

    monkeypatch.setitem(__import__("sys").modules, "librosa", fake_librosa)

    breaks = bs.detect_instrumental_breaks("vocals.wav", min_break_duration=2.0)

    assert len(breaks) == 2
    assert breaks[0].start == 0
    assert breaks[0].end == 2
    assert breaks[1].start == 4
    assert breaks[1].end == 6


def test_shorten_break_respects_crossfade_and_beat_alignment(monkeypatch):
    break_info = bs.InstrumentalBreak(start=0.0, end=20.0)

    def fake_find_beat_near(path, target_time, **kwargs):
        if target_time < 10:
            return 10.0
        return 11.0

    monkeypatch.setattr(bs, "find_beat_near", fake_find_beat_near)

    cut_start, cut_end, after_start, time_removed = bs.shorten_break(
        "inst.wav",
        break_info,
        keep_start=5.0,
        keep_end=3.0,
        crossfade_duration=2.0,
        align_to_beats=True,
    )

    assert cut_start == 10.0
    assert cut_end == 12.0
    assert after_start == 12.0
    assert time_removed == 0.0


def test_shorten_break_returns_none_when_break_too_short():
    break_info = bs.InstrumentalBreak(start=0.0, end=5.0)

    cut_start, cut_end, after_start, time_removed = bs.shorten_break(
        "inst.wav",
        break_info,
        keep_start=3.0,
        keep_end=2.0,
        crossfade_duration=2.0,
        align_to_beats=False,
    )

    assert cut_start is None
    assert cut_end is None
    assert after_start == break_info.end
    assert time_removed == 0.0
