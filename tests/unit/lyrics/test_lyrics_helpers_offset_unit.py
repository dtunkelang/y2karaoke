import pytest

from y2karaoke.core.components.lyrics import helpers as lh


def test_detect_and_apply_offset_skips_suspicious_delta(monkeypatch):
    monkeypatch.setattr(
        "y2karaoke.core.components.alignment.alignment.detect_song_start", lambda _: 5.0
    )

    line_timings = [(1.0, "Line")]
    updated, offset = lh._detect_and_apply_offset(
        "vocals.wav", line_timings, lyrics_offset=None
    )

    assert offset == 0.0
    assert updated[0][0] == pytest.approx(1.0)


def test_detect_and_apply_offset_respects_manual(monkeypatch):
    monkeypatch.setattr(
        "y2karaoke.core.components.alignment.alignment.detect_song_start", lambda _: 5.0
    )

    line_timings = [(1.0, "Line")]
    updated, offset = lh._detect_and_apply_offset(
        "vocals.wav", line_timings, lyrics_offset=-1.0
    )

    assert offset == pytest.approx(-1.0)
    assert updated[0][0] == pytest.approx(0.0)


def test_detect_and_apply_offset_skips_large_delta(monkeypatch):
    monkeypatch.setattr(
        "y2karaoke.core.components.alignment.alignment.detect_song_start",
        lambda _: 100.0,
    )

    line_timings = [(1.0, "Line")]
    updated, offset = lh._detect_and_apply_offset(
        "vocals.wav", line_timings, lyrics_offset=None
    )

    assert offset == pytest.approx(0.0)
    assert updated == line_timings


def test_detect_and_apply_offset_uses_second_line_after_long_interjection_gap(
    monkeypatch,
):
    monkeypatch.setattr(
        "y2karaoke.core.components.alignment.alignment.detect_song_start",
        lambda _: 27.98,
    )

    line_timings = [
        (13.42, "Yeah"),
        (26.95, "I've been tryna call"),
    ]
    updated, offset = lh._detect_and_apply_offset(
        "vocals.wav", line_timings, lyrics_offset=None
    )

    assert offset == pytest.approx(0.618, abs=0.01)
    assert updated[0][0] == pytest.approx(14.04, abs=0.01)
    assert updated[1][0] == pytest.approx(27.57, abs=0.01)
