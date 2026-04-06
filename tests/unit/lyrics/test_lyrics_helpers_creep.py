import pytest

from y2karaoke.core.components.lyrics import helpers as lh


def test_anchor_plain_text_lines_to_audio_window_rebalances_four_line_staggered_chorus(
    monkeypatch,
):
    lines = lh._create_lines_from_plain_text(
        [
            "But I'm a creep",
            "I'm a weirdo",
            "What the hell am I doing here?",
            "I don't belong here",
        ]
    )
    monkeypatch.setattr(
        "y2karaoke.core.components.alignment.alignment.detect_song_start",
        lambda *_: 0.0,
    )

    lh._anchor_plain_text_lines_to_audio_window(lines, 20.0, "vocals.wav")

    assert lines[0].start_time == pytest.approx(1.25, abs=0.05)
    assert 2.7 < lines[0].end_time < 3.2
    assert 5.0 < lines[1].start_time < 5.5
    assert 7.3 < lines[1].end_time < 7.8
    assert 9.5 < lines[2].start_time < 10.1
    assert 14.3 < lines[2].end_time < 14.9
    assert 16.6 < lines[3].start_time < 17.3
    assert lines[3].end_time > 19.7
