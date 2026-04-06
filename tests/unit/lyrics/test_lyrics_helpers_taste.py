import pytest

from y2karaoke.core.components.lyrics import helpers as lh


def test_anchor_plain_text_lines_to_audio_window_rebalances_four_line_late_tail_chorus(
    monkeypatch,
):
    lines = lh._create_lines_from_plain_text(
        [
            "I heard you're back together and if that's true",
            "You'll just have to taste me when he's kissin' you",
            "If you want forever, I bet you do",
            "Just know you'll taste me too",
        ]
    )
    monkeypatch.setattr(
        "y2karaoke.core.components.alignment.alignment.detect_song_start",
        lambda *_: 0.0,
    )

    lh._anchor_plain_text_lines_to_audio_window(lines, 19.0, "vocals.wav")

    assert lines[0].start_time == pytest.approx(1.2, abs=0.05)
    assert 5.4 < lines[0].end_time < 5.8
    assert 5.8 < lines[1].start_time < 6.2
    assert 9.7 < lines[1].end_time < 10.2
    assert 10.0 < lines[2].start_time < 10.4
    assert 13.0 < lines[2].end_time < 13.4
    assert 13.2 < lines[3].start_time < 13.6
    assert lines[3].end_time > 18.8
