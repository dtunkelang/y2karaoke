import pytest

import y2karaoke.core.components.lyrics.helpers as lh


def test_anchor_plain_text_lines_to_audio_window_rebalances_repeated_hook_bridge(
    monkeypatch,
):
    lines = lh._create_lines_from_plain_text(
        [
            "Hey, I just met you",
            "And this is crazy",
            "But here's my number",
            "So call me maybe",
            "It's hard to look right",
            "At you baby",
            "But here's my number, so call me maybe",
        ]
    )
    monkeypatch.setattr(
        "y2karaoke.core.components.alignment.alignment.detect_song_start",
        lambda *_: 2.2,
    )

    lh._anchor_plain_text_lines_to_audio_window(lines, 28.2, "vocals.wav")

    assert lines[0].start_time == pytest.approx(2.2, abs=0.08)
    assert 6.0 < lines[1].start_time < 6.2
    assert 9.2 < lines[2].start_time < 9.4
    assert 12.4 < lines[3].start_time < 12.7
    assert 15.6 < lines[4].start_time < 15.9
    assert 19.5 < lines[5].start_time < 19.8
    assert 22.1 < lines[6].start_time < 22.4
    assert lines[6].end_time > 27.2
