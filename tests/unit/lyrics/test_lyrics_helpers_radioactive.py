import pytest

import y2karaoke.core.components.lyrics.helpers as lh


def test_anchor_plain_text_lines_to_audio_window_rebalances_alternating_hook_chorus(
    monkeypatch,
):
    lines = lh._create_lines_from_plain_text(
        [
            "I'm waking up, I feel it in my bones",
            "Enough to make my system blow",
            "Welcome to the new age, to the new age",
            "Welcome to the new age, to the new age",
            "Whoa-oh-oh-oh, oh",
            "Whoa-oh-oh-oh",
            "I'm radioactive, radioactive",
            "Whoa-oh-oh-oh, oh",
            "Whoa-oh-oh-oh",
            "I'm radioactive, radioactive",
        ]
    )
    monkeypatch.setattr(
        "y2karaoke.core.components.alignment.alignment.detect_song_start",
        lambda *_: 0.4,
    )

    lh._anchor_plain_text_lines_to_audio_window(lines, 29.7, "vocals.wav")

    assert lines[0].start_time == pytest.approx(0.36, abs=0.08)
    assert 5.1 < lines[1].start_time < 5.6
    assert 9.1 < lines[2].start_time < 9.6
    assert 12.7 < lines[3].start_time < 13.1
    assert 16.7 < lines[4].start_time < 17.2
    assert 18.3 < lines[5].start_time < 18.7
    assert 20.0 < lines[6].start_time < 20.5
    assert 24.8 < lines[8].start_time < 25.2
    assert lines[9].end_time > 29.4
