from y2karaoke.core.visual.reconstruction import reconstruct_lyrics_from_visuals


def test_reconstruct_collapses_overlapping_short_refrain_noise() -> None:
    raw_frames = [
        {
            "time": 80.0,
            "words": [
                {"text": "Oh", "x": 25, "y": 110, "w": 28, "h": 18},
                {"text": "I", "x": 58, "y": 110, "w": 18, "h": 18},
                {"text": "oh", "x": 82, "y": 110, "w": 24, "h": 18},
                {"text": "I", "x": 112, "y": 110, "w": 18, "h": 18},
            ],
        },
        {
            "time": 81.0,
            "words": [
                {"text": "Oh", "x": 25, "y": 111, "w": 28, "h": 18},
                {"text": "loh", "x": 58, "y": 111, "w": 24, "h": 18},
                {"text": "oh", "x": 86, "y": 111, "w": 24, "h": 18},
                {"text": "l", "x": 116, "y": 111, "w": 14, "h": 18},
            ],
        },
        {"time": 82.6, "words": []},
    ]

    lines = reconstruct_lyrics_from_visuals(raw_frames, 1.0)
    assert len(lines) == 1
    assert lines[0].text == "Oh I oh I"


def test_reconstruct_keeps_separated_short_refrain_repetitions() -> None:
    raw_frames = [
        {
            "time": 80.0,
            "words": [
                {"text": "Oh", "x": 25, "y": 110, "w": 28, "h": 18},
                {"text": "I", "x": 58, "y": 110, "w": 18, "h": 18},
                {"text": "oh", "x": 82, "y": 110, "w": 24, "h": 18},
                {"text": "I", "x": 112, "y": 110, "w": 18, "h": 18},
            ],
        },
        {"time": 82.6, "words": []},
        {
            "time": 90.0,
            "words": [
                {"text": "Oh", "x": 25, "y": 110, "w": 28, "h": 18},
                {"text": "I", "x": 58, "y": 110, "w": 18, "h": 18},
                {"text": "oh", "x": 82, "y": 110, "w": 24, "h": 18},
                {"text": "I", "x": 112, "y": 110, "w": 18, "h": 18},
            ],
        },
        {"time": 92.6, "words": []},
    ]

    lines = reconstruct_lyrics_from_visuals(raw_frames, 1.0)
    assert len([ln for ln in lines if ln.text == "Oh I oh I"]) == 2
