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


def test_reconstruct_uses_spacing_to_split_confusable_glyph_runs() -> None:
    raw_frames = [
        {
            "time": 50.0,
            "words": [
                {"text": "Oh", "x": 20, "y": 100, "w": 26, "h": 18},
                {"text": "loh", "x": 49, "y": 100, "w": 20, "h": 18},
                {"text": "l", "x": 71, "y": 100, "w": 10, "h": 18},
                # Wider gap indicates a word boundary, not an intra-word character gap.
                {"text": "oh", "x": 94, "y": 100, "w": 22, "h": 18},
                {"text": "l", "x": 118, "y": 100, "w": 10, "h": 18},
            ],
        },
        {"time": 51.4, "words": []},
    ]

    lines = reconstruct_lyrics_from_visuals(raw_frames, 1.0)
    assert len(lines) == 1
    assert lines[0].text in {
        "Oh I oh I oh",
        "Oh I oh I",
        "Oh I oh",
        "oh I oh I",
        "oh I oh",
        "Oh I oh oh I",
        "oh I oh oh I",
    }


def test_reconstruct_merges_same_lane_continuation_reentry_for_long_lines() -> None:
    raw_frames = [
        {
            "time": 30.0,
            "words": [
                {"text": "a", "x": 20, "y": 100, "w": 14, "h": 18},
                {"text": "conversation", "x": 38, "y": 100, "w": 98, "h": 18},
                {"text": "with", "x": 140, "y": 100, "w": 32, "h": 18},
                {"text": "just", "x": 176, "y": 100, "w": 28, "h": 18},
                {"text": "me", "x": 208, "y": 100, "w": 22, "h": 18},
            ],
        },
        {
            "time": 30.4,
            "words": [
                {"text": "a", "x": 20, "y": 100, "w": 14, "h": 18},
                {"text": "conversation", "x": 38, "y": 100, "w": 98, "h": 18},
                {"text": "with", "x": 140, "y": 100, "w": 32, "h": 18},
                {"text": "just", "x": 176, "y": 100, "w": 28, "h": 18},
                {"text": "me", "x": 208, "y": 100, "w": 22, "h": 18},
            ],
        },
        {"time": 31.7, "words": []},
        {
            "time": 31.9,
            "words": [
                {"text": "a", "x": 20, "y": 101, "w": 14, "h": 18},
                {"text": "conversation", "x": 38, "y": 101, "w": 98, "h": 18},
                {"text": "with", "x": 140, "y": 101, "w": 32, "h": 18},
                {"text": "just", "x": 176, "y": 101, "w": 28, "h": 18},
                {"text": "me", "x": 208, "y": 101, "w": 22, "h": 18},
            ],
        },
        {"time": 33.1, "words": []},
    ]

    lines = reconstruct_lyrics_from_visuals(raw_frames, 3.0)
    assert len(lines) == 1
    assert lines[0].text == "a conversation with just me"


def test_reconstruct_expands_overlapped_same_text_repetitions() -> None:
    body_top = [
        {"text": "I'm", "x": 20, "y": 20, "w": 30, "h": 18},
        {"text": "in", "x": 52, "y": 20, "w": 16, "h": 18},
        {"text": "love", "x": 70, "y": 20, "w": 34, "h": 18},
        {"text": "with", "x": 108, "y": 20, "w": 30, "h": 18},
        {"text": "your", "x": 142, "y": 20, "w": 30, "h": 18},
        {"text": "body", "x": 176, "y": 20, "w": 32, "h": 18},
    ]
    body_mid = [{**w, "y": 160} for w in body_top]
    refrain_mid_a = [
        {"text": "Oh", "x": 20, "y": 90, "w": 20, "h": 18},
        {"text": "I", "x": 44, "y": 90, "w": 10, "h": 18},
        {"text": "oh", "x": 58, "y": 90, "w": 20, "h": 18},
        {"text": "I", "x": 82, "y": 90, "w": 10, "h": 18},
    ]
    refrain_mid_b = [
        {"text": "Oh", "x": 20, "y": 90, "w": 20, "h": 18},
        {"text": "I", "x": 44, "y": 90, "w": 10, "h": 18},
        {"text": "oh", "x": 58, "y": 90, "w": 20, "h": 18},
        {"text": "I", "x": 82, "y": 90, "w": 10, "h": 18},
        {"text": "oh", "x": 96, "y": 90, "w": 20, "h": 18},
        {"text": "I", "x": 120, "y": 90, "w": 10, "h": 18},
    ]
    refrain_mid_c = [
        {"text": "Oh", "x": 20, "y": 90, "w": 20, "h": 18},
        {"text": "I", "x": 44, "y": 90, "w": 10, "h": 18},
        {"text": "oh", "x": 58, "y": 90, "w": 20, "h": 18},
        {"text": "oh", "x": 82, "y": 90, "w": 20, "h": 18},
        {"text": "oh", "x": 96, "y": 90, "w": 20, "h": 18},
        {"text": "I", "x": 120, "y": 90, "w": 10, "h": 18},
    ]

    raw_frames = [
        {"time": 70.0, "words": body_mid},
        {"time": 80.0, "words": body_mid + body_top},
        {"time": 82.6, "words": body_mid + body_top + refrain_mid_a},
        {"time": 86.0, "words": body_mid + body_top + refrain_mid_b},
        {"time": 90.0, "words": body_mid + body_top + refrain_mid_c},
        {"time": 92.6, "words": []},
    ]

    lines = reconstruct_lyrics_from_visuals(raw_frames, 3.0)
    body_lines = [ln for ln in lines if "love with your body" in ln.text.lower()]
    assert len(body_lines) >= 2
