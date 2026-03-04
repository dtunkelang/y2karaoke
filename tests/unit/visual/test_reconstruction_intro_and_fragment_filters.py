from y2karaoke.core.visual.reconstruction import reconstruct_lyrics_from_visuals


def test_reconstruct_lyrics_filters_intro_title_card_artifacts():
    raw_frames = [
        {
            "time": 3.0,
            "words": [
                {"text": "KAI", "x": 30, "y": 120, "w": 30, "h": 20},
                {"text": "AOK", "x": 70, "y": 120, "w": 35, "h": 20},
            ],
        },
        {
            "time": 5.0,
            "words": [
                {"text": "SingKING", "x": 20, "y": 20, "w": 80, "h": 18},
                {"text": "KARAOKE", "x": 105, "y": 20, "w": 90, "h": 18},
            ],
        },
        {
            "time": 17.0,
            "words": [
                {"text": "White", "x": 25, "y": 130, "w": 45, "h": 20},
                {"text": "shirt", "x": 80, "y": 130, "w": 45, "h": 20},
                {"text": "now", "x": 130, "y": 130, "w": 35, "h": 20},
                {"text": "red", "x": 172, "y": 130, "w": 35, "h": 20},
            ],
        },
        {
            "time": 18.0,
            "words": [
                {"text": "White", "x": 25, "y": 130, "w": 45, "h": 20},
                {"text": "shirt", "x": 80, "y": 130, "w": 45, "h": 20},
                {"text": "now", "x": 130, "y": 130, "w": 35, "h": 20},
                {"text": "red", "x": 172, "y": 130, "w": 35, "h": 20},
            ],
        },
        {"time": 19.5, "words": []},
    ]

    lines = reconstruct_lyrics_from_visuals(raw_frames, 1.0)
    texts = [ln.text for ln in lines]
    assert "KAI AOK" not in texts
    assert all("SingKING" not in t and "KARAOKE" not in t for t in texts)
    assert all("Connell" not in t for t in texts)
    assert any("White shirt now red" == t for t in texts)


def test_reconstruct_lyrics_keeps_early_short_real_lyric_when_no_intro_gap():
    raw_frames = [
        {"time": 5.0, "words": [{"text": "Duh", "x": 30, "y": 120, "w": 35, "h": 20}]},
        {"time": 6.0, "words": [{"text": "Duh", "x": 30, "y": 120, "w": 35, "h": 20}]},
        {"time": 7.2, "words": []},
        {
            "time": 8.0,
            "words": [
                {"text": "White", "x": 20, "y": 140, "w": 45, "h": 20},
                {"text": "shirt", "x": 75, "y": 140, "w": 45, "h": 20},
                {"text": "now", "x": 125, "y": 140, "w": 35, "h": 20},
            ],
        },
        {
            "time": 9.0,
            "words": [
                {"text": "White", "x": 20, "y": 140, "w": 45, "h": 20},
                {"text": "shirt", "x": 75, "y": 140, "w": 45, "h": 20},
                {"text": "now", "x": 125, "y": 140, "w": 35, "h": 20},
            ],
        },
        {"time": 10.5, "words": []},
    ]

    lines = reconstruct_lyrics_from_visuals(raw_frames, 1.0)
    texts = [ln.text for ln in lines]
    assert any(t == "Duh" for t in texts)


def test_reconstruct_lyrics_filters_early_legal_credit_lines() -> None:
    raw_frames = [
        {
            "time": 6.0,
            "words": [
                {"text": "O'Connell", "x": 120, "y": 205, "w": 90, "h": 20},
                {"text": "O'Connell", "x": 215, "y": 205, "w": 92, "h": 20},
            ],
        },
        {
            "time": 9.0,
            "words": [
                {"text": "Universal", "x": 95, "y": 235, "w": 100, "h": 20},
                {"text": "Music", "x": 202, "y": 235, "w": 62, "h": 20},
                {"text": "Kobalt", "x": 270, "y": 235, "w": 76, "h": 20},
                {"text": "Publishing", "x": 352, "y": 235, "w": 104, "h": 20},
                {"text": "Ltd", "x": 462, "y": 235, "w": 34, "h": 20},
            ],
        },
        {
            "time": 17.0,
            "words": [
                {"text": "White", "x": 25, "y": 130, "w": 45, "h": 20},
                {"text": "shirt", "x": 80, "y": 130, "w": 45, "h": 20},
                {"text": "now", "x": 130, "y": 130, "w": 35, "h": 20},
                {"text": "red", "x": 172, "y": 130, "w": 35, "h": 20},
            ],
        },
        {
            "time": 18.0,
            "words": [
                {"text": "White", "x": 25, "y": 130, "w": 45, "h": 20},
                {"text": "shirt", "x": 80, "y": 130, "w": 45, "h": 20},
                {"text": "now", "x": 130, "y": 130, "w": 35, "h": 20},
                {"text": "red", "x": 172, "y": 130, "w": 35, "h": 20},
            ],
        },
        {"time": 19.4, "words": []},
    ]

    lines = reconstruct_lyrics_from_visuals(raw_frames, 1.0)
    texts = [ln.text for ln in lines]
    assert all("Connell" not in t for t in texts)
    assert all("Universal" not in t and "Kobalt" not in t for t in texts)
    assert "White shirt now red" in texts


def test_reconstruct_lyrics_filters_recurrent_bottom_fragment_family():
    raw_frames = [
        {
            "time": 20.0,
            "words": [
                {"text": "White", "x": 30, "y": 120, "w": 45, "h": 20},
                {"text": "shirt", "x": 80, "y": 120, "w": 45, "h": 20},
                {"text": "now", "x": 130, "y": 120, "w": 35, "h": 20},
            ],
        },
        {
            "time": 21.0,
            "words": [{"text": "KIN", "x": 180, "y": 316, "w": 35, "h": 18}],
        },
        {
            "time": 24.0,
            "words": [{"text": "KIR", "x": 190, "y": 318, "w": 35, "h": 18}],
        },
        {
            "time": 27.0,
            "words": [{"text": "KAPA", "x": 195, "y": 320, "w": 40, "h": 18}],
        },
        {
            "time": 29.0,
            "words": [{"text": "KAR", "x": 188, "y": 317, "w": 35, "h": 18}],
        },
        {
            "time": 31.0,
            "words": [
                {"text": "My", "x": 30, "y": 150, "w": 28, "h": 20},
                {"text": "bloody", "x": 65, "y": 150, "w": 55, "h": 20},
                {"text": "nose", "x": 125, "y": 150, "w": 40, "h": 20},
            ],
        },
        {"time": 33.0, "words": []},
    ]

    lines = reconstruct_lyrics_from_visuals(raw_frames, 1.0)
    texts = [ln.text for ln in lines]
    assert "White shirt now" in texts
    assert "My bloody nose" in texts
    assert all(t not in texts for t in ["KIN", "KIR", "KAPA", "KAR"])
