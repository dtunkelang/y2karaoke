from y2karaoke.core.visual.reconstruction import reconstruct_lyrics_from_visuals
from y2karaoke.core.models import TargetLine


def test_reconstruct_lyrics_single_frame_single_line():
    raw_frames = [
        {
            "time": 1.0,
            "words": [
                {"text": "Hello", "x": 10, "y": 100, "w": 50, "h": 20},
                {"text": "World", "x": 70, "y": 100, "w": 50, "h": 20},
            ],
        }
    ]
    lines = reconstruct_lyrics_from_visuals(raw_frames, 1.0)

    # Should detect 1 line
    assert len(lines) == 1
    assert lines[0].text == "Hello World"
    # Start time snapped from 1.0
    assert lines[0].start == 1.0
    # End time snapped from last seen (1.0) + 2.0 = 3.0
    assert lines[0].end == 3.0
    assert lines[0].visibility_start == 1.0
    assert lines[0].visibility_end == 1.0


def test_reconstruct_lyrics_detects_multiple_lines():
    raw_frames = [
        {
            "time": 1.0,
            "words": [
                {"text": "Line", "x": 10, "y": 100, "w": 50, "h": 20},
                {"text": "One", "x": 70, "y": 100, "w": 50, "h": 20},
                {"text": "Line", "x": 10, "y": 200, "w": 50, "h": 20},
                {"text": "Two", "x": 70, "y": 200, "w": 50, "h": 20},
            ],
        }
    ]
    lines = reconstruct_lyrics_from_visuals(raw_frames, 1.0)

    assert len(lines) == 2
    assert lines[0].text == "Line One"
    assert lines[1].text == "Line Two"


def test_reconstruct_lyrics_tracks_duration():
    # Line appears at 1.0, disappears after 2.0
    raw_frames = [
        {
            "time": 1.0,
            "words": [{"text": "Hello", "x": 10, "y": 100, "w": 50, "h": 20}],
        },
        {
            "time": 2.0,
            "words": [{"text": "Hello", "x": 10, "y": 100, "w": 50, "h": 20}],
        },
        {"time": 3.0, "words": []},  # Disappeared
        # Ensure it's committed (needs > 1.0s gap)
        {"time": 4.5, "words": []},
    ]
    lines = reconstruct_lyrics_from_visuals(raw_frames, 1.0)

    assert len(lines) == 1
    assert lines[0].start == 1.0
    # Last seen 2.0. End = 2.0 + 2.0 = 4.0
    assert lines[0].end == 4.0
    assert lines[0].visibility_start == 1.0
    assert lines[0].visibility_end == 2.0


def test_reconstruct_lyrics_deduplicates_similar_lines():
    # Line appears, disappears, reappears slightly shifted (noise/jitter)
    raw_frames = [
        {
            "time": 1.0,
            "words": [{"text": "Hello", "x": 10, "y": 100, "w": 50, "h": 20}],
        },
        {"time": 3.0, "words": []},  # Gap > 1.0s
        {
            "time": 3.5,
            "words": [
                {"text": "Hello", "x": 12, "y": 102, "w": 50, "h": 20}
            ],  # Shifted
        },
        {"time": 5.0, "words": []},  # Commit
    ]
    lines = reconstruct_lyrics_from_visuals(raw_frames, 1.0)

    # Should probably merge or deduplicate if they are considered "same occurrence"
    # But current logic only deduplicates if they overlap in time/space?
    # Logic: "if abs(ent["y"] - u["y"]) < 20 and abs(ent["first"] - u["first"]) < 2.0"
    # Here first=1.0 and first=3.5. Diff = 2.5. Not deduplicated by default logic.

    # Wait, if they are temporally distinct, they SHOULD stay distinct (repeated line).
    assert len(lines) == 2

    # Try one that SHOULD be deduplicated (close in time/space)
    raw_frames_dup = [
        {
            "time": 1.0,
            "words": [{"text": "Hello", "x": 10, "y": 100, "w": 50, "h": 20}],
        },
        {"time": 2.0, "words": []},  # Gap < 1.0s? No, just later frame.
        {
            "time": 2.5,  # first=2.5. Diff=1.5. < 2.0.
            "words": [{"text": "Hello", "x": 10, "y": 100, "w": 50, "h": 20}],
        },
        {"time": 4.0, "words": []},
    ]
    # This might fail if the first line is committed before the second appears?
    # Logic: "Commit lines that have disappeared ... if frame["time"] - on_screen[nt]["last"] > 1.0"
    # At 2.0, last=1.0. Diff=1.0. Not > 1.0.
    # At 2.5, "Hello" reappears. It matches `norm` key "y3_hello".
    # So it updates `last`.

    # If key matches, it's the SAME line object.

    lines_dup = reconstruct_lyrics_from_visuals(raw_frames_dup, 1.0)
    assert len(lines_dup) == 1
    assert lines_dup[0].start == 1.0
    # Last seen 2.5. End = 4.5.
    assert lines_dup[0].end == 4.5


def test_reconstruct_lyrics_filters_static_top_overlay_tokens():
    raw_frames = []
    for i in range(24):
        raw_frames.append(
            {
                "time": float(i),
                "words": [
                    {"text": "SingKIN", "x": 20, "y": 8, "w": 60, "h": 18},
                    {"text": "KARAO", "x": 90, "y": 8, "w": 60, "h": 18},
                    {"text": "White", "x": 20, "y": 120, "w": 55, "h": 20},
                    {"text": "shirt", "x": 85, "y": 120, "w": 50, "h": 20},
                ],
            }
        )

    lines = reconstruct_lyrics_from_visuals(raw_frames, 1.0)
    texts = [ln.text for ln in lines]
    assert all("SingKIN" not in t and "KARAO" not in t for t in texts)
    assert any("White shirt" in t for t in texts)


def test_reconstruct_lyrics_merges_apostrophe_fragment_tokens():
    raw_frames = [
        {
            "time": 10.0,
            "words": [
                {"text": "you", "x": 10, "y": 100, "w": 20, "h": 20},
                {"text": "'", "x": 34, "y": 100, "w": 5, "h": 20},
                {"text": "re", "x": 42, "y": 100, "w": 18, "h": 20},
                {"text": "here", "x": 66, "y": 100, "w": 35, "h": 20},
            ],
        }
    ]
    lines = reconstruct_lyrics_from_visuals(raw_frames, 1.0)
    assert len(lines) == 1
    assert lines[0].words == ["you're", "here"]


def test_reconstruct_lyrics_suppresses_short_duplicate_reentry_glitch():
    raw_frames = []
    for t in [10.0, 11.0, 12.0]:
        raw_frames.append(
            {
                "time": t,
                "words": [
                    {"text": "White", "x": 20, "y": 80, "w": 40, "h": 20},
                    {"text": "shirt", "x": 70, "y": 80, "w": 45, "h": 20},
                    {"text": "now", "x": 125, "y": 80, "w": 35, "h": 20},
                    {"text": "red", "x": 170, "y": 80, "w": 35, "h": 20},
                ],
            }
        )
    raw_frames.append({"time": 13.0, "words": []})
    raw_frames.append(
        {
            "time": 16.0,
            "words": [
                {"text": "White", "x": 20, "y": 80, "w": 40, "h": 20},
                {"text": "e", "x": 65, "y": 80, "w": 10, "h": 20},
                {"text": "shirt", "x": 80, "y": 80, "w": 45, "h": 20},
                {"text": "now", "x": 130, "y": 80, "w": 35, "h": 20},
                {"text": "red", "x": 175, "y": 80, "w": 35, "h": 20},
            ],
        }
    )
    raw_frames.append({"time": 18.2, "words": []})

    lines = reconstruct_lyrics_from_visuals(raw_frames, 1.0)
    texts = [ln.text for ln in lines]
    assert any("White shirt now red" in t for t in texts)
    assert not any("White e shirt now red" in t for t in texts)


def test_reconstruct_lyrics_keeps_legitimate_repeated_line():
    raw_frames = []
    for t in [10.0, 11.0, 12.0]:
        raw_frames.append(
            {
                "time": t,
                "words": [
                    {"text": "Bad", "x": 30, "y": 120, "w": 35, "h": 20},
                    {"text": "guy", "x": 75, "y": 120, "w": 35, "h": 20},
                ],
            }
        )
    raw_frames.append({"time": 13.5, "words": []})
    for t in [19.0, 20.0, 21.0]:
        raw_frames.append(
            {
                "time": t,
                "words": [
                    {"text": "Bad", "x": 30, "y": 120, "w": 35, "h": 20},
                    {"text": "guy", "x": 75, "y": 120, "w": 35, "h": 20},
                ],
            }
        )
    raw_frames.append({"time": 22.5, "words": []})

    lines = reconstruct_lyrics_from_visuals(raw_frames, 1.0)
    bad_guy_lines = [ln for ln in lines if ln.text == "Bad guy"]
    assert len(bad_guy_lines) == 2


def test_reconstruct_lyrics_keeps_concurrent_repeated_text_in_different_lanes():
    raw_frames = [
        {
            "time": 10.0,
            "words": [
                {"text": "Duh", "x": 30, "y": 100, "w": 35, "h": 20},
                {"text": "Duh", "x": 30, "y": 130, "w": 35, "h": 20},
            ],
        },
        {
            "time": 11.0,
            "words": [
                {"text": "Duh", "x": 30, "y": 100, "w": 35, "h": 20},
            ],
        },
        {
            "time": 12.0,
            "words": [
                {"text": "Duh", "x": 30, "y": 100, "w": 35, "h": 20},
            ],
        },
        {"time": 13.5, "words": []},
    ]

    lines = reconstruct_lyrics_from_visuals(raw_frames, 1.0)
    duh_lines = [ln for ln in lines if ln.text == "Duh"]
    assert len(duh_lines) == 2
    ys = sorted(int(ln.y) for ln in duh_lines)
    assert ys == [100, 130]


def test_reconstruct_lyrics_merges_short_same_lane_reentry_flicker():
    raw_frames = [
        {
            "time": 10.0,
            "words": [
                {"text": "Duh", "x": 30, "y": 90, "w": 35, "h": 20},
                {"text": "Duh", "x": 30, "y": 250, "w": 35, "h": 20},
            ],
        },
        {
            "time": 11.0,
            "words": [
                {"text": "Duh", "x": 30, "y": 90, "w": 35, "h": 20},
                {"text": "Duh", "x": 30, "y": 250, "w": 35, "h": 20},
            ],
        },
        {
            "time": 12.0,
            "words": [
                {"text": "Duh", "x": 30, "y": 250, "w": 35, "h": 20},
            ],
        },
        {
            "time": 13.0,
            "words": [
                {"text": "Duh", "x": 30, "y": 90, "w": 35, "h": 20},
            ],
        },
        {"time": 14.6, "words": []},
    ]

    lines = reconstruct_lyrics_from_visuals(raw_frames, 1.0)
    duh_lines = [ln for ln in lines if ln.text == "Duh"]
    assert len(duh_lines) == 2
    ys = sorted(int(ln.y) for ln in duh_lines)
    assert ys == [90, 250]


def test_reconstruct_lyrics_merges_same_lane_reentry_across_lane_bin_boundary():
    raw_frames = [
        {
            "time": 10.0,
            "words": [
                {"text": "Duh", "x": 30, "y": 89, "w": 35, "h": 20},
                {"text": "Duh", "x": 30, "y": 247, "w": 35, "h": 20},
            ],
        },
        {"time": 11.0, "words": [{"text": "Duh", "x": 30, "y": 247, "w": 35, "h": 20}]},
        {"time": 13.0, "words": [{"text": "Duh", "x": 30, "y": 93, "w": 35, "h": 20}]},
        {"time": 14.6, "words": []},
    ]

    lines = reconstruct_lyrics_from_visuals(raw_frames, 1.0)
    duh_lines = [ln for ln in lines if ln.text == "Duh"]
    assert len(duh_lines) == 2


def test_reconstruct_lyrics_merges_continuation_split_with_long_second_segment():
    raw_frames = [
        {
            "time": 10.0,
            "words": [
                {"text": "Duh", "x": 30, "y": 89, "w": 35, "h": 20},
                {"text": "Duh", "x": 30, "y": 247, "w": 35, "h": 20},
            ],
        },
        {
            "time": 11.0,
            "words": [
                {"text": "Duh", "x": 30, "y": 89, "w": 35, "h": 20},
                {"text": "Duh", "x": 30, "y": 247, "w": 35, "h": 20},
            ],
        },
        {
            "time": 12.0,
            "words": [
                {"text": "Duh", "x": 30, "y": 93, "w": 35, "h": 20},
                {"text": "Duh", "x": 30, "y": 247, "w": 35, "h": 20},
            ],
        },
        {
            "time": 13.0,
            "words": [
                {"text": "Duh", "x": 30, "y": 93, "w": 35, "h": 20},
            ],
        },
        {
            "time": 14.0,
            "words": [
                {"text": "Duh", "x": 30, "y": 93, "w": 35, "h": 20},
            ],
        },
        {"time": 15.6, "words": []},
    ]

    lines = reconstruct_lyrics_from_visuals(raw_frames, 1.0)
    duh_lines = [ln for ln in lines if ln.text == "Duh"]
    assert len(duh_lines) == 2
