from y2karaoke.core.visual.reconstruction_frame_accumulation import (
    TrackedLine,
    accumulate_persistent_lines_from_frames,
)


def _entry(text: str, *, y: int = 100, lane: int = 2, t: float = 0.0) -> dict:
    words = text.split()
    return {
        "text": text,
        "words": words,
        "first": t,
        "last": t,
        "y": y,
        "lane": lane,
        "w_rois": [(10 * i, y, 8, 12) for i, _ in enumerate(words)],
        "brightness": 200.0,
    }


def test_tracked_line_falls_back_to_observed_candidate_on_weak_consensus() -> None:
    entries = [
        _entry("and i feel my rhythm", t=0.0),
        _entry("and we dance your rhythm", t=0.1),
        _entry("and you dance their rhythm", t=0.2),
        _entry("and they dance our rhythm", t=0.3),
    ]
    track = TrackedLine(entries[0], "track_1")
    track.entries = entries

    voted = track.get_voted_text()

    # Per-word voting would synthesize "and i dance my rhythm" (never observed).
    assert voted != "and i dance my rhythm"
    assert voted in {e["text"] for e in entries}


def test_tracked_line_keeps_stable_majority_vote_when_consensus_is_strong() -> None:
    entries = [
        _entry("i'm levitating you moonlight", t=0.0),
        _entry("i'm levitating you moonlight", t=0.1),
        _entry("im levitating you moonlight", t=0.2),
        _entry("i'm levitating you moonlight", t=0.3),
    ]
    track = TrackedLine(entries[0], "track_1")
    track.entries = entries

    voted = track.get_voted_text()

    assert voted == "i'm levitating you moonlight"


def test_tracked_line_to_dict_includes_reconstruction_meta() -> None:
    entries = [
        _entry("and i feel my rhythm", t=0.0),
        _entry("and we dance your rhythm", t=0.1),
        _entry("and you dance their rhythm", t=0.2),
        _entry("and they dance our rhythm", t=0.3),
    ]
    track = TrackedLine(entries[0], "track_1")
    track.entries = entries

    out = track.to_dict()
    meta = out.get("reconstruction_meta")

    assert isinstance(meta, dict)
    assert "uncertainty_score" in meta
    assert "selected_text_support_ratio" in meta
    assert "position_support_mean" in meta
    assert "used_observed_fallback" in meta


def test_tracked_line_does_not_extend_end_on_dim_ghost_updates_after_visible() -> None:
    track = TrackedLine(_entry("we'll be counting stars", t=23.5), "track_1")
    track.visible_yet = True
    track.vis_count = 1
    track.first_visible = 23.5
    track.last_visible_seen = 23.5

    track.update(_entry("we'll be counting stars", t=24.0), 24.0, is_visible=True)
    track.update(_entry("we'll be counting stars", t=34.0), 34.0, is_visible=False)
    track.update(_entry("we'll be counting stars", t=35.0), 35.0, is_visible=False)

    out = track.to_dict()
    assert out["last"] == 24.0
    assert track.last_seen == 24.0


def test_accumulator_splits_line_across_long_dim_ghost_gap() -> None:
    def _word(text: str, t: float, brightness: float) -> dict:
        return {
            "text": text,
            "x": 100 if text == "counting" else 180,
            "y": 100,
            "w": 60,
            "h": 16,
            "brightness": brightness,
        }

    raw_frames = []
    # First visible lyric burst.
    for i in range(0, 5):
        t = round(i * 0.1, 2)
        raw_frames.append(
            {
                "time": t,
                "words": [
                    _word("counting", t, 200.0),
                    _word("stars", t, 200.0),
                ],
            }
        )
    # Long ghost OCR tail: text still detected but too dim to be visible.
    for i in range(5, 22):
        t = round(i * 0.1, 2)
        raw_frames.append(
            {
                "time": t,
                "words": [
                    _word("counting", t, 60.0),
                    _word("stars", t, 60.0),
                ],
            }
        )
    # Second visible burst should become a new persistent line.
    for i in range(22, 27):
        t = round(i * 0.1, 2)
        raw_frames.append(
            {
                "time": t,
                "words": [
                    _word("counting", t, 200.0),
                    _word("stars", t, 200.0),
                ],
            }
        )

    committed = accumulate_persistent_lines_from_frames(
        raw_frames,
        filter_static_overlay_words=lambda frames: frames,
        visual_fps=10.0,
    )

    counting = [c for c in committed if "counting" in c.get("text", "")]
    assert len(counting) == 2
    counting.sort(key=lambda c: c["first"])
    assert counting[0]["first"] == 0.0
    assert counting[0]["last"] <= 0.4
    assert counting[1]["first"] < 2.2  # dim ghost tail may spawn a pre-visible track
    assert counting[1]["first_visible"] >= 2.19
