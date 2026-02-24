from y2karaoke.core.visual.reconstruction_frame_accumulation import TrackedLine


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
