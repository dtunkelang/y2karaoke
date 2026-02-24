from y2karaoke.core.visual.reconstruction_pipeline import (
    _sequence_by_visual_neighborhood,
)


def _line(first: float, last: float, lane: int, text: str) -> dict:
    return {
        "first": first,
        "last": last,
        "first_visible": first,
        "lane": lane,
        "text": text,
        "words": text.split(),
    }


def test_sequence_preserves_chronology_for_long_transitive_overlap_chain() -> None:
    # A-B and B-C overlap strongly, creating a transitive component spanning a long interval.
    # Lane-sorting the whole component would scramble chronology and can hurt refinement.
    lines = [
        _line(0.0, 10.0, 2, "A"),
        _line(3.0, 13.0, 0, "B"),
        _line(6.0, 16.0, 1, "C"),
    ]

    ordered = _sequence_by_visual_neighborhood(lines)

    assert [ln["text"] for ln in ordered] == ["A", "B", "C"]


def test_sequence_still_lane_sorts_small_local_simultaneous_block() -> None:
    lines = [
        _line(10.0, 12.0, 1, "bottom"),
        _line(10.1, 12.1, 0, "top"),
    ]

    ordered = _sequence_by_visual_neighborhood(lines)

    assert [ln["text"] for ln in ordered] == ["top", "bottom"]


def test_sequence_global_fallback_on_pathological_large_long_block() -> None:
    # Four transitive-overlap lines spanning >20s should force chronology-only fallback.
    lines = [
        _line(0.0, 10.0, 3, "A"),
        _line(6.0, 16.0, 0, "B"),
        _line(12.0, 22.0, 2, "C"),
        _line(18.0, 28.0, 1, "D"),
    ]

    ordered = _sequence_by_visual_neighborhood(lines)

    assert [ln["text"] for ln in ordered] == ["A", "B", "C", "D"]
