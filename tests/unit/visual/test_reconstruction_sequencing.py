from y2karaoke.core.visual.reconstruction_pipeline import (
    _order_visual_block_locally,
    _suppress_repeated_short_fragment_clusters,
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


def test_local_band_orders_full_line_before_fragment_subphrase_same_lane() -> None:
    block = [
        _line(10.00, 11.20, 1, "we could be"),
        _line(10.05, 11.40, 1, "dreaming about the thing that we could be"),
    ]

    ordered = _order_visual_block_locally(block)

    assert [ln["text"] for ln in ordered] == [
        "dreaming about the thing that we could be",
        "we could be",
    ]


def test_local_band_keeps_non_fragment_same_lane_chronology() -> None:
    block = [
        _line(10.00, 11.00, 1, "counting stars"),
        _line(10.05, 11.10, 1, "lately I been"),
    ]

    ordered = _order_visual_block_locally(block)

    assert [ln["text"] for ln in ordered] == [
        "counting stars",
        "lately I been",
    ]


def test_local_band_demotes_fragment_even_when_fragment_lane_sorts_earlier() -> None:
    block = [
        _line(10.00, 11.10, 0, "we could be"),
        _line(10.02, 11.30, 1, "dreaming about the things that we could be"),
    ]

    ordered = _order_visual_block_locally(block)

    assert [ln["text"] for ln in ordered] == [
        "dreaming about the things that we could be",
        "we could be",
    ]


def test_local_band_demotes_split_word_fragment_line() -> None:
    block = [
        _line(10.00, 10.80, 0, "con ting"),
        _line(10.02, 11.00, 1, "counting dollars we'll be counting stars"),
    ]

    ordered = _order_visual_block_locally(block)

    assert [ln["text"] for ln in ordered] == [
        "counting dollars we'll be counting stars",
        "con ting",
    ]


def test_local_band_demotes_single_token_fragment_inside_fuller_line() -> None:
    block = [
        _line(10.00, 10.75, 0, "dollar"),
        _line(10.03, 10.98, 1, "said no more counting dollar"),
    ]

    ordered = _order_visual_block_locally(block)

    assert [ln["text"] for ln in ordered] == [
        "said no more counting dollar",
        "dollar",
    ]


def test_suppress_repeated_short_fragment_clusters_drops_supported_fragments() -> None:
    lines = [
        _line(0.0, 0.7, 0, "dollar"),
        _line(0.2, 1.4, 1, "said no more counting dollars"),
        _line(3.0, 3.6, 0, "dollar"),
        _line(3.1, 4.0, 1, "we'll be counting dollars"),
        _line(6.0, 7.0, 1, "take that money watch it burn"),
    ]

    kept = _suppress_repeated_short_fragment_clusters(lines)

    assert [ln["text"] for ln in kept] == [
        "said no more counting dollars",
        "we'll be counting dollars",
        "take that money watch it burn",
    ]
