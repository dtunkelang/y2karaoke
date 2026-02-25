from y2karaoke.core.visual.reconstruction_lane_merge import merge_dim_fade_in_fragments


def _entry(
    text: str,
    *,
    first: float,
    last: float,
    y: int = 20,
    lane: int = 0,
    avg_brightness: float = 180.0,
) -> dict:
    words = text.split()
    return {
        "text": text,
        "words": words,
        "first": first,
        "last": last,
        "y": y,
        "lane": lane,
        "avg_brightness": avg_brightness,
        "w_rois": [(10 * i, y, 8, 12) for i, _ in enumerate(words)],
    }


def _same_lane(a: dict, b: dict) -> bool:
    return abs(float(a.get("y", 0.0)) - float(b.get("y", 0.0))) <= 18.0


def test_merge_dim_fade_keeps_distinct_same_lane_lines() -> None:
    early = _entry(
        "But baby I've been",
        first=19.84,
        last=20.00,
        y=20,
        avg_brightness=180.0,
    )
    longer = _entry(
        "Lately I've been l've been",
        first=10.08,
        last=19.76,
        y=21,
        avg_brightness=210.0,
    )

    out = merge_dim_fade_in_fragments([longer, early], is_same_lane=_same_lane)

    assert len(out) == 2
    assert any(e["text"] == "But baby I've been" for e in out)


def test_merge_dim_fade_merges_true_same_line_fade_fragment() -> None:
    dim_frag = _entry(
        "We'll be counting stars",
        first=24.8,
        last=25.5,
        y=89,
        avg_brightness=70.0,
    )
    bright = _entry(
        "We'll be counting stars",
        first=23.52,
        last=27.08,
        y=88,
        avg_brightness=160.0,
    )

    out = merge_dim_fade_in_fragments([bright, dim_frag], is_same_lane=_same_lane)

    assert len(out) == 1
    assert out[0]["first"] <= 23.52
    assert out[0]["last"] >= 27.08
