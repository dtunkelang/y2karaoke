from y2karaoke.core.visual.reconstruction_reentry import merge_short_same_lane_reentries


def _entry(
    text: str,
    *,
    first: float,
    last: float,
    y: int = 100,
    lane: int = 2,
    visible_yet: bool = True,
) -> dict:
    words = text.split()
    return {
        "text": text,
        "words": words,
        "first": first,
        "last": last,
        "y": y,
        "lane": lane,
        "visible_yet": visible_yet,
        "w_rois": [(10 * i, y, 8, 12) for i, _ in enumerate(words)],
    }


def _same_lane(a: dict, b: dict) -> bool:
    return a.get("lane") == b.get("lane")


def _never_short_refrain(_: dict) -> bool:
    return False


def test_merge_short_same_lane_reentries_does_not_merge_nonvisible_tail() -> None:
    prev = _entry(
        "We'll be counting stars",
        first=23.5,
        last=26.1,
        lane=2,
        y=88,
        visible_yet=True,
    )
    ghost_tail = _entry(
        "We'll be counting stars",
        first=27.1,
        last=45.1,
        lane=2,
        y=89,
        visible_yet=False,
    )

    out = merge_short_same_lane_reentries(
        [prev, ghost_tail],
        is_same_lane=_same_lane,
        is_short_refrain_entry=_never_short_refrain,
    )

    assert len(out) == 2
    assert out[0]["last"] == 26.1
    assert out[1]["first"] == 27.1


def test_merge_short_same_lane_reentries_still_merges_true_short_reentry() -> None:
    prev = _entry("losing sleep", first=10.0, last=11.0, lane=2, visible_yet=True)
    reentry = _entry("losing sleep", first=11.6, last=12.0, lane=2, visible_yet=True)

    out = merge_short_same_lane_reentries(
        [prev, reentry],
        is_same_lane=_same_lane,
        is_short_refrain_entry=_never_short_refrain,
    )

    assert len(out) == 1
    assert out[0]["first"] == 10.0
    assert out[0]["last"] == 12.0
