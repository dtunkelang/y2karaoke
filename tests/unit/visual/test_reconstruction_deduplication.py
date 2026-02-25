from y2karaoke.core.visual.reconstruction_deduplication import (
    deduplicate_persistent_lines,
)


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
        "first_visible": first if visible_yet else first,
        "last": last,
        "y": y,
        "lane": lane,
        "visible_yet": visible_yet,
        "vis_count": 3 if visible_yet else 0,
        "w_rois": [(10 * i, y, 8, 12) for i, _ in enumerate(words)],
    }


def test_dedup_does_not_merge_later_nonvisible_ghost_tail() -> None:
    visible = _entry(
        "We'll be counting stars",
        first=23.5,
        last=26.1,
        y=88,
        visible_yet=True,
    )
    ghost_tail = _entry(
        "We'll be counting stars",
        first=27.1,
        last=45.1,
        y=88,
        visible_yet=False,
    )

    out = deduplicate_persistent_lines([visible, ghost_tail])

    assert len(out) == 2
    assert any(e["last"] == 26.1 for e in out)
    assert any(e["first"] == 27.1 and e["visible_yet"] is False for e in out)


def test_dedup_still_merges_nearby_visible_duplicates() -> None:
    a = _entry("We'll be counting stars", first=23.5, last=26.1, y=88, visible_yet=True)
    b = _entry("We'll be counting stars", first=23.8, last=26.0, y=90, visible_yet=True)

    out = deduplicate_persistent_lines([a, b])

    assert len(out) == 1
    assert out[0]["first"] == 23.5
    assert out[0]["last"] == 26.1
