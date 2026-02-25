from y2karaoke.core.visual.reconstruction_block_first import (
    apply_block_first_ordering_to_persistent_entries,
    build_persistent_visibility_blocks,
)
from y2karaoke.core.visual.reconstruction_block_first_frames import (
    _FrameRow,
    _FrameState,
    _split_block_on_row_cycle_resets,
)


def _ent(text, *, first, first_visible, last, y):
    return {
        "text": text,
        "words": [{"text": w} for w in text.split()],
        "first": first,
        "first_visible": first_visible,
        "last": last,
        "y": y,
        "w_rois": [],
        "reconstruction_meta": {},
    }


def test_build_persistent_visibility_blocks_groups_screen_rows():
    entries = [
        _ent("row2", first=12.0, first_visible=10.0, last=18.0, y=120),
        _ent("row1", first=11.0, first_visible=10.1, last=18.1, y=80),
        _ent("row3", first=14.0, first_visible=10.2, last=18.2, y=160),
        _ent("next1", first=24.0, first_visible=23.0, last=45.0, y=90),
        _ent("next2", first=25.0, first_visible=23.2, last=45.2, y=130),
    ]
    blocks = build_persistent_visibility_blocks(entries)
    assert len(blocks) >= 2
    assert [r.entry["text"] for r in blocks[0].rows] == ["row1", "row2", "row3"]
    assert [r.entry["text"] for r in blocks[1].rows] == ["next1", "next2"]


def test_apply_block_first_ordering_to_persistent_entries_interleaves_singletons():
    entries = [
        _ent("block row b", first=12.5, first_visible=10.0, last=18.0, y=120),
        _ent("block row a", first=11.5, first_visible=10.0, last=18.0, y=80),
        _ent("singleton", first=19.0, first_visible=19.0, last=20.0, y=70),
        _ent("later row", first=24.0, first_visible=23.0, last=30.0, y=80),
        _ent("later row 2", first=25.0, first_visible=23.0, last=30.0, y=120),
    ]

    out = apply_block_first_ordering_to_persistent_entries(entries)

    assert [e["text"] for e in out] == [
        "block row a",
        "block row b",
        "singleton",
        "later row",
        "later row 2",
    ]


def test_split_block_on_row_cycle_resets_splits_two_pass_block():
    frames = []
    t = 100.0
    # Two highlight sweeps over the same 4 visible rows.
    seq = (
        ([0] * 8)
        + ([1] * 8)
        + ([2] * 8)
        + ([3] * 8)
        + ([0] * 8)
        + ([1] * 8)
        + ([2] * 8)
        + ([3] * 8)
    )
    for best in seq:
        rows = []
        for idx, y in enumerate([20.0, 90.0, 160.0, 230.0]):
            bright = 90.0 if idx == best else 55.0
            rows.append(
                _FrameRow(
                    time=t,
                    y=y,
                    lane=0,
                    text=f"row{idx}",
                    words=[f"row{idx}"],
                    w_rois=[],
                    is_visible=True,
                    brightness=bright,
                )
            )
        frames.append(_FrameState(time=t, rows=rows))
        t += 0.2

    blocks = _split_block_on_row_cycle_resets(frames)
    assert len(blocks) == 2
    assert len(blocks[0]) >= 20
    assert len(blocks[1]) >= 20
