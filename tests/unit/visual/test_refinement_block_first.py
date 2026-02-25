from y2karaoke.core.models import TargetLine
from y2karaoke.core.visual.refinement_block_first import (
    apply_block_first_prototype_ordering,
    build_visibility_blocks,
)


def _line(
    idx: int,
    text: str,
    *,
    start: float,
    y: float,
    vs: float,
    ve: float,
) -> TargetLine:
    return TargetLine(
        line_index=idx,
        start=start,
        end=start + 1.0,
        text=text,
        words=text.split(),
        y=y,
        visibility_start=vs,
        visibility_end=ve,
    )


def test_build_visibility_blocks_groups_shared_screen_rows():
    lines = [
        _line(1, "row2", start=12.5, y=120, vs=10.0, ve=18.0),
        _line(2, "row1", start=11.8, y=80, vs=10.1, ve=18.2),
        _line(3, "row3", start=14.0, y=160, vs=10.2, ve=17.9),
        _line(4, "next1", start=24.0, y=90, vs=23.0, ve=45.0),
        _line(5, "next2", start=25.0, y=130, vs=23.1, ve=45.1),
    ]

    blocks = build_visibility_blocks(lines)

    assert len(blocks) >= 2
    assert [r.line.text for r in blocks[0].rows] == ["row1", "row2", "row3"]
    assert [r.line.text for r in blocks[1].rows] == ["next1", "next2"]


def test_apply_block_first_prototype_ordering_assigns_hints_and_metadata():
    lines = [
        _line(1, "row2", start=12.5, y=120, vs=10.0, ve=18.0),
        _line(2, "row1", start=11.8, y=80, vs=10.0, ve=18.0),
        _line(3, "single", start=50.0, y=100, vs=50.0, ve=50.2),
    ]

    changed = apply_block_first_prototype_ordering(lines)

    assert changed is True
    # First block should be ordered by y regardless of original input order.
    hints = {ln.text: ln.block_order_hint for ln in lines}
    assert hints["row1"] < hints["row2"]
    # Non-block singleton should still receive a later hint.
    assert hints["single"] > hints["row2"]
    assert lines[0].reconstruction_meta["block_first"]["block_id"] == 0
    assert lines[1].reconstruction_meta["block_first"]["block_id"] == 0
