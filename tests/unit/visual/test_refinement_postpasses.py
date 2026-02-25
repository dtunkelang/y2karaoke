from y2karaoke.core.models import TargetLine
from y2karaoke.core.visual.refinement_postpasses import (
    _assign_line_level_word_timings,
    _reorder_clean_screen_blocks_target_lines,
    _retime_clean_screen_blocks_by_vertical_order,
)


def _mk_line(
    idx: int, text: str, y: float, start: float, end: float, vs: float, ve: float
) -> TargetLine:
    ln = TargetLine(
        line_index=idx,
        start=start,
        end=end,
        text=text,
        words=text.split(),
        y=y,
        visibility_start=vs,
        visibility_end=ve,
    )
    _assign_line_level_word_timings(ln, start, end, 0.5)
    return ln


def test_retime_clean_screen_blocks_by_vertical_order_fixes_inverted_block() -> None:
    # Same screen/block, but line starts are inverted relative to vertical order.
    l1 = _mk_line(1, "Dreaming about the thing that", 160.0, 15.7, 16.7, 10.0, 18.0)
    l2 = _mk_line(2, "we could be", 230.0, 17.3, 18.0, 10.0, 18.0)
    l3 = _mk_line(3, "Lately I've been", 20.0, 10.1, 10.8, 10.0, 18.0)
    l4 = _mk_line(4, "losing sleep", 90.0, 13.1, 13.4, 10.0, 18.0)
    jobs = [(l1, 10.0, 18.0), (l2, 10.0, 18.0), (l3, 10.0, 18.0), (l4, 10.0, 18.0)]

    _retime_clean_screen_blocks_by_vertical_order(jobs)

    starts = [ln.word_starts[0] for ln in [l3, l4, l1, l2]]
    assert starts == sorted(starts)
    assert abs(float(l3.word_starts[0]) - 10.1) < 0.25
    assert abs(float(l1.word_starts[0]) - 15.7) < 0.5


def test_retime_clean_screen_blocks_by_vertical_order_ignores_noncontiguous_block() -> (
    None
):
    # Similar visibility windows but split by an unrelated intervening line; should not retime.
    l1 = _mk_line(1, "Dreaming", 160.0, 15.7, 16.7, 10.0, 18.0)
    mid = _mk_line(2, "Unrelated", 40.0, 20.0, 20.6, 19.8, 22.0)
    l2 = _mk_line(3, "Lately", 20.0, 10.1, 10.8, 10.0, 18.0)
    jobs = [(l1, 10.0, 18.0), (mid, 19.8, 22.0), (l2, 10.0, 18.0)]
    before = [
        float(l1.word_starts[0]),
        float(mid.word_starts[0]),
        float(l2.word_starts[0]),
    ]

    _retime_clean_screen_blocks_by_vertical_order(jobs)

    after = [
        float(l1.word_starts[0]),
        float(mid.word_starts[0]),
        float(l2.word_starts[0]),
    ]
    assert after == before


def test_reorder_clean_screen_blocks_target_lines_reorders_clean_inverted_block() -> (
    None
):
    l1 = _mk_line(1, "Dreaming about the thing that", 160.0, 10.1, 11.1, 10.0, 18.0)
    l2 = _mk_line(2, "we could be", 230.0, 16.8, 17.5, 10.0, 18.0)
    l3 = _mk_line(3, "Lately I've been", 20.0, 17.6, 18.2, 10.0, 18.0)
    l4 = _mk_line(4, "losing sleep", 90.0, 18.2, 18.5, 10.0, 18.0)
    lines = [l1, l2, l3, l4]

    _reorder_clean_screen_blocks_target_lines(lines)

    assert [ln.text for ln in lines] == [
        "Lately I've been",
        "losing sleep",
        "Dreaming about the thing that",
        "we could be",
    ]
