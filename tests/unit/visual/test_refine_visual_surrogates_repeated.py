from y2karaoke.core.models import TargetLine
from y2karaoke.core.visual.refinement import (
    _compress_overlong_sparse_line_timings,
    _promote_unresolved_first_repeated_lines,
)


def test_promote_unresolved_first_repeated_lines_backfills_early_repeat():
    prev = TargetLine(
        line_index=1,
        start=210.0,
        end=211.2,
        text="anchor",
        words=["anchor"],
        y=20,
        word_rois=[(0, 0, 2, 2)],
        word_starts=[210.0],
        word_ends=[211.2],
        visibility_start=209.5,
        visibility_end=211.2,
    )
    early = TargetLine(
        line_index=2,
        start=211.5,
        end=213.0,
        text="I'm in love with your body",
        words=["I'm", "in", "love", "with", "your", "body"],
        y=90,
        word_rois=[(0, 0, 2, 2)] * 6,
        visibility_start=212.0,
        visibility_end=219.0,
    )
    later = TargetLine(
        line_index=3,
        start=223.6,
        end=225.4,
        text="I'm in love with your body",
        words=["I'm", "in", "love", "with", "your", "body"],
        y=90,
        word_rois=[(0, 0, 2, 2)] * 6,
        word_starts=[223.6],
        word_ends=[225.4],
        visibility_start=218.3,
        visibility_end=226.0,
    )
    g_jobs = [(prev, 0.0, 0.0), (early, 0.0, 0.0), (later, 0.0, 0.0)]

    _promote_unresolved_first_repeated_lines(g_jobs)

    assert early.word_starts is not None
    assert early.word_ends is not None
    assert early.word_starts[0] >= 212.0
    assert early.word_starts[0] <= 212.4
    assert early.word_ends[-1] < 223.6


def test_promote_unresolved_first_repeated_lines_skips_without_overlap():
    early = TargetLine(
        line_index=1,
        start=30.0,
        end=31.0,
        text="Every day discovering",
        words=["Every", "day", "discovering"],
        y=100,
        word_rois=[(0, 0, 2, 2)] * 3,
        visibility_start=30.0,
        visibility_end=31.0,
    )
    later = TargetLine(
        line_index=2,
        start=40.0,
        end=41.0,
        text="Every day discovering",
        words=["Every", "day", "discovering"],
        y=100,
        word_rois=[(0, 0, 2, 2)] * 3,
        word_starts=[40.0],
        word_ends=[41.0],
        visibility_start=40.0,
        visibility_end=41.0,
    )

    _promote_unresolved_first_repeated_lines([(early, 0.0, 0.0), (later, 0.0, 0.0)])

    assert early.word_starts is None


def test_compress_overlong_sparse_line_timings_in_overlap_block():
    a = TargetLine(
        line_index=1,
        start=216.0,
        end=223.4,
        text="Every day discovering",
        words=["Every", "day", "discovering"],
        y=90,
        word_rois=[(0, 0, 2, 2)] * 3,
        word_starts=[216.0, 216.4, 217.0],
        word_ends=[216.2, 216.6, 217.2],
        visibility_start=216.0,
        visibility_end=223.4,
    )
    b = TargetLine(
        line_index=2,
        start=216.0,
        end=233.3,
        text="I'm in love with your body",
        words=["I'm", "in", "love", "with", "your", "body"],
        y=150,
        word_rois=[(0, 0, 2, 2)] * 6,
        word_starts=[216.0, 216.4, 228.0, 229.0, 230.0, 231.0],
        word_ends=[216.2, 216.6, 228.2, 229.2, 230.2, 231.2],
        visibility_start=216.0,
        visibility_end=233.3,
    )
    nxt = TargetLine(
        line_index=3,
        start=223.7,
        end=225.0,
        text="next",
        words=["next"],
        y=220,
        word_rois=[(0, 0, 2, 2)],
        word_starts=[223.7],
        word_ends=[225.0],
        visibility_start=223.7,
        visibility_end=233.0,
    )

    _compress_overlong_sparse_line_timings(
        [(a, 0.0, 0.0), (b, 0.0, 0.0), (nxt, 0.0, 0.0)]
    )

    assert b.word_starts is not None
    assert b.word_ends is not None
    assert b.word_starts[0] == 216.0
    assert b.word_ends[-1] < 223.7
