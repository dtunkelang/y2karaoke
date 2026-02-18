from y2karaoke.core.models import TargetLine
from y2karaoke.core.visual.refinement import (
    _assign_surrogate_timings_for_unresolved_overlap_blocks,
)


def test_assign_surrogate_timings_for_unresolved_overlap_blocks_sequences_lines():
    prev = TargetLine(
        line_index=1,
        start=38.0,
        end=39.0,
        text="prev",
        words=["prev"],
        y=80,
        word_rois=[(0, 0, 2, 2)],
        word_starts=[38.0],
        word_ends=[39.0],
        visibility_start=36.0,
        visibility_end=39.0,
    )
    a = TargetLine(
        line_index=2,
        start=39.0,
        end=40.0,
        text="Bruises on both",
        words=["Bruises", "on", "both"],
        y=100,
        word_rois=[(0, 0, 2, 2)],
        visibility_start=37.0,
        visibility_end=44.0,
    )
    b = TargetLine(
        line_index=3,
        start=40.0,
        end=41.0,
        text="Don't say thank you or please",
        words=["Don't", "say", "thank", "you", "or", "please"],
        y=130,
        word_rois=[(0, 0, 2, 2)],
        visibility_start=37.0,
        visibility_end=44.0,
    )
    c = TargetLine(
        line_index=4,
        start=41.0,
        end=42.0,
        text="I do",
        words=["I", "do"],
        y=160,
        word_rois=[(0, 0, 2, 2)],
        visibility_start=37.0,
        visibility_end=44.0,
    )
    nxt = TargetLine(
        line_index=5,
        start=45.0,
        end=46.0,
        text="next",
        words=["next"],
        y=90,
        word_rois=[(0, 0, 2, 2)],
        visibility_start=45.0,
        visibility_end=46.0,
    )

    g_jobs = [
        (prev, 0.0, 0.0),
        (a, 0.0, 0.0),
        (b, 0.0, 0.0),
        (c, 0.0, 0.0),
        (nxt, 0.0, 0.0),
    ]
    _assign_surrogate_timings_for_unresolved_overlap_blocks(g_jobs)

    assert (
        a.word_starts is not None
        and b.word_starts is not None
        and c.word_starts is not None
    )
    assert a.word_starts[0] >= 39.0
    assert b.word_starts[0] > a.word_starts[0] + 1.0
    assert c.word_starts[0] > b.word_starts[0] + 1.0
    assert c.word_starts[0] < 45.0
