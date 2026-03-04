from y2karaoke.core.models import TargetLine
from y2karaoke.core.visual.refinement import (
    _retime_dense_runs_after_overlong_lead,
    _retime_repeated_blocks_with_long_tail_gap,
    _shrink_overlong_leads_in_dense_shared_visibility_runs,
)


def test_shrink_overlong_leads_in_dense_shared_visibility_runs_skips_repeated_text():
    lead = TargetLine(
        line_index=1,
        start=182.8,
        end=185.3,
        text="I'm a bad guy",
        words=["I'm", "a", "bad", "guy"],
        y=80,
        word_rois=[(0, 0, 2, 2)] * 4,
        word_starts=[182.8],
        word_ends=[185.3],
        visibility_start=184.0,
        visibility_end=200.0,
    )
    b = TargetLine(
        line_index=2,
        start=185.3,
        end=186.0,
        text="I'm a bad guy",
        words=["I'm", "a", "bad", "guy"],
        y=130,
        word_rois=[(0, 0, 2, 2)] * 4,
        word_starts=[185.3],
        word_ends=[186.0],
        visibility_start=184.0,
        visibility_end=200.0,
    )
    c = TargetLine(
        line_index=3,
        start=186.0,
        end=187.7,
        text="Bad guy bad guy",
        words=["Bad", "guy", "bad", "guy"],
        y=180,
        word_rois=[(0, 0, 2, 2)] * 4,
        word_starts=[186.0],
        word_ends=[187.7],
        visibility_start=184.0,
        visibility_end=200.0,
    )
    d = TargetLine(
        line_index=4,
        start=197.6,
        end=199.6,
        text="I'm a",
        words=["I'm", "a"],
        y=230,
        word_rois=[(0, 0, 2, 2)] * 2,
        word_starts=[197.6],
        word_ends=[199.6],
        visibility_start=184.0,
        visibility_end=200.0,
    )
    g_jobs = [(lead, 0.0, 0.0), (b, 0.0, 0.0), (c, 0.0, 0.0), (d, 0.0, 0.0)]
    _shrink_overlong_leads_in_dense_shared_visibility_runs(g_jobs)
    assert lead.word_starts is not None and b.word_starts is not None
    assert c.word_starts is not None and d.word_starts is not None
    assert lead.word_starts[0] == 182.8
    assert b.word_starts[0] == 185.3
    assert c.word_starts[0] == 186.0
    assert d.word_starts[0] == 197.6


def test_retime_dense_runs_after_overlong_lead_skips_repeated_text():
    prev = TargetLine(
        line_index=1,
        start=181.8,
        end=182.7,
        text="I'm wearing your cologne",
        words=["I'm", "wearing", "your", "cologne"],
        y=10,
        word_rois=[(0, 0, 2, 2)] * 4,
        word_starts=[181.8],
        word_ends=[182.7],
    )
    a = TargetLine(
        line_index=2,
        start=182.8,
        end=185.3,
        text="I'm a bad guy",
        words=["I'm", "a", "bad", "guy"],
        y=80,
        word_rois=[(0, 0, 2, 2)] * 4,
        word_starts=[182.8],
        word_ends=[185.3],
    )
    b = TargetLine(
        line_index=3,
        start=185.3,
        end=186.0,
        text="I'm a bad guy",
        words=["I'm", "a", "bad", "guy"],
        y=130,
        word_rois=[(0, 0, 2, 2)] * 4,
        word_starts=[185.3],
        word_ends=[186.0],
    )
    c = TargetLine(
        line_index=4,
        start=186.0,
        end=187.7,
        text="Bad guy bad guy",
        words=["Bad", "guy", "bad", "guy"],
        y=180,
        word_rois=[(0, 0, 2, 2)] * 4,
        word_starts=[186.0],
        word_ends=[187.7],
    )
    d = TargetLine(
        line_index=5,
        start=197.6,
        end=199.6,
        text="I'm a",
        words=["I'm", "a"],
        y=230,
        word_rois=[(0, 0, 2, 2)] * 2,
        word_starts=[197.6],
        word_ends=[199.6],
    )
    g_jobs = [
        (prev, 0.0, 0.0),
        (a, 0.0, 0.0),
        (b, 0.0, 0.0),
        (c, 0.0, 0.0),
        (d, 0.0, 0.0),
    ]
    _retime_dense_runs_after_overlong_lead(g_jobs)
    assert a.word_starts is not None and b.word_starts is not None
    assert c.word_starts is not None and d.word_starts is not None
    assert a.word_starts[0] == 182.8
    assert b.word_starts[0] == 185.3
    assert c.word_starts[0] == 186.0
    assert d.word_starts[0] == 197.6


def test_retime_repeated_blocks_with_long_tail_gap_delays_middle_lines():
    prev = TargetLine(
        line_index=0,
        start=181.95,
        end=183.1,
        text="I'm wearing your cologne",
        words=["I'm", "wearing", "your", "cologne"],
        y=10,
        word_rois=[(0, 0, 2, 2)] * 4,
        word_starts=[181.95],
        word_ends=[183.1],
    )
    a = TargetLine(
        line_index=1,
        start=183.8,
        end=185.3,
        text="I'm a bad guy",
        words=["I'm", "a", "bad", "guy"],
        y=80,
        word_rois=[(0, 0, 2, 2)] * 4,
        word_starts=[183.8],
        word_ends=[185.3],
    )
    b = TargetLine(
        line_index=2,
        start=186.5,
        end=187.2,
        text="I'm a bad guy",
        words=["I'm", "a", "bad", "guy"],
        y=130,
        word_rois=[(0, 0, 2, 2)] * 4,
        word_starts=[186.5],
        word_ends=[187.2],
    )
    c = TargetLine(
        line_index=3,
        start=187.2,
        end=188.9,
        text="Bad guy bad guy",
        words=["Bad", "guy", "bad", "guy"],
        y=180,
        word_rois=[(0, 0, 2, 2)] * 4,
        word_starts=[187.2],
        word_ends=[188.9],
    )
    d = TargetLine(
        line_index=4,
        start=198.4,
        end=200.1,
        text="I'm a",
        words=["I'm", "a"],
        y=230,
        word_rois=[(0, 0, 2, 2)] * 2,
        word_starts=[198.4],
        word_ends=[200.1],
    )
    g_jobs = [
        (prev, 0.0, 0.0),
        (a, 0.0, 0.0),
        (b, 0.0, 0.0),
        (c, 0.0, 0.0),
        (d, 0.0, 0.0),
    ]
    _retime_repeated_blocks_with_long_tail_gap(g_jobs)
    assert (
        a.word_starts is not None
        and b.word_starts is not None
        and c.word_starts is not None
    )
    assert 185.5 <= a.word_starts[0] <= 186.2
    assert 194.0 <= b.word_starts[0] <= 194.6
    assert 196.0 <= c.word_starts[0] <= 196.7


def test_retime_repeated_blocks_with_long_tail_gap_skips_mixed_lead_lines():
    a = TargetLine(
        line_index=1,
        start=125.2,
        end=126.0,
        text="Might seduce your dad type",
        words=["Might", "seduce", "your", "dad", "type"],
        y=40,
        word_rois=[(0, 0, 2, 2)] * 5,
        word_starts=[125.2],
        word_ends=[126.0],
    )
    b = TargetLine(
        line_index=2,
        start=132.15,
        end=133.6,
        text="I'm the bad guy",
        words=["I'm", "the", "bad", "guy"],
        y=90,
        word_rois=[(0, 0, 2, 2)] * 4,
        word_starts=[132.15],
        word_ends=[133.6],
    )
    c = TargetLine(
        line_index=3,
        start=134.15,
        end=134.95,
        text="Duh",
        words=["Duh"],
        y=140,
        word_rois=[(0, 0, 2, 2)],
        word_starts=[134.15],
        word_ends=[134.95],
    )
    d = TargetLine(
        line_index=4,
        start=143.35,
        end=144.1,
        text="Duh",
        words=["Duh"],
        y=190,
        word_rois=[(0, 0, 2, 2)],
        word_starts=[143.35],
        word_ends=[144.1],
    )
    g_jobs = [(a, 0.0, 0.0), (b, 0.0, 0.0), (c, 0.0, 0.0), (d, 0.0, 0.0)]
    _retime_repeated_blocks_with_long_tail_gap(g_jobs)
    assert a.word_starts is not None and b.word_starts is not None
    assert c.word_starts is not None and d.word_starts is not None
    assert a.word_starts[0] == 125.2
    assert b.word_starts[0] == 132.15
    assert c.word_starts[0] == 134.15
    assert d.word_starts[0] == 143.35
