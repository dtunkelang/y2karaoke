from y2karaoke.core.models import TargetLine
from y2karaoke.core.visual.refinement import (
    _retime_compressed_shared_visibility_blocks,
    _retime_large_gaps_with_early_visibility,
    _retime_followups_in_short_lead_shared_visibility_runs,
    _rebalance_two_followups_after_short_lead,
    _rebalance_early_lead_shared_visibility_runs,
    _shrink_overlong_leads_in_dense_shared_visibility_runs,
    _retime_dense_runs_after_overlong_lead,
    _pull_dense_short_runs_toward_previous_anchor,
    _clamp_line_ends_to_visibility_windows,
    _pull_lines_earlier_after_visibility_transitions,
    _assign_surrogate_cluster_timings,
    _assign_surrogate_timings_for_unresolved_overlap_blocks,
    _retime_late_first_lines_in_shared_visibility_blocks,
)


def test_clamp_line_ends_to_visibility_windows_shortens_overrun():
    ln = TargetLine(
        line_index=1,
        start=107.9,
        end=112.0,
        text="she'll pity the men I know",
        words=["she'll", "pity", "the", "men", "I", "know"],
        y=170,
        word_rois=[(0, 0, 2, 2)] * 6,
        word_starts=[107.9],
        word_ends=[112.0],
        visibility_start=103.0,
        visibility_end=111.0,
    )
    g_jobs = [(ln, 0.0, 0.0)]
    _clamp_line_ends_to_visibility_windows(g_jobs)
    assert ln.word_ends is not None
    assert ln.word_ends[-1] <= 111.1


def test_pull_lines_earlier_after_visibility_transitions_reduces_gap():
    prev = TargetLine(
        line_index=1,
        start=107.9,
        end=111.1,
        text="prev",
        words=["prev"],
        y=170,
        word_rois=[(0, 0, 2, 2)],
        word_starts=[107.9],
        word_ends=[111.1],
        visibility_start=103.0,
        visibility_end=111.0,
    )
    curr = TargetLine(
        line_index=2,
        start=112.65,
        end=116.05,
        text="So you're a tough guy",
        words=["So", "you're", "a", "tough", "guy"],
        y=7,
        word_rois=[(0, 0, 2, 2)] * 5,
        word_starts=[112.65],
        word_ends=[116.05],
        visibility_start=112.0,
        visibility_end=118.0,
    )
    g_jobs = [(prev, 0.0, 0.0), (curr, 0.0, 0.0)]
    _pull_lines_earlier_after_visibility_transitions(g_jobs)
    assert curr.word_starts is not None
    assert curr.word_starts[0] <= 111.7


def test_rebalance_early_lead_shared_visibility_runs_spreads_three_lines():
    a = TargetLine(
        line_index=1,
        start=101.5,
        end=103.4,
        text="But she won't sing this song",
        words=["But", "she", "won't", "sing", "this", "song"],
        y=6,
        word_rois=[(0, 0, 2, 2)] * 6,
        word_starts=[101.5],
        word_ends=[103.4],
        visibility_start=103.0,
        visibility_end=111.0,
    )
    b = TargetLine(
        line_index=2,
        start=103.6,
        end=107.7,
        text="If she reads all the lyrics",
        words=["If", "she", "reads", "all", "the", "lyrics"],
        y=86,
        word_rois=[(0, 0, 2, 2)] * 6,
        word_starts=[103.6],
        word_ends=[107.7],
        visibility_start=103.0,
        visibility_end=111.0,
    )
    c = TargetLine(
        line_index=3,
        start=107.9,
        end=111.1,
        text="she'll pity the men I know",
        words=["she'll", "pity", "the", "men", "I", "know"],
        y=167,
        word_rois=[(0, 0, 2, 2)] * 6,
        word_starts=[107.9],
        word_ends=[111.1],
        visibility_start=103.0,
        visibility_end=111.0,
    )
    nxt = TargetLine(
        line_index=4,
        start=111.45,
        end=114.85,
        text="So you're a tough guy",
        words=["So", "you're", "a", "tough", "guy"],
        y=7,
        word_rois=[(0, 0, 2, 2)] * 5,
        word_starts=[111.45],
        word_ends=[114.85],
        visibility_start=112.0,
        visibility_end=118.0,
    )
    g_jobs = [(a, 0.0, 0.0), (b, 0.0, 0.0), (c, 0.0, 0.0), (nxt, 0.0, 0.0)]

    _rebalance_early_lead_shared_visibility_runs(g_jobs)

    assert (
        a.word_starts is not None
        and b.word_starts is not None
        and c.word_starts is not None
    )
    assert a.word_starts[0] >= 102.0
    assert b.word_starts[0] >= 104.8
    assert c.word_starts[0] >= 107.8


def test_shrink_overlong_leads_in_dense_shared_visibility_runs_rebalances_tail():
    lead = TargetLine(
        line_index=1,
        start=111.45,
        end=114.85,
        text="So you're a tough guy",
        words=["So", "you're", "a", "tough", "guy"],
        y=7,
        word_rois=[(0, 0, 2, 2)] * 5,
        word_starts=[111.45],
        word_ends=[114.85],
        visibility_start=112.0,
        visibility_end=118.0,
    )
    b = TargetLine(
        line_index=2,
        start=114.85,
        end=115.85,
        text="Like it really rough guy",
        words=["Like", "it", "really", "rough", "guy"],
        y=88,
        word_rois=[(0, 0, 2, 2)] * 5,
        word_starts=[114.85],
        word_ends=[115.85],
        visibility_start=112.0,
        visibility_end=118.0,
    )
    c = TargetLine(
        line_index=3,
        start=115.85,
        end=116.85,
        text="Just can't get enough guy",
        words=["Just", "can't", "get", "enough", "guy"],
        y=167,
        word_rois=[(0, 0, 2, 2)] * 5,
        word_starts=[115.85],
        word_ends=[116.85],
        visibility_start=112.0,
        visibility_end=118.0,
    )
    d = TargetLine(
        line_index=4,
        start=116.85,
        end=118.1,
        text="Chest always so puffed guy",
        words=["Chest", "always", "so", "puffed", "guy"],
        y=247,
        word_rois=[(0, 0, 2, 2)] * 5,
        word_starts=[116.85],
        word_ends=[118.1],
        visibility_start=112.0,
        visibility_end=118.0,
    )
    nxt = TargetLine(
        line_index=5,
        start=119.05,
        end=120.1,
        text="I'm that bad type",
        words=["I'm", "that", "bad", "type"],
        y=10,
        word_rois=[(0, 0, 2, 2)] * 4,
        word_starts=[119.05],
        word_ends=[120.1],
        visibility_start=118.0,
        visibility_end=123.0,
    )
    g_jobs = [
        (lead, 0.0, 0.0),
        (b, 0.0, 0.0),
        (c, 0.0, 0.0),
        (d, 0.0, 0.0),
        (nxt, 0.0, 0.0),
    ]

    _shrink_overlong_leads_in_dense_shared_visibility_runs(g_jobs)

    assert lead.word_ends is not None
    assert (
        b.word_starts is not None
        and c.word_starts is not None
        and d.word_starts is not None
    )
    assert lead.word_ends[-1] <= 112.8
    assert 112.6 <= b.word_starts[0] <= 113.1
    assert 113.5 <= c.word_starts[0] <= 114.2
    assert 114.5 <= d.word_starts[0] <= 115.2


def test_shrink_overlong_leads_in_dense_shared_visibility_runs_applies_early_block_shift():
    prev = TargetLine(
        line_index=0,
        start=108.2,
        end=111.1,
        text="she'll pity the men I know",
        words=["she'll", "pity", "the", "men", "I", "know"],
        y=167,
        word_rois=[(0, 0, 2, 2)] * 6,
        word_starts=[108.2],
        word_ends=[111.1],
        visibility_start=103.0,
        visibility_end=111.0,
    )
    lead = TargetLine(
        line_index=1,
        start=112.1,
        end=113.2,
        text="So you're a tough guy",
        words=["So", "you're", "a", "tough", "guy"],
        y=7,
        word_rois=[(0, 0, 2, 2)] * 5,
        word_starts=[112.1],
        word_ends=[113.2],
        visibility_start=112.0,
        visibility_end=118.0,
    )
    b = TargetLine(
        line_index=2,
        start=113.4,
        end=114.5,
        text="Like it really rough guy",
        words=["Like", "it", "really", "rough", "guy"],
        y=88,
        word_rois=[(0, 0, 2, 2)] * 5,
        word_starts=[113.4],
        word_ends=[114.5],
        visibility_start=112.0,
        visibility_end=118.0,
    )
    c = TargetLine(
        line_index=3,
        start=114.6,
        end=115.6,
        text="Just can't get enough guy",
        words=["Just", "can't", "get", "enough", "guy"],
        y=167,
        word_rois=[(0, 0, 2, 2)] * 5,
        word_starts=[114.6],
        word_ends=[115.6],
        visibility_start=112.0,
        visibility_end=118.0,
    )
    d = TargetLine(
        line_index=4,
        start=115.8,
        end=116.8,
        text="Chest always so puffed guy",
        words=["Chest", "always", "so", "puffed", "guy"],
        y=247,
        word_rois=[(0, 0, 2, 2)] * 5,
        word_starts=[115.8],
        word_ends=[116.8],
        visibility_start=112.0,
        visibility_end=118.0,
    )
    g_jobs = [
        (prev, 0.0, 0.0),
        (lead, 0.0, 0.0),
        (b, 0.0, 0.0),
        (c, 0.0, 0.0),
        (d, 0.0, 0.0),
    ]

    _shrink_overlong_leads_in_dense_shared_visibility_runs(g_jobs)

    assert lead.word_starts is not None and b.word_starts is not None
    assert lead.word_starts[0] <= 111.4
    assert b.word_starts[0] <= 112.7


def test_retime_dense_runs_after_overlong_lead_pulls_four_line_tail_forward():
    prev = TargetLine(
        line_index=1,
        start=108.2,
        end=111.1,
        text="she'll pity the men I know",
        words=["she'll", "pity", "the", "men", "I", "know"],
        y=167,
        word_rois=[(0, 0, 2, 2)] * 6,
        word_starts=[108.2],
        word_ends=[111.1],
    )
    a = TargetLine(
        line_index=2,
        start=112.05,
        end=113.1,
        text="So you're a tough guy",
        words=["So", "you're", "a", "tough", "guy"],
        y=7,
        word_rois=[(0, 0, 2, 2)] * 5,
        word_starts=[112.05],
        word_ends=[114.05],
    )
    b = TargetLine(
        line_index=3,
        start=114.05,
        end=115.1,
        text="Like it really rough guy",
        words=["Like", "it", "really", "rough", "guy"],
        y=88,
        word_rois=[(0, 0, 2, 2)] * 5,
        word_starts=[114.05],
        word_ends=[115.1],
    )
    c = TargetLine(
        line_index=4,
        start=115.3,
        end=116.35,
        text="Just can't get enough guy",
        words=["Just", "can't", "get", "enough", "guy"],
        y=167,
        word_rois=[(0, 0, 2, 2)] * 5,
        word_starts=[115.3],
        word_ends=[116.35],
    )
    d = TargetLine(
        line_index=5,
        start=117.25,
        end=118.1,
        text="Chest always so puffed guy",
        words=["Chest", "always", "so", "puffed", "guy"],
        y=247,
        word_rois=[(0, 0, 2, 2)] * 5,
        word_starts=[117.25],
        word_ends=[118.1],
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
    assert a.word_starts[0] <= 111.4
    assert b.word_starts[0] <= 112.4
    assert c.word_starts[0] <= 113.7
    assert d.word_starts[0] <= 115.6


def test_pull_dense_short_runs_toward_previous_anchor_reduces_local_offset():
    prev = TargetLine(
        line_index=1,
        start=108.2,
        end=111.1,
        text="she'll pity the men I know",
        words=["she'll", "pity", "the", "men", "I", "know"],
        y=167,
        word_rois=[(0, 0, 2, 2)] * 6,
        word_starts=[108.2],
        word_ends=[111.1],
    )
    a = TargetLine(
        line_index=2,
        start=112.05,
        end=113.1,
        text="So you're a tough guy",
        words=["So", "you're", "a", "tough", "guy"],
        y=7,
        word_rois=[(0, 0, 2, 2)] * 5,
        word_starts=[112.05],
        word_ends=[113.1],
    )
    b = TargetLine(
        line_index=3,
        start=113.3,
        end=115.1,
        text="Like it really rough guy",
        words=["Like", "it", "really", "rough", "guy"],
        y=88,
        word_rois=[(0, 0, 2, 2)] * 5,
        word_starts=[113.3],
        word_ends=[115.1],
    )
    c = TargetLine(
        line_index=4,
        start=115.3,
        end=117.05,
        text="Just can't get enough guy",
        words=["Just", "can't", "get", "enough", "guy"],
        y=167,
        word_rois=[(0, 0, 2, 2)] * 5,
        word_starts=[115.3],
        word_ends=[117.05],
    )
    d = TargetLine(
        line_index=5,
        start=117.25,
        end=118.1,
        text="Chest always so puffed guy",
        words=["Chest", "always", "so", "puffed", "guy"],
        y=247,
        word_rois=[(0, 0, 2, 2)] * 5,
        word_starts=[117.25],
        word_ends=[118.1],
    )
    g_jobs = [
        (prev, 0.0, 0.0),
        (a, 0.0, 0.0),
        (b, 0.0, 0.0),
        (c, 0.0, 0.0),
        (d, 0.0, 0.0),
    ]

    _pull_dense_short_runs_toward_previous_anchor(g_jobs)

    assert a.word_starts is not None and b.word_starts is not None
    assert c.word_starts is not None and d.word_starts is not None
    assert a.word_starts[0] <= 111.2
    assert b.word_starts[0] <= 112.5
    assert c.word_starts[0] <= 114.5
    assert d.word_starts[0] <= 116.5
