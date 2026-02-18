from y2karaoke.core.models import TargetLine
from y2karaoke.core.visual.refinement import (
    _rebalance_early_lead_shared_visibility_runs,
    _shrink_overlong_leads_in_dense_shared_visibility_runs,
    _retime_dense_runs_after_overlong_lead,
    _retime_repeated_blocks_with_long_tail_gap,
    _pull_dense_short_runs_toward_previous_anchor,
    _clamp_line_ends_to_visibility_windows,
    _pull_lines_earlier_after_visibility_transitions,
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


def test_retime_dense_runs_after_overlong_lead_respects_visibility_lag_floor():
    prev = TargetLine(
        line_index=1,
        start=45.05,
        end=46.15,
        text="What I want",
        words=["What", "I", "want"],
        y=7,
        word_rois=[(0, 0, 2, 2)] * 3,
        word_starts=[45.05],
        word_ends=[46.15],
    )
    a = TargetLine(
        line_index=2,
        start=46.35,
        end=48.55,
        text="when I'm wanting to",
        words=["when", "I'm", "wanting", "to"],
        y=87,
        word_rois=[(0, 0, 2, 2)] * 4,
        word_starts=[46.35],
        word_ends=[48.55],
    )
    b = TargetLine(
        line_index=3,
        start=48.60,
        end=49.50,
        text="My soul so cynical",
        words=["My", "soul", "so", "cynical"],
        y=167,
        word_rois=[(0, 0, 2, 2)] * 4,
        word_starts=[48.60],
        word_ends=[49.50],
        visibility_start=48.00,
        visibility_end=60.00,
    )
    c = TargetLine(
        line_index=4,
        start=49.60,
        end=50.60,
        text="So you're a tough guy",
        words=["So", "you're", "a", "tough", "guy"],
        y=247,
        word_rois=[(0, 0, 2, 2)] * 5,
        word_starts=[49.60],
        word_ends=[50.60],
        visibility_start=49.80,
        visibility_end=61.60,
    )
    d = TargetLine(
        line_index=5,
        start=50.80,
        end=52.25,
        text="Like it really rough guy",
        words=["Like", "it", "really", "rough", "guy"],
        y=327,
        word_rois=[(0, 0, 2, 2)] * 5,
        word_starts=[50.80],
        word_ends=[52.25],
        visibility_start=51.40,
        visibility_end=63.40,
    )
    g_jobs = [
        (prev, 0.0, 0.0),
        (a, 0.0, 0.0),
        (b, 0.0, 0.0),
        (c, 0.0, 0.0),
        (d, 0.0, 0.0),
    ]

    _retime_dense_runs_after_overlong_lead(g_jobs)

    assert (
        b.word_starts is not None
        and c.word_starts is not None
        and d.word_starts is not None
    )
    assert b.word_starts[0] >= 48.60
    assert c.word_starts[0] >= 50.40
    assert d.word_starts[0] >= 52.00


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


def test_pull_dense_short_runs_toward_previous_anchor_skips_repeated_text_block():
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
        end=184.25,
        text="I'm a bad guy",
        words=["I'm", "a", "bad", "guy"],
        y=80,
        word_rois=[(0, 0, 2, 2)] * 4,
        word_starts=[182.8],
        word_ends=[184.25],
    )
    b = TargetLine(
        line_index=3,
        start=184.25,
        end=184.95,
        text="I'm a bad guy",
        words=["I'm", "a", "bad", "guy"],
        y=130,
        word_rois=[(0, 0, 2, 2)] * 4,
        word_starts=[184.25],
        word_ends=[184.95],
    )
    c = TargetLine(
        line_index=4,
        start=184.95,
        end=186.65,
        text="Bad guy bad guy",
        words=["Bad", "guy", "bad", "guy"],
        y=180,
        word_rois=[(0, 0, 2, 2)] * 4,
        word_starts=[184.95],
        word_ends=[186.65],
    )
    d = TargetLine(
        line_index=5,
        start=197.55,
        end=199.6,
        text="I'm a",
        words=["I'm", "a"],
        y=230,
        word_rois=[(0, 0, 2, 2)] * 2,
        word_starts=[197.55],
        word_ends=[199.6],
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
    assert a.word_starts[0] == 182.8
    assert b.word_starts[0] == 184.25
    assert c.word_starts[0] == 184.95
    assert d.word_starts[0] == 197.55


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
