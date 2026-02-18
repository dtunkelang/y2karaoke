from y2karaoke.core.models import TargetLine
from y2karaoke.core.visual.refinement import (
    _pull_late_first_lines_in_alternating_repeated_blocks,
    _rebalance_middle_lines_in_four_line_shared_visibility_runs,
    _retime_short_interstitial_lines_between_anchors,
)


def test_retime_short_interstitial_lines_between_anchors_delays_short_bridge_line():
    prev = TargetLine(
        line_index=1,
        start=40.7,
        end=43.1,
        text="Don't say thank you or please",
        words=["Don't", "say", "thank", "you", "or", "please"],
        y=170,
        word_rois=[(0, 0, 2, 2)] * 6,
        word_starts=[40.7],
        word_ends=[43.1],
    )
    curr = TargetLine(
        line_index=2,
        start=43.3,
        end=44.1,
        text="I do",
        words=["I", "do"],
        y=240,
        word_rois=[(0, 0, 2, 2)] * 2,
        word_starts=[43.3],
        word_ends=[44.1],
        visibility_start=43.0,
        visibility_end=54.8,
    )
    nxt = TargetLine(
        line_index=3,
        start=45.05,
        end=46.15,
        text="What I want",
        words=["What", "I", "want"],
        y=310,
        word_rois=[(0, 0, 2, 2)] * 3,
        word_starts=[45.05],
        word_ends=[46.15],
    )
    g_jobs = [(prev, 0.0, 0.0), (curr, 0.0, 0.0), (nxt, 0.0, 0.0)]

    _retime_short_interstitial_lines_between_anchors(g_jobs)

    assert curr.word_starts is not None and curr.word_ends is not None
    assert curr.word_starts[0] >= 43.7
    assert curr.word_ends[-1] <= 44.95


def test_pull_late_first_lines_in_alternating_repeated_blocks_pulls_first_line_earlier():
    prev = TargetLine(
        line_index=1,
        start=125.2,
        end=125.95,
        text="Might seduce your dad type",
        words=["Might", "seduce", "your", "dad", "type"],
        y=10,
        word_rois=[(0, 0, 2, 2)] * 5,
        word_starts=[125.2],
        word_ends=[125.95],
    )
    a = TargetLine(
        line_index=2,
        start=127.85,
        end=129.3,
        text="I'm the bad guy",
        words=["I'm", "the", "bad", "guy"],
        y=80,
        word_rois=[(0, 0, 2, 2)] * 4,
        word_starts=[127.85],
        word_ends=[129.3],
        visibility_start=126.0,
        visibility_end=144.0,
    )
    b = TargetLine(
        line_index=3,
        start=130.1,
        end=130.3,
        text="Duh",
        words=["Duh"],
        y=130,
        word_rois=[(0, 0, 2, 2)],
        word_starts=[130.1],
        word_ends=[130.3],
        visibility_start=126.0,
        visibility_end=144.0,
    )
    c = TargetLine(
        line_index=4,
        start=136.15,
        end=138.35,
        text="I'm the bad guy",
        words=["I'm", "the", "bad", "guy"],
        y=180,
        word_rois=[(0, 0, 2, 2)] * 4,
        word_starts=[136.15],
        word_ends=[138.35],
        visibility_start=126.0,
        visibility_end=144.0,
    )
    d = TargetLine(
        line_index=5,
        start=143.35,
        end=144.1,
        text="Duh",
        words=["Duh"],
        y=230,
        word_rois=[(0, 0, 2, 2)],
        word_starts=[143.35],
        word_ends=[144.1],
        visibility_start=126.0,
        visibility_end=144.0,
    )
    g_jobs = [
        (prev, 0.0, 0.0),
        (a, 0.0, 0.0),
        (b, 0.0, 0.0),
        (c, 0.0, 0.0),
        (d, 0.0, 0.0),
    ]

    _pull_late_first_lines_in_alternating_repeated_blocks(g_jobs)

    assert a.word_starts is not None
    assert a.word_starts[0] <= 126.1


def test_rebalance_middle_lines_in_four_line_shared_visibility_runs_spreads_middle():
    a = TargetLine(
        line_index=1,
        start=51.2,
        end=52.15,
        text="So you're a tough guy",
        words=["So", "you're", "a", "tough", "guy"],
        y=7,
        word_rois=[(0, 0, 2, 2)] * 5,
        word_starts=[51.2],
        word_ends=[52.15],
        visibility_start=51.0,
        visibility_end=57.0,
    )
    b = TargetLine(
        line_index=2,
        start=52.15,
        end=52.85,
        text="Like it really rough guy",
        words=["Like", "it", "really", "rough", "guy"],
        y=85,
        word_rois=[(0, 0, 2, 2)] * 5,
        word_starts=[52.15],
        word_ends=[52.85],
        visibility_start=51.0,
        visibility_end=57.0,
    )
    c = TargetLine(
        line_index=3,
        start=52.85,
        end=53.9,
        text="Just can't get enough guy",
        words=["Just", "can't", "get", "enough", "guy"],
        y=169,
        word_rois=[(0, 0, 2, 2)] * 5,
        word_starts=[52.85],
        word_ends=[53.9],
        visibility_start=51.0,
        visibility_end=57.0,
    )
    d = TargetLine(
        line_index=4,
        start=55.5,
        end=56.95,
        text="Chest always so puffed guy",
        words=["Chest", "always", "so", "puffed", "guy"],
        y=248,
        word_rois=[(0, 0, 2, 2)] * 5,
        word_starts=[55.5],
        word_ends=[56.95],
        visibility_start=51.0,
        visibility_end=57.0,
    )
    g_jobs = [(a, 0.0, 0.0), (b, 0.0, 0.0), (c, 0.0, 0.0), (d, 0.0, 0.0)]

    _rebalance_middle_lines_in_four_line_shared_visibility_runs(g_jobs)

    assert b.word_starts is not None and c.word_starts is not None
    assert b.word_starts[0] >= 52.5
    assert c.word_starts[0] >= 54.0
