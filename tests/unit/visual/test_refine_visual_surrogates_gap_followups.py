from y2karaoke.core.models import TargetLine
from y2karaoke.core.visual.refinement import (
    _assign_surrogate_cluster_timings,
    _rebalance_two_followups_after_short_lead,
    _retime_followups_in_short_lead_shared_visibility_runs,
    _retime_large_gaps_with_early_visibility,
)


def test_retime_large_gaps_with_early_visibility_pulls_line_forward():
    prev = TargetLine(
        line_index=1,
        start=77.9,
        end=81.0,
        text="Duh",
        words=["Duh"],
        y=90,
        word_rois=[(0, 0, 2, 2)],
        word_starts=[77.9],
        word_ends=[81.0],
        visibility_start=68.0,
        visibility_end=79.0,
    )
    curr = TargetLine(
        line_index=2,
        start=80.0,
        end=80.0,
        text="I like it when you take control",
        words=["I", "like", "it", "when", "you", "take", "control"],
        y=20,
        word_rois=[(0, 0, 2, 2)] * 7,
        word_starts=[84.8],
        word_ends=[88.5],
        visibility_start=80.0,
        visibility_end=93.0,
    )
    nxt = TargetLine(
        line_index=3,
        start=80.0,
        end=80.0,
        text="Even if you know that you don't",
        words=["Even", "if", "you", "know", "that", "you", "don't"],
        y=97,
        word_rois=[(0, 0, 2, 2)] * 7,
        word_starts=[88.5],
        word_ends=[91.05],
        visibility_start=80.0,
        visibility_end=94.0,
    )
    g_jobs = [(prev, 0.0, 0.0), (curr, 0.0, 0.0), (nxt, 0.0, 0.0)]

    _retime_large_gaps_with_early_visibility(g_jobs)

    assert curr.word_starts is not None
    assert curr.word_starts[0] <= 82.8


def test_retime_large_gaps_with_early_visibility_skips_without_nearby_anchor():
    prev = TargetLine(
        line_index=1,
        start=6.0,
        end=8.0,
        text="artifact",
        words=["artifact"],
        y=20,
        word_rois=[(0, 0, 2, 2)],
        word_starts=[6.0],
        word_ends=[8.0],
        visibility_start=6.0,
        visibility_end=8.0,
    )
    curr = TargetLine(
        line_index=2,
        start=17.0,
        end=17.0,
        text="White shirt now red",
        words=["White", "shirt", "now", "red"],
        y=80,
        word_rois=[(0, 0, 2, 2)] * 4,
        word_starts=[23.35],
        word_ends=[25.6],
        visibility_start=17.0,
        visibility_end=30.0,
    )
    g_jobs = [(prev, 0.0, 0.0), (curr, 0.0, 0.0)]

    _retime_large_gaps_with_early_visibility(g_jobs)

    assert curr.word_starts is not None
    assert curr.word_starts[0] == 23.35


def test_retime_large_gaps_with_early_visibility_skips_long_real_gap():
    prev = TargetLine(
        line_index=1,
        start=64.9,
        end=69.4,
        text="I'm the bad guy",
        words=["I'm", "the", "bad", "guy"],
        y=20,
        word_rois=[(0, 0, 2, 2)] * 4,
        word_starts=[64.9],
        word_ends=[69.4],
        visibility_start=65.0,
        visibility_end=79.0,
    )
    curr = TargetLine(
        line_index=2,
        start=65.0,
        end=81.0,
        text="I'm the bad guy",
        words=["I'm", "the", "bad", "guy"],
        y=168,
        word_rois=[(0, 0, 2, 2)] * 4,
        word_starts=[76.15],
        word_ends=[77.9],
        visibility_start=65.0,
        visibility_end=79.0,
    )
    g_jobs = [(prev, 0.0, 0.0), (curr, 0.0, 0.0)]

    _retime_large_gaps_with_early_visibility(g_jobs)

    assert curr.word_starts is not None
    assert curr.word_starts[0] == 76.15


def test_assign_surrogate_cluster_timings_adds_leading_slack_for_long_two_line_window():
    a = TargetLine(
        line_index=1,
        start=95.0,
        end=95.0,
        text="My mommy likes to",
        words=["My", "mommy", "likes", "to"],
        y=90,
        word_rois=[(0, 0, 2, 2)] * 4,
        visibility_start=95.0,
        visibility_end=102.0,
    )
    b = TargetLine(
        line_index=2,
        start=95.0,
        end=95.0,
        text="sing along with me",
        words=["sing", "along", "with", "me"],
        y=170,
        word_rois=[(0, 0, 2, 2)] * 4,
        visibility_start=95.0,
        visibility_end=102.0,
    )

    _assign_surrogate_cluster_timings(
        [a, b],
        prev_end_floor=93.5,
        next_start_cap=None,
        onset_hints=None,
    )

    assert a.word_starts is not None
    assert a.word_starts[0] >= 96.0


def test_retime_followups_in_short_lead_shared_visibility_runs_shifts_followups():
    lead = TargetLine(
        line_index=1,
        start=95.0,
        end=95.7,
        text="I'll be your animal",
        words=["I'll", "be", "your", "animal"],
        y=10,
        word_rois=[(0, 0, 2, 2)] * 4,
        word_starts=[95.0],
        word_ends=[95.7],
        visibility_start=95.0,
        visibility_end=102.0,
    )
    f1 = TargetLine(
        line_index=2,
        start=95.9,
        end=96.6,
        text="My mommy likes to",
        words=["My", "mommy", "likes", "to"],
        y=90,
        word_rois=[(0, 0, 2, 2)] * 4,
        word_starts=[95.9],
        word_ends=[96.6],
        visibility_start=95.0,
        visibility_end=102.0,
    )
    f2 = TargetLine(
        line_index=3,
        start=96.6,
        end=99.05,
        text="sing along with me",
        words=["sing", "along", "with", "me"],
        y=170,
        word_rois=[(0, 0, 2, 2)] * 4,
        word_starts=[96.6],
        word_ends=[99.05],
        visibility_start=95.0,
        visibility_end=102.0,
    )
    next_line = TargetLine(
        line_index=4,
        start=100.0,
        end=102.0,
        text="next",
        words=["next"],
        y=20,
        word_rois=[(0, 0, 2, 2)],
        word_starts=[100.0],
        word_ends=[102.0],
        visibility_start=100.0,
        visibility_end=100.0,
    )
    g_jobs = [(lead, 0.0, 0.0), (f1, 0.0, 0.0), (f2, 0.0, 0.0), (next_line, 0.0, 0.0)]

    _retime_followups_in_short_lead_shared_visibility_runs(g_jobs)

    assert f1.word_starts is not None
    assert f2.word_starts is not None
    assert f1.word_starts[0] > 95.9
    assert f2.word_starts[0] > 96.6


def test_rebalance_two_followups_after_short_lead_retimes_second_followup_later():
    lead = TargetLine(
        line_index=1,
        start=95.0,
        end=95.7,
        text="lead",
        words=["lead"],
        y=12,
        word_rois=[(0, 0, 2, 2)],
        word_starts=[95.0],
        word_ends=[95.7],
        visibility_start=95.0,
        visibility_end=102.0,
    )
    b = TargetLine(
        line_index=2,
        start=96.9,
        end=97.6,
        text="My mommy likes to",
        words=["My", "mommy", "likes", "to"],
        y=90,
        word_rois=[(0, 0, 2, 2)] * 4,
        word_starts=[96.9],
        word_ends=[97.6],
        visibility_start=95.0,
        visibility_end=102.0,
    )
    c = TargetLine(
        line_index=3,
        start=97.6,
        end=100.0,
        text="sing along with me",
        words=["sing", "along", "with", "me"],
        y=170,
        word_rois=[(0, 0, 2, 2)] * 4,
        word_starts=[97.6],
        word_ends=[100.0],
        visibility_start=95.0,
        visibility_end=102.0,
    )
    nxt = TargetLine(
        line_index=4,
        start=101.5,
        end=103.4,
        text="next",
        words=["next"],
        y=8,
        word_rois=[(0, 0, 2, 2)],
        word_starts=[101.5],
        word_ends=[103.4],
        visibility_start=103.0,
        visibility_end=111.0,
    )
    g_jobs = [(lead, 0.0, 0.0), (b, 0.0, 0.0), (c, 0.0, 0.0), (nxt, 0.0, 0.0)]

    _rebalance_two_followups_after_short_lead(g_jobs)

    assert c.word_starts is not None
    assert c.word_starts[0] >= 99.5
