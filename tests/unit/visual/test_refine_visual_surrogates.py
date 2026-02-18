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


def test_assign_surrogate_timings_handles_short_visibility_unresolved_lines():
    prev = TargetLine(
        line_index=1,
        start=111.4,
        end=114.8,
        text="So you're a tough guy",
        words=["So", "you're", "a", "tough", "guy"],
        y=10,
        word_rois=[(0, 0, 2, 2)] * 5,
        word_starts=[111.4],
        word_ends=[114.8],
        visibility_start=112.0,
        visibility_end=118.0,
    )
    a = TargetLine(
        line_index=2,
        start=112.0,
        end=115.0,
        text="Like it really rough guy",
        words=["Like", "it", "really", "rough", "guy"],
        y=90,
        word_rois=[(0, 0, 2, 2)] * 5,
        visibility_start=112.0,
        visibility_end=115.0,
    )
    b = TargetLine(
        line_index=3,
        start=112.0,
        end=115.0,
        text="Just can't get enough guy",
        words=["Just", "can't", "get", "enough", "guy"],
        y=170,
        word_rois=[(0, 0, 2, 2)] * 5,
        visibility_start=112.0,
        visibility_end=115.0,
    )
    nxt = TargetLine(
        line_index=4,
        start=115.6,
        end=118.1,
        text="Chest always so puffed guy",
        words=["Chest", "always", "so", "puffed", "guy"],
        y=250,
        word_rois=[(0, 0, 2, 2)] * 5,
        word_starts=[115.6],
        word_ends=[118.1],
        visibility_start=112.0,
        visibility_end=118.0,
    )

    g_jobs = [(prev, 0.0, 0.0), (a, 0.0, 0.0), (b, 0.0, 0.0), (nxt, 0.0, 0.0)]
    _assign_surrogate_timings_for_unresolved_overlap_blocks(g_jobs)

    assert a.word_starts is not None and b.word_starts is not None
    assert 114.75 <= a.word_starts[0] <= 115.2
    assert a.word_starts[0] < b.word_starts[0] <= 115.8


def test_retime_late_first_lines_in_shared_visibility_blocks_nudges_first_line():
    prev = TargetLine(
        line_index=1,
        start=35.0,
        end=37.0,
        text="prev",
        words=["prev"],
        y=80,
        word_rois=[(0, 0, 2, 2)],
        word_starts=[35.0],
        word_ends=[37.0],
        visibility_start=31.0,
        visibility_end=36.0,
    )
    first = TargetLine(
        line_index=2,
        start=37.0,
        end=37.0,
        text="Bruises on both",
        words=["Bruises", "on", "both"],
        y=100,
        word_rois=[(0, 0, 2, 2)],
        word_starts=[38.9],
        word_ends=[40.6],
        visibility_start=37.0,
        visibility_end=44.0,
    )
    second = TargetLine(
        line_index=3,
        start=37.0,
        end=37.0,
        text="my knees for you",
        words=["my", "knees", "for", "you"],
        y=130,
        word_rois=[(0, 0, 2, 2)],
        word_starts=[40.8],
        word_ends=[42.7],
        visibility_start=37.0,
        visibility_end=44.0,
    )
    third = TargetLine(
        line_index=4,
        start=37.0,
        end=37.0,
        text="Don't say thank you or please",
        words=["Don't", "say", "thank", "you", "or", "please"],
        y=160,
        word_rois=[(0, 0, 2, 2)],
        word_starts=[42.7],
        word_ends=[44.1],
        visibility_start=37.0,
        visibility_end=44.0,
    )

    g_jobs = [
        (prev, 0.0, 0.0),
        (first, 0.0, 0.0),
        (second, 0.0, 0.0),
        (third, 0.0, 0.0),
    ]
    _retime_late_first_lines_in_shared_visibility_blocks(g_jobs)

    assert first.word_starts is not None
    assert first.word_starts[0] <= 37.4
    assert second.word_starts is not None
    assert second.word_starts[0] >= 40.5


def test_retime_late_first_lines_allows_longer_shared_visibility_windows():
    prev = TargetLine(
        line_index=1,
        start=76.0,
        end=77.9,
        text="prev",
        words=["prev"],
        y=70,
        word_rois=[(0, 0, 2, 2)],
        word_starts=[76.0],
        word_ends=[77.9],
        visibility_start=65.0,
        visibility_end=79.0,
    )
    first = TargetLine(
        line_index=2,
        start=80.0,
        end=80.0,
        text="I like it when you take control",
        words=["I", "like", "it", "when", "you", "take", "control"],
        y=20,
        word_rois=[(0, 0, 2, 2)],
        word_starts=[84.8],
        word_ends=[88.5],
        visibility_start=80.0,
        visibility_end=93.0,
    )
    second = TargetLine(
        line_index=3,
        start=80.0,
        end=80.0,
        text="Even if you know that you don't",
        words=["Even", "if", "you", "know", "that", "you", "don't"],
        y=90,
        word_rois=[(0, 0, 2, 2)],
        word_starts=[88.5],
        word_ends=[91.0],
        visibility_start=80.0,
        visibility_end=94.0,
    )
    third = TargetLine(
        line_index=4,
        start=80.0,
        end=95.0,
        text="own me I'll let you play the role",
        words=["own", "me", "I'll", "let", "you", "play", "the", "role"],
        y=170,
        word_rois=[(0, 0, 2, 2)],
        word_starts=[91.0],
        word_ends=[93.5],
        visibility_start=80.0,
        visibility_end=93.0,
    )
    g_jobs = [
        (prev, 0.0, 0.0),
        (first, 0.0, 0.0),
        (second, 0.0, 0.0),
        (third, 0.0, 0.0),
    ]

    _retime_late_first_lines_in_shared_visibility_blocks(g_jobs)

    assert first.word_starts is not None
    assert first.word_starts[0] <= 83.0


def test_retime_late_first_lines_skips_blocks_without_previous_anchor():
    first = TargetLine(
        line_index=1,
        start=17.0,
        end=17.0,
        text="White shirt now red",
        words=["White", "shirt", "now", "red"],
        y=20,
        word_rois=[(0, 0, 2, 2)],
        word_starts=[23.35],
        word_ends=[25.6],
        visibility_start=17.0,
        visibility_end=30.0,
    )
    second = TargetLine(
        line_index=2,
        start=17.0,
        end=17.0,
        text="My bloody nose",
        words=["My", "bloody", "nose"],
        y=90,
        word_rois=[(0, 0, 2, 2)],
        word_starts=[25.6],
        word_ends=[27.05],
        visibility_start=17.0,
        visibility_end=30.0,
    )
    third = TargetLine(
        line_index=3,
        start=17.0,
        end=17.0,
        text="Sleepin' you're on",
        words=["Sleepin'", "you're", "on"],
        y=150,
        word_rois=[(0, 0, 2, 2)],
        word_starts=[27.05],
        word_ends=[28.95],
        visibility_start=17.0,
        visibility_end=30.0,
    )
    g_jobs = [
        (first, 0.0, 0.0),
        (second, 0.0, 0.0),
        (third, 0.0, 0.0),
    ]

    _retime_late_first_lines_in_shared_visibility_blocks(g_jobs)

    assert first.word_starts is not None
    assert first.word_starts[0] == 23.35


def test_retime_late_first_lines_skips_with_distant_previous_anchor():
    prev = TargetLine(
        line_index=0,
        start=6.0,
        end=10.0,
        text="artifact",
        words=["artifact"],
        y=10,
        word_rois=[(0, 0, 2, 2)],
        word_starts=[6.0],
        word_ends=[8.0],
        visibility_start=6.0,
        visibility_end=8.0,
    )
    first = TargetLine(
        line_index=1,
        start=17.0,
        end=17.0,
        text="White shirt now red",
        words=["White", "shirt", "now", "red"],
        y=20,
        word_rois=[(0, 0, 2, 2)],
        word_starts=[23.35],
        word_ends=[25.6],
        visibility_start=17.0,
        visibility_end=30.0,
    )
    second = TargetLine(
        line_index=2,
        start=17.0,
        end=17.0,
        text="My bloody nose",
        words=["My", "bloody", "nose"],
        y=90,
        word_rois=[(0, 0, 2, 2)],
        word_starts=[25.6],
        word_ends=[27.05],
        visibility_start=17.0,
        visibility_end=30.0,
    )
    third = TargetLine(
        line_index=3,
        start=17.0,
        end=17.0,
        text="Sleepin' you're on",
        words=["Sleepin'", "you're", "on"],
        y=150,
        word_rois=[(0, 0, 2, 2)],
        word_starts=[27.05],
        word_ends=[28.95],
        visibility_start=17.0,
        visibility_end=30.0,
    )

    g_jobs = [
        (prev, 0.0, 0.0),
        (first, 0.0, 0.0),
        (second, 0.0, 0.0),
        (third, 0.0, 0.0),
    ]
    _retime_late_first_lines_in_shared_visibility_blocks(g_jobs)

    assert first.word_starts is not None
    assert first.word_starts[0] == 23.35


def test_retime_compressed_shared_visibility_blocks_spreads_late_cluster():
    prev = TargetLine(
        line_index=0,
        start=30.0,
        end=31.0,
        text="prev",
        words=["prev"],
        y=60,
        word_rois=[(0, 0, 2, 2)],
        word_starts=[30.0],
        word_ends=[31.0],
        visibility_start=28.0,
        visibility_end=31.0,
    )
    line1 = TargetLine(
        line_index=1,
        start=31.0,
        end=31.0,
        text="Creepin' around",
        words=["Creepin'", "around"],
        y=80,
        word_rois=[(0, 0, 2, 2)],
        word_starts=[31.0],
        word_ends=[32.6],
        visibility_start=31.0,
        visibility_end=36.0,
    )
    line2 = TargetLine(
        line_index=2,
        start=31.0,
        end=31.0,
        text="like no one knows",
        words=["like", "no", "one", "knows"],
        y=100,
        word_rois=[(0, 0, 2, 2)],
        word_starts=[34.1],
        word_ends=[36.1],
        visibility_start=31.0,
        visibility_end=36.0,
    )
    line3 = TargetLine(
        line_index=3,
        start=31.0,
        end=38.0,
        text="Think you're so criminal",
        words=["Think", "you're", "so", "criminal"],
        y=120,
        word_rois=[(0, 0, 2, 2)],
        word_starts=[35.0],
        word_ends=[37.0],
        visibility_start=31.0,
        visibility_end=36.0,
    )
    next_line = TargetLine(
        line_index=4,
        start=37.05,
        end=38.8,
        text="Bruises on both",
        words=["Bruises", "on", "both"],
        y=150,
        word_rois=[(0, 0, 2, 2)],
        word_starts=[37.05],
        word_ends=[38.8],
        visibility_start=37.0,
        visibility_end=44.0,
    )

    g_jobs = [
        (prev, 0.0, 0.0),
        (line1, 0.0, 0.0),
        (line2, 0.0, 0.0),
        (line3, 0.0, 0.0),
        (next_line, 0.0, 0.0),
    ]
    _retime_compressed_shared_visibility_blocks(g_jobs)

    assert line2.word_starts is not None
    assert line3.word_starts is not None
    assert line2.word_starts[0] < 33.6
    assert line3.word_starts[0] < 35.0


def test_retime_compressed_shared_visibility_blocks_handles_late_second_line_gap():
    prev = TargetLine(
        line_index=1,
        start=34.5,
        end=36.9,
        text="prev",
        words=["prev"],
        y=70,
        word_rois=[(0, 0, 2, 2)],
        word_starts=[34.5],
        word_ends=[36.9],
        visibility_start=31.0,
        visibility_end=36.0,
    )
    line1 = TargetLine(
        line_index=2,
        start=37.05,
        end=38.8,
        text="Bruises on both",
        words=["Bruises", "on", "both"],
        y=100,
        word_rois=[(0, 0, 2, 2)],
        word_starts=[37.05],
        word_ends=[38.8],
        visibility_start=37.0,
        visibility_end=44.0,
    )
    line2 = TargetLine(
        line_index=3,
        start=40.85,
        end=42.75,
        text="my knees for you",
        words=["my", "knees", "for", "you"],
        y=130,
        word_rois=[(0, 0, 2, 2)],
        word_starts=[40.85],
        word_ends=[42.75],
        visibility_start=37.0,
        visibility_end=44.0,
    )
    line3 = TargetLine(
        line_index=4,
        start=42.75,
        end=44.1,
        text="Don't say thank you or please",
        words=["Don't", "say", "thank", "you", "or", "please"],
        y=160,
        word_rois=[(0, 0, 2, 2)],
        word_starts=[42.75],
        word_ends=[44.1],
        visibility_start=37.0,
        visibility_end=44.0,
    )
    line4 = TargetLine(
        line_index=5,
        start=44.1,
        end=45.0,
        text="I do",
        words=["I", "do"],
        y=190,
        word_rois=[(0, 0, 2, 2)],
        word_starts=[44.1],
        word_ends=[45.0],
        visibility_start=37.0,
        visibility_end=44.0,
    )

    g_jobs = [
        (prev, 0.0, 0.0),
        (line1, 0.0, 0.0),
        (line2, 0.0, 0.0),
        (line3, 0.0, 0.0),
        (line4, 0.0, 0.0),
    ]
    _retime_compressed_shared_visibility_blocks(g_jobs)

    assert line2.word_starts is not None
    assert line2.word_starts[0] < 40.0


def test_retime_compressed_shared_visibility_blocks_skips_without_previous_anchor():
    line1 = TargetLine(
        line_index=1,
        start=17.0,
        end=17.0,
        text="White shirt now red",
        words=["White", "shirt", "now", "red"],
        y=20,
        word_rois=[(0, 0, 2, 2)],
        word_starts=[23.35],
        word_ends=[25.6],
        visibility_start=17.0,
        visibility_end=30.0,
    )
    line2 = TargetLine(
        line_index=2,
        start=17.0,
        end=17.0,
        text="My bloody nose",
        words=["My", "bloody", "nose"],
        y=90,
        word_rois=[(0, 0, 2, 2)],
        word_starts=[25.6],
        word_ends=[27.05],
        visibility_start=17.0,
        visibility_end=30.0,
    )
    line3 = TargetLine(
        line_index=3,
        start=17.0,
        end=17.0,
        text="Sleepin' you're on",
        words=["Sleepin'", "you're", "on"],
        y=150,
        word_rois=[(0, 0, 2, 2)],
        word_starts=[27.05],
        word_ends=[28.95],
        visibility_start=17.0,
        visibility_end=30.0,
    )

    g_jobs = [(line1, 0.0, 0.0), (line2, 0.0, 0.0), (line3, 0.0, 0.0)]
    _retime_compressed_shared_visibility_blocks(g_jobs)

    assert line1.word_starts is not None
    assert line1.word_starts[0] == 23.35


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
