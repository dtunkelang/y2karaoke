from y2karaoke.core.models import TargetLine
from y2karaoke.core.visual.refinement import (
    _retime_compressed_shared_visibility_blocks,
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
