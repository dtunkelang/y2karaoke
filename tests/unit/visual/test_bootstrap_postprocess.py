from y2karaoke.core.models import TargetLine
from y2karaoke.core.visual.bootstrap_postprocess import (
    build_refined_lines_output,
    nearest_known_word_indices,
)


def test_nearest_known_word_indices_mapping():
    prev_known, next_known = nearest_known_word_indices([1, 4], 6)
    assert prev_known == [-1, 1, 1, 1, 4, 4]
    assert next_known == [1, 1, 4, 4, 4, -1]


def test_build_refined_lines_output_filters_title_artist():
    lines = [
        TargetLine(
            line_index=1,
            start=8.0,
            end=10.0,
            text="Song Title",
            words=["Song", "Title"],
            y=50,
            word_starts=[8.1, 8.7],
            word_ends=[8.5, 9.2],
            word_rois=[(0, 0, 1, 1), (1, 0, 1, 1)],
        ),
        TargetLine(
            line_index=2,
            start=11.0,
            end=13.0,
            text="real lyric",
            words=["real", "lyric"],
            y=60,
            word_starts=[11.1, 11.8],
            word_ends=[11.5, 12.4],
            word_rois=[(0, 0, 1, 1), (1, 0, 1, 1)],
        ),
    ]
    out = build_refined_lines_output(lines, artist="Artist Name", title="Song Title")
    assert len(out) == 1
    assert out[0]["text"] == "real lyric"


def test_build_refined_lines_output_splits_fused_word_tokens():
    lines = [
        TargetLine(
            line_index=1,
            start=45.0,
            end=46.2,
            text="What I want",
            words=["What", "Iwant"],
            y=15,
            word_starts=[45.05, 45.6],
            word_ends=[45.55, 46.15],
            word_rois=[(0, 0, 1, 1), (1, 0, 1, 1)],
            word_confidences=[0.25, 0.25],
        )
    ]

    out = build_refined_lines_output(lines, artist=None, title=None)
    assert len(out) == 1
    assert [w["text"] for w in out[0]["words"]] == ["What", "I", "want"]


def test_build_refined_lines_output_delays_short_interstitial_line():
    lines = [
        TargetLine(
            line_index=1,
            start=40.7,
            end=43.1,
            text="Don't say thank you or please",
            words=["Don't", "say", "thank", "you", "or", "please"],
            y=10,
            word_starts=[40.7, 41.1, 41.4, 41.9, 42.4, 42.85],
            word_ends=[41.0, 41.35, 41.75, 42.25, 42.75, 43.1],
            word_rois=[(0, 0, 1, 1)] * 6,
            word_confidences=[0.25] * 6,
        ),
        TargetLine(
            line_index=2,
            start=43.3,
            end=44.1,
            text="I do",
            words=["I", "do"],
            y=20,
            word_starts=[43.3, 43.7],
            word_ends=[43.65, 44.1],
            word_rois=[(0, 0, 1, 1)] * 2,
            word_confidences=[0.25] * 2,
        ),
        TargetLine(
            line_index=3,
            start=45.05,
            end=46.15,
            text="What I want",
            words=["What", "I", "want"],
            y=30,
            word_starts=[45.05, 45.45, 45.8],
            word_ends=[45.35, 45.75, 46.15],
            word_rois=[(0, 0, 1, 1)] * 3,
            word_confidences=[0.25] * 3,
        ),
    ]

    out = build_refined_lines_output(lines, artist=None, title=None)
    assert len(out) == 3
    assert out[1]["text"] == "I do"
    assert out[1]["start"] >= 43.7
    assert out[1]["end"] <= 44.95


def test_build_refined_lines_output_rebalances_compressed_middle_four_line_sequence():
    lines = [
        TargetLine(
            line_index=1,
            start=51.2,
            end=52.15,
            text="So you're a tough guy",
            words=["So", "you're", "a", "tough", "guy"],
            y=10,
            word_starts=[51.2, 51.4, 51.55, 51.75, 51.95],
            word_ends=[51.35, 51.52, 51.7, 51.9, 52.15],
            word_rois=[(0, 0, 1, 1)] * 5,
            word_confidences=[0.25] * 5,
        ),
        TargetLine(
            line_index=2,
            start=52.15,
            end=52.85,
            text="Like it really rough guy",
            words=["Like", "it", "really", "rough", "guy"],
            y=20,
            word_starts=[52.15, 52.3, 52.4, 52.55, 52.7],
            word_ends=[52.27, 52.38, 52.53, 52.68, 52.85],
            word_rois=[(0, 0, 1, 1)] * 5,
            word_confidences=[0.25] * 5,
        ),
        TargetLine(
            line_index=3,
            start=52.85,
            end=53.9,
            text="Just can't get enough guy",
            words=["Just", "can't", "get", "enough", "guy"],
            y=30,
            word_starts=[52.85, 53.0, 53.2, 53.4, 53.65],
            word_ends=[52.97, 53.15, 53.35, 53.58, 53.9],
            word_rois=[(0, 0, 1, 1)] * 5,
            word_confidences=[0.25] * 5,
        ),
        TargetLine(
            line_index=4,
            start=55.5,
            end=56.95,
            text="Chest always so puffed guy",
            words=["Chest", "always", "so", "puffed", "guy"],
            y=40,
            word_starts=[55.5, 55.8, 56.1, 56.4, 56.7],
            word_ends=[55.75, 56.0, 56.25, 56.55, 56.95],
            word_rois=[(0, 0, 1, 1)] * 5,
            word_confidences=[0.25] * 5,
        ),
    ]

    out = build_refined_lines_output(lines, artist=None, title=None)
    assert len(out) == 4
    assert out[1]["start"] >= 52.5
    assert out[2]["start"] >= 54.0
