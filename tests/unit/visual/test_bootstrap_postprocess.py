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
