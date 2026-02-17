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
