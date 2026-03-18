from y2karaoke.core.components.whisper import (
    whisper_segment_assignment_experimental as exp,
)
from y2karaoke.core.components.whisper.whisper_mapping_runtime_config import (
    SegmentAssignmentRuntimeConfig,
)


def test_experimental_segment_score_downweights_function_word_only_overlap_in_placeholder_heavy_segments() -> (
    None
):
    stats = exp._score_segment_for_line_experimental(
        ["je", "remue", "le", "ciel", "le", "jour", "la", "nuit"],
        ["[vocal]", "[vocal]", "[vocal]", "[vocal]", "je", "le"],
    )

    assert float(stats["raw_score"]) > float(stats["score"])
    assert stats["content_hits"] == 0
    assert stats["function_hits"] >= 2
    assert float(stats["score"]) < 0.2


def test_experimental_segment_score_preserves_default_score_for_clean_segments() -> (
    None
):
    stats = exp._score_segment_for_line_experimental(
        ["stylin", "wilin", "livin", "it", "up", "in", "the", "city"],
        ["saturday", "night", "and", "we", "in", "the", "spot"],
    )

    assert float(stats["raw_score"]) == float(stats["score"])


def test_experimental_assigner_limits_forward_jump_without_content_hits() -> None:
    lrc_lines_words = [
        [(0, "oh"), (1, "ma"), (2, "douce"), (3, "souffrance")],
        [
            (4, "je"),
            (5, "remue"),
            (6, "le"),
            (7, "ciel"),
            (8, "le"),
            (9, "jour"),
            (10, "la"),
            (11, "nuit"),
        ],
    ]
    seg_word_bags = [
        ["ma", "douce", "souffrance"],
        ["[vocal]", "[vocal]"],
        ["je", "[vocal]", "[vocal]", "revance", "[vocal]"],
        ["[vocal]", "[vocal]"],
        ["et", "dans", "le", "bruit", "je", "cours", "et", "j", "ai", "peur"],
    ]
    seg_durations = [2.0, 1.0, 18.0, 1.0, 4.0]
    config = SegmentAssignmentRuntimeConfig()

    line_to_seg = exp._assign_lrc_lines_to_segments_experimental(
        lrc_lines_words=lrc_lines_words,
        seg_word_bags=seg_word_bags,
        seg_durations=seg_durations,
        config=config,
    )

    assert line_to_seg[0] == 0
    assert line_to_seg[1] <= 2


def test_experimental_assigner_falls_back_when_placeholder_positive_lines_are_sparse() -> (
    None
):
    trace_rows = [
        {
            "scores": [
                {"placeholder_ratio": 0.0, "score": 0.2},
                {"placeholder_ratio": 1.0, "score": 0.0},
            ]
        }
        for _ in range(10)
    ]
    trace_rows[0]["scores"][1]["score"] = 0.1

    assert not exp._should_keep_experimental_assignments(trace_rows, line_count=10)


def test_experimental_assigner_keeps_placeholder_heavy_song_patterns() -> None:
    trace_rows = [
        {
            "scores": [
                {"placeholder_ratio": 0.0, "score": 0.2},
                {"placeholder_ratio": 0.9, "score": 0.1},
            ]
        }
        for _ in range(10)
    ]
    for row in trace_rows[2:]:
        row["scores"][1]["score"] = 0.0

    assert exp._should_keep_experimental_assignments(trace_rows, line_count=10)
