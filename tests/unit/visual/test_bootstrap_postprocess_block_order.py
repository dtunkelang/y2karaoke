from y2karaoke.core.visual.bootstrap_postprocess_line_passes import (
    _consolidate_block_first_fragment_rows,
    _reorder_clean_visibility_blocks,
)


def test_reorder_clean_visibility_blocks_sorts_by_block_then_y() -> None:
    lines_out = [
        {
            "line_index": 1,
            "text": "Dreaming about the things",
            "start": 10.1,
            "end": 11.1,
            "y": 160,
            "_visibility_start": 10.0,
            "_visibility_end": 18.0,
            "words": [{"text": "Dreaming", "start": 10.1, "end": 10.2}],
        },
        {
            "line_index": 2,
            "text": "we could be",
            "start": 16.8,
            "end": 17.55,
            "y": 230,
            "_visibility_start": 10.0,
            "_visibility_end": 18.0,
            "words": [{"text": "we", "start": 16.8, "end": 16.9}],
        },
        {
            "line_index": 3,
            "text": "Lately I've been",
            "start": 17.55,
            "end": 18.25,
            "y": 20,
            "_visibility_start": 10.0,
            "_visibility_end": 18.0,
            "words": [{"text": "Lately", "start": 17.55, "end": 17.65}],
        },
        {
            "line_index": 4,
            "text": "losing sleep",
            "start": 18.25,
            "end": 18.5,
            "y": 90,
            "_visibility_start": 10.0,
            "_visibility_end": 18.0,
            "words": [{"text": "losing", "start": 18.25, "end": 18.35}],
        },
        {
            "line_index": 5,
            "text": "Later block top",
            "start": 23.5,
            "end": 24.5,
            "y": 20,
            "_visibility_start": 23.5,
            "_visibility_end": 45.0,
            "words": [{"text": "Later", "start": 23.5, "end": 23.6}],
        },
    ]

    _reorder_clean_visibility_blocks(lines_out)

    assert [ln["text"] for ln in lines_out] == [
        "Lately I've been",
        "losing sleep",
        "Dreaming about the things",
        "we could be",
        "Later block top",
    ]


def test_consolidate_block_first_fragment_rows_merges_suffix_fragment() -> None:
    lines_out = [
        {
            "line_index": 1,
            "text": "But baby I've been",
            "start": 17.8,
            "end": 19.0,
            "y": 20,
            "confidence": 0.5,
            "words": [
                {"word_index": 1, "text": "But", "start": 17.8, "end": 18.0},
                {"word_index": 2, "text": "baby", "start": 18.0, "end": 18.3},
                {"word_index": 3, "text": "I've", "start": 18.3, "end": 18.6},
                {"word_index": 4, "text": "been", "start": 18.6, "end": 19.0},
            ],
            "_reconstruction_meta": {"block_first": {"block_id": 1, "row_order": 0}},
        },
        {
            "line_index": 2,
            "text": "I've been",
            "start": 17.9,
            "end": 18.8,
            "y": 90,
            "confidence": 0.4,
            "words": [
                {"word_index": 1, "text": "I've", "start": 17.9, "end": 18.3},
                {"word_index": 2, "text": "been", "start": 18.3, "end": 18.8},
            ],
            "_reconstruction_meta": {"block_first": {"block_id": 1, "row_order": 1}},
        },
        {
            "line_index": 3,
            "text": "praying hard",
            "start": 18.0,
            "end": 19.2,
            "y": 160,
            "confidence": 0.6,
            "words": [
                {"word_index": 1, "text": "praying", "start": 18.0, "end": 18.6},
                {"word_index": 2, "text": "hard", "start": 18.6, "end": 19.2},
            ],
            "_reconstruction_meta": {"block_first": {"block_id": 1, "row_order": 2}},
        },
    ]

    _consolidate_block_first_fragment_rows(lines_out)

    assert [ln["text"] for ln in lines_out] == [
        "But baby I've been I've been",
        "praying hard",
    ]
