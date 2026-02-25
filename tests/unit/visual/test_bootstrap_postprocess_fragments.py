from y2karaoke.core.models import TargetLine
from y2karaoke.core.visual.bootstrap_postprocess import build_refined_lines_output
from y2karaoke.core.visual.bootstrap_postprocess_line_passes import (
    _neighbor_supports_fragment_tokens,
    _repair_repeat_cluster_tokenization_variants,
)
from y2karaoke.core.visual.bootstrap_postprocess import nearest_known_word_indices


def _make_line(line_index: int, start: float, text: str, conf: float) -> TargetLine:
    words = text.split()
    return TargetLine(
        line_index=line_index,
        start=start,
        end=start + 0.9,
        text=text,
        words=words,
        y=100,
        word_starts=[start + 0.05 * i for i in range(len(words))],
        word_ends=[start + 0.05 * i + 0.04 for i in range(len(words))],
        word_confidences=[conf] * len(words),
        word_rois=[(0, 0, 1, 1)] * len(words),
    )


def test_nearest_known_word_indices_mapping():
    prev_known, next_known = nearest_known_word_indices([1, 4], 6)
    assert prev_known == [-1, 1, 1, 1, 4, 4]
    assert next_known == [1, 1, 4, 4, 4, -1]


def test_neighbor_supports_split_word_fragment_tokens() -> None:
    assert _neighbor_supports_fragment_tokens(
        ["con", "ting"], ["counting", "dollars", "we'll", "be", "counting", "stars"]
    )


def test_build_refined_lines_output_removes_repeated_singular_plural_fragment_noise_lines():
    lines = [
        _make_line(1, 0.0, "dollar", 0.2),
        _make_line(2, 1.0, "said no more counting dollars", 0.85),
        _make_line(3, 2.0, "dollar", 0.2),
        _make_line(4, 3.0, "we'll be counting dollars", 0.8),
        _make_line(5, 4.0, "dollar", 0.2),
        _make_line(6, 5.0, "take that money watch it burn", 0.8),
    ]

    out = build_refined_lines_output(lines, artist=None, title=None)

    assert [ln["text"] for ln in out] == [
        "said no more counting dollars",
        "we'll be counting dollars",
        "take that money watch it burn",
    ]


def test_build_refined_lines_output_removes_two_occurrence_low_conf_fragment_when_supported():
    lines = [
        _make_line(1, 0.0, "dollar", 0.2),
        _make_line(2, 1.0, "said no more counting dollars", 0.85),
        _make_line(3, 2.0, "dollar", 0.2),
        _make_line(4, 3.0, "take that money watch it burn", 0.8),
    ]

    out = build_refined_lines_output(lines, artist=None, title=None)

    assert [ln["text"] for ln in out] == [
        "said no more counting dollars",
        "take that money watch it burn",
    ]


def _line_dict(text: str, start: float, conf: float = 0.3) -> dict:
    words = text.split()
    return {
        "line_index": 1,
        "text": text,
        "start": start,
        "end": start + 1.0,
        "confidence": conf,
        "words": [
            {
                "word_index": i + 1,
                "text": w,
                "start": start + i * 0.1,
                "end": start + i * 0.1 + 0.08,
                "confidence": conf,
            }
            for i, w in enumerate(words)
        ],
        "_reconstruction_meta": {
            "uncertainty_score": 0.3,
            "selected_text_support_ratio": 0.5,
        },
    }


def test_repair_repeat_cluster_tokenization_variants_merges_split_word_and_pluralizes():
    lines = [
        _line_dict("Said no more con ting dollars", 0.0, 0.45),
        _line_dict("Said no more counting dollar", 90.0, 0.25),
        _line_dict("Said no more con ting dollars", 180.0, 0.25),
    ]

    _repair_repeat_cluster_tokenization_variants(lines, [0, 1, 2])

    assert [ln["text"].lower() for ln in lines] == [
        "said no more counting dollars",
        "said no more counting dollars",
        "said no more counting dollars",
    ]


def test_repair_repeat_cluster_tokenization_variants_repairs_ocr_confusable_token():
    lines = [
        _line_dict("But baby I've been I've been", 0.0, 0.45),
        _line_dict("But baby I've been l've been", 90.0, 0.25),
        _line_dict("But baby I've been I've been", 180.0, 0.4),
    ]

    _repair_repeat_cluster_tokenization_variants(lines, [0, 1, 2])

    assert [ln["text"].lower() for ln in lines] == [
        "but baby i've been i've been",
        "but baby i've been i've been",
        "but baby i've been i've been",
    ]
