from y2karaoke.core.models import TargetLine
from y2karaoke.core.visual import bootstrap_postprocess
from y2karaoke.core.visual.bootstrap_postprocess import build_refined_lines_output
from y2karaoke.core.visual.bootstrap_postprocess_line_passes import (
    _canonicalize_local_chant_token_variants,
    _neighbor_supports_fragment_tokens,
    _repair_repeat_cluster_tokenization_variants,
    _trim_short_adlib_tails,
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


def test_build_refined_lines_output_removes_high_conf_repeated_fragment_when_supported():
    lines = [
        _make_line(1, 0.0, "just watch", 0.85),
        _make_line(2, 1.0, "don't believe me just watch", 0.95),
        _make_line(3, 2.0, "just watch", 0.85),
        _make_line(4, 3.0, "don't believe me just watch", 0.95),
        _make_line(5, 4.0, "just watch", 0.85),
        _make_line(6, 5.0, "don't believe me just watch", 0.95),
        _make_line(7, 6.0, "just watch", 0.85),
        _make_line(8, 7.0, "don't believe me just watch", 0.95),
        _make_line(9, 8.0, "just watch", 0.85),
        _make_line(10, 9.0, "don't believe me just watch", 0.95),
    ]

    out = build_refined_lines_output(lines, artist=None, title=None)

    assert all(ln["text"] == "don't believe me just watch" for ln in out)


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


def test_remove_high_repeat_nonlexical_chant_noise_lines_removes_doh_family(
    monkeypatch,
):
    monkeypatch.setattr(
        bootstrap_postprocess,
        "_is_spelled_word",
        lambda token: token.lower() in {"watch"},
    )
    lines_out = [
        {
            "line_index": 1,
            "text": "doh doh doh",
            "start": 10.0,
            "end": 10.8,
            "confidence": 0.95,
            "words": [{"text": "doh"}, {"text": "doh"}, {"text": "doh"}],
        },
        {
            "line_index": 2,
            "text": "Dohi doh doh",
            "start": 20.0,
            "end": 20.8,
            "confidence": 0.9,
            "words": [{"text": "Dohi"}, {"text": "doh"}, {"text": "doh"}],
        },
        {
            "line_index": 3,
            "text": "doh doh",
            "start": 30.0,
            "end": 30.6,
            "confidence": 0.9,
            "words": [{"text": "doh"}, {"text": "doh"}],
        },
        {
            "line_index": 4,
            "text": "doh doh doh",
            "start": 40.0,
            "end": 40.8,
            "confidence": 0.9,
            "words": [{"text": "doh"}, {"text": "doh"}, {"text": "doh"}],
        },
        {
            "line_index": 5,
            "text": "doh doh",
            "start": 50.0,
            "end": 50.6,
            "confidence": 0.9,
            "words": [{"text": "doh"}, {"text": "doh"}],
        },
        {
            "line_index": 6,
            "text": "just watch",
            "start": 51.0,
            "end": 51.8,
            "confidence": 0.7,
            "words": [{"text": "just"}, {"text": "watch"}],
        },
    ]

    bootstrap_postprocess._remove_high_repeat_nonlexical_chant_noise_lines(lines_out)

    assert [ln["text"] for ln in lines_out] == ["just watch"]


def test_remove_high_repeat_nonlexical_chant_noise_lines_keeps_real_word_chants(
    monkeypatch,
):
    monkeypatch.setattr(bootstrap_postprocess, "_is_spelled_word", lambda token: True)
    lines_out = [
        {
            "line_index": i + 1,
            "text": "hey hey hey",
            "start": 10.0 * (i + 1),
            "end": 10.0 * (i + 1) + 0.8,
            "confidence": 0.9,
            "words": [{"text": "hey"}, {"text": "hey"}, {"text": "hey"}],
        }
        for i in range(5)
    ]

    bootstrap_postprocess._remove_high_repeat_nonlexical_chant_noise_lines(lines_out)

    assert len(lines_out) == 5


def test_remove_high_repeat_nonlexical_chant_noise_lines_handles_dictionary_doh(
    monkeypatch,
):
    monkeypatch.setattr(
        bootstrap_postprocess, "_is_spelled_word", lambda token: token.lower() == "doh"
    )
    lines_out = [
        {
            "line_index": i + 1,
            "text": "doh doh doh",
            "start": 10.0 * (i + 1),
            "end": 10.0 * (i + 1) + 0.7,
            "confidence": 0.9,
            "words": [{"text": "doh"}, {"text": "doh"}, {"text": "doh"}],
        }
        for i in range(8)
    ]
    lines_out.append(
        {
            "line_index": 99,
            "text": "just watch",
            "start": 90.0,
            "end": 90.8,
            "confidence": 0.7,
            "words": [{"text": "just"}, {"text": "watch"}],
        }
    )

    bootstrap_postprocess._remove_high_repeat_nonlexical_chant_noise_lines(lines_out)

    assert [ln["text"] for ln in lines_out] == ["just watch"]


def test_canonicalize_local_chant_token_variants_normalizes_dohi_family() -> None:
    lines_out = [
        _line_dict("Dohi doh doh doh", 0.0, 0.9),
        _line_dict("hey hey hey oh", 2.0, 0.9),
    ]

    _canonicalize_local_chant_token_variants(lines_out)

    assert lines_out[0]["text"] == "Doh doh doh doh"
    assert lines_out[1]["text"] == "hey hey hey oh"


def test_trim_short_adlib_tails_trims_supported_come_on_suffix() -> None:
    lines_out = [
        _line_dict("Don't believe me", 0.0, 0.6),
        _line_dict("just watch Come on", 0.8, 0.8),
        _line_dict("Don't believe me just watch", 1.0, 1.0),
    ]
    lines_out[1]["end"] = lines_out[1]["start"] + 0.7

    _trim_short_adlib_tails(lines_out)

    assert [ln["text"] for ln in lines_out] == [
        "Don't believe me",
        "just watch",
        "Don't believe me just watch",
    ]


def test_trim_short_adlib_tails_keeps_unsuported_short_line() -> None:
    lines_out = [
        _line_dict("Come on baby", 0.0, 0.9),
        _line_dict("Shake it now", 1.0, 0.8),
        _line_dict("Move your feet", 2.0, 0.8),
    ]

    _trim_short_adlib_tails(lines_out)

    assert [ln["text"] for ln in lines_out] == [
        "Come on baby",
        "Shake it now",
        "Move your feet",
    ]
