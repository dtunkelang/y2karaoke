from types import SimpleNamespace

from tools.analyze_transcription_variants import _summarize_variant


def test_summarize_variant_flags_dominant_single_segment() -> None:
    summary = _summarize_variant(
        name="test",
        language="en",
        segments=[SimpleNamespace(start=0.0, end=18.6, text="Take on me, take me on")],
        word_count=14,
        text_attr="text",
    )

    assert summary.segment_count == 1
    assert summary.word_count == 14
    assert summary.dominant_single_segment is True
    assert summary.max_segment_ratio == 1.0


def test_summarize_variant_keeps_multi_segment_variant_clean() -> None:
    summary = _summarize_variant(
        name="test",
        language="en",
        segments=[
            SimpleNamespace(start=0.0, end=2.4, text="Sweet Caroline"),
            SimpleNamespace(start=5.5, end=8.6, text="Good times never seemed so good"),
            SimpleNamespace(start=12.0, end=13.8, text="I've been inclined"),
        ],
        word_count=23,
        text_attr="text",
    )

    assert summary.segment_count == 3
    assert summary.dominant_single_segment is False
    assert summary.max_segment_ratio < 0.5
