from tools import analyze_fuzzy_segment_support_variants as tool


def test_best_segment_window_prefers_fuzzy_multilingual_span() -> None:
    result = tool._best_segment_window(
        line_text="De guayarte ma",
        segment_text="si te gana me dam dam dam debo hallarte mami",
        language="es",
    )

    assert result["span_text"] == "debo hallarte mami"
    assert result["joint_score"] >= 0.55


def test_best_segment_window_prefers_exact_span_when_available() -> None:
    result = tool._best_segment_window(
        line_text="I've been inclined",
        segment_text="I've been inclined to believe they never were",
        language="en",
    )

    assert result["span_text"] == "i've been inclined"
    assert result["joint_score"] == 1.0


def test_estimate_span_times_uses_token_offsets() -> None:
    start, end = tool._estimate_span_times(
        segment_start=28.0,
        segment_end=30.0,
        segment_text="si te gana me dam dam dam debo hallarte mami",
        span_start=7,
        span_end=10,
    )

    assert start == 29.4
    assert end == 30.0
