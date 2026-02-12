from y2karaoke.core.components.whisper import whisper_mapping_pipeline as wmp
from y2karaoke.core.components.whisper.whisper_dtw_tokens import _LineMappingContext
from y2karaoke.core.components.alignment.timing_models import TranscriptionSegment
from y2karaoke.core.models import Line, Word


def _segments():
    return [
        TranscriptionSegment(start=10.0, end=12.0, text="a", words=[]),
        TranscriptionSegment(start=16.0, end=18.0, text="b", words=[]),
        TranscriptionSegment(start=22.0, end=24.0, text="c", words=[]),
        TranscriptionSegment(start=32.0, end=34.0, text="d", words=[]),
    ]


def test_should_override_line_segment_allows_local_jump():
    assert wmp._should_override_line_segment(
        current_segment=1,
        override_segment=2,
        override_hits=1,
        line_word_count=4,
        line_anchor_time=20.0,
        segments=_segments(),
    )


def test_should_override_line_segment_blocks_far_weak_jump():
    assert not wmp._should_override_line_segment(
        current_segment=0,
        override_segment=3,
        override_hits=1,
        line_word_count=5,
        line_anchor_time=10.0,
        segments=_segments(),
    )


def test_should_override_line_segment_allows_far_strong_but_bounded_jump():
    assert wmp._should_override_line_segment(
        current_segment=1,
        override_segment=3,
        override_hits=4,
        line_word_count=5,
        line_anchor_time=20.0,
        segments=_segments(),
    )


def test_should_override_line_segment_blocks_anchor_far_even_if_segment_jump_ok():
    assert not wmp._should_override_line_segment(
        current_segment=1,
        override_segment=3,
        override_hits=5,
        line_word_count=5,
        line_anchor_time=8.0,
        segments=_segments(),
    )


def test_prepare_line_context_caps_anchor_runaway_from_lrc_start():
    ctx = _LineMappingContext(
        all_words=[],
        segments=_segments(),
        word_segment_idx={},
        language="eng-Latn",
        total_lrc_words=1,
        total_whisper_words=1,
        last_line_start=80.0,
        prev_line_end=82.0,
    )
    line = Line(words=[Word("x", start_time=40.0, end_time=41.0)])

    _seg, anchor, shift = wmp._prepare_line_context(ctx, line)

    assert anchor == 46.0
    assert shift == 6.0


def test_prepare_line_context_keeps_nearby_anchor():
    ctx = _LineMappingContext(
        all_words=[],
        segments=_segments(),
        word_segment_idx={},
        language="eng-Latn",
        total_lrc_words=1,
        total_whisper_words=1,
        last_line_start=45.0,
        prev_line_end=46.0,
    )
    line = Line(words=[Word("x", start_time=40.0, end_time=41.0)])

    _seg, anchor, shift = wmp._prepare_line_context(ctx, line)

    assert anchor == 46.0
    assert shift == 6.0


def test_clamp_match_window_to_anchor_limits_large_forward_drift():
    start, end = wmp._clamp_match_window_to_anchor(
        actual_start=120.0,
        actual_end=124.0,
        line_anchor_time=55.0,
    )
    assert start == 63.0
    assert end == 67.0


def test_clamp_match_window_to_anchor_preserves_window_when_reasonable():
    start, end = wmp._clamp_match_window_to_anchor(
        actual_start=58.0,
        actual_end=61.0,
        line_anchor_time=55.0,
    )
    assert start == 58.0
    assert end == 61.0


def test_clamp_line_shift_vs_original_limits_large_forward_shift():
    original = Line(words=[Word("a", start_time=50.0, end_time=51.0)])
    mapped = Line(words=[Word("a", start_time=90.0, end_time=91.0)])

    clamped = wmp._clamp_line_shift_vs_original(mapped, original)

    assert clamped.start_time == 55.0
    assert clamped.end_time == 56.0


def test_clamp_line_shift_vs_original_preserves_reasonable_shift():
    original = Line(
        words=[
            Word("a", start_time=50.0, end_time=50.1),
            Word("b", start_time=50.1, end_time=50.2),
            Word("c", start_time=50.2, end_time=50.3),
            Word("d", start_time=50.3, end_time=50.4),
            Word("e", start_time=50.4, end_time=50.5),
            Word("f", start_time=50.5, end_time=50.6),
            Word("g", start_time=50.6, end_time=50.7),
            Word("h", start_time=50.7, end_time=50.8),
        ]
    )
    mapped = Line(
        words=[
            Word("a", start_time=57.0, end_time=57.1),
            Word("b", start_time=57.1, end_time=57.2),
            Word("c", start_time=57.2, end_time=57.3),
            Word("d", start_time=57.3, end_time=57.4),
            Word("e", start_time=57.4, end_time=57.5),
            Word("f", start_time=57.5, end_time=57.6),
            Word("g", start_time=57.6, end_time=57.7),
            Word("h", start_time=57.7, end_time=57.8),
        ]
    )

    clamped = wmp._clamp_line_shift_vs_original(mapped, original)

    assert clamped.start_time == 55.0
    assert clamped.end_time == 55.8


def test_clamp_line_duration_vs_original_limits_bleed():
    original = Line(words=[Word("a", start_time=20.0, end_time=22.0)])
    mapped = Line(
        words=[
            Word("a", start_time=20.0, end_time=21.0),
            Word("b", start_time=21.0, end_time=27.0),
        ]
    )

    clamped = wmp._clamp_line_duration_vs_original(
        mapped, original, next_original_start=22.5
    )

    assert clamped.end_time <= 23.4


def test_clamp_line_duration_vs_original_preserves_reasonable_span():
    original = Line(words=[Word("a", start_time=20.0, end_time=22.0)])
    mapped = Line(
        words=[
            Word("a", start_time=20.0, end_time=20.8),
            Word("b", start_time=20.8, end_time=22.6),
        ]
    )

    clamped = wmp._clamp_line_duration_vs_original(
        mapped, original, next_original_start=22.5
    )

    assert clamped.end_time == 22.6


def test_fallback_unmatched_line_duration_enforces_min_for_multiword_lines():
    line = Line(
        words=[
            Word("Hands", start_time=10.0, end_time=10.06),
            Word("off", start_time=10.07, end_time=10.13),
            Word("Gabriela", start_time=10.14, end_time=10.2),
        ]
    )

    fallback = wmp._fallback_unmatched_line_duration(line)

    assert fallback == 0.48


def test_fallback_unmatched_line_duration_keeps_short_single_word():
    line = Line(words=[Word("ooh", start_time=10.0, end_time=10.15)])

    fallback = wmp._fallback_unmatched_line_duration(line)

    assert fallback == 0.22
