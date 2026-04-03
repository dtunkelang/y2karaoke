from y2karaoke.core.components.whisper import whisper_mapping_pipeline as wmp
from y2karaoke.core.components.whisper import (
    whisper_mapping_pipeline_candidates as candidates,
)
from y2karaoke.core.components.whisper import (
    whisper_mapping_pipeline_matching as matching,
)
from y2karaoke.core.components.whisper.whisper_dtw_tokens import _LineMappingContext
from y2karaoke.core.components.alignment.timing_models import (
    TranscriptionSegment,
    TranscriptionWord,
)
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


def test_prepare_line_context_with_details_reports_fallbacks():
    ctx = _LineMappingContext(
        all_words=[],
        segments=_segments(),
        word_segment_idx={},
        language="eng-Latn",
        total_lrc_words=1,
        total_whisper_words=1,
        current_segment=1,
        last_line_start=45.0,
        prev_line_end=46.0,
    )
    line = Line(words=[Word("x", start_time=40.0, end_time=41.0)])

    seg, anchor, shift, details = wmp._prepare_line_context_with_details(ctx, line)

    assert seg is None
    assert anchor == 46.0
    assert shift == 6.0
    assert details == {
        "text_choice_segment": None,
        "time_fallback_segment": 3,
        "cleared_for_monotonicity": True,
    }


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

    assert clamped.start_time == 58.0
    assert clamped.end_time == 59.0


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

    assert clamped.start_time == 57.0
    assert clamped.end_time == 57.8


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


def test_register_word_match_advances_current_segment_to_line_override():
    whisper_word = TranscriptionWord(text="foo", start=12.0, end=12.4)
    ctx = _LineMappingContext(
        all_words=[whisper_word],
        segments=[],
        word_segment_idx={0: 5},
        language="eng-Latn",
        total_lrc_words=1,
        total_whisper_words=1,
        current_segment=4,
    )
    line_matches = []
    line_match_intervals = {}
    line_match_word_indices = {}
    line_last_idx_ref = [None]

    candidates._register_word_match(
        ctx,
        line_idx=0,
        word=Word("foo", start_time=12.0, end_time=12.4),
        best_word=whisper_word,
        best_idx=0,
        candidates=[(whisper_word, 0)],
        line_segment=7,
        line_matches=line_matches,
        line_match_intervals=line_match_intervals,
        line_match_word_indices=line_match_word_indices,
        word_idx=0,
        line_last_idx_ref=line_last_idx_ref,
        phonetic_similarity_fn=lambda *_: 1.0,
    )

    assert ctx.current_segment == 7


def test_register_word_match_keeps_later_best_segment_when_ahead_of_override():
    whisper_word = TranscriptionWord(text="foo", start=12.0, end=12.4)
    ctx = _LineMappingContext(
        all_words=[whisper_word],
        segments=[],
        word_segment_idx={0: 8},
        language="eng-Latn",
        total_lrc_words=1,
        total_whisper_words=1,
        current_segment=4,
    )

    candidates._register_word_match(
        ctx,
        line_idx=0,
        word=Word("foo", start_time=12.0, end_time=12.4),
        best_word=whisper_word,
        best_idx=0,
        candidates=[(whisper_word, 0)],
        line_segment=7,
        line_matches=[],
        line_match_intervals={},
        line_match_word_indices={},
        word_idx=0,
        line_last_idx_ref=[None],
        phonetic_similarity_fn=lambda *_: 1.0,
    )

    assert ctx.current_segment == 8


def test_candidate_is_lexically_plausible_rejects_placeholder():
    assert not candidates._candidate_is_lexically_plausible(
        target_token="lo",
        candidate_text="[VOCAL]",
        phonetic_similarity_fn=lambda *_: 0.0,
        target_text="Lo",
        language="eng-Latn",
    )


def test_select_best_candidate_prefers_lexical_match_over_tighter_time_fit():
    ctx = _LineMappingContext(
        all_words=[
            TranscriptionWord(text="la", start=62.83, end=63.03),
            TranscriptionWord(text="Fuck", start=63.05, end=63.25),
        ],
        segments=[],
        word_segment_idx={0: 0, 1: 0},
        language="eng-Latn",
        total_lrc_words=2,
        total_whisper_words=2,
    )
    word = Word("Fuck", start_time=61.737, end_time=61.9)

    best_word, best_idx = candidates._select_best_candidate(
        ctx,
        [(ctx.all_words[0], 0), (ctx.all_words[1], 1)],
        word,
        line_shift=1.093,
        line_segment=0,
        line_anchor_time=62.83,
        lrc_idx_opt=0,
        time_drift_threshold=0.8,
        phonetic_similarity_fn=lambda left, right, _lang: (
            1.0 if left.lower().strip("',.!?") == right.lower().strip("',.!?") else 0.0
        ),
    )

    assert best_idx == 1
    assert best_word.text == "Fuck"


def test_select_best_candidate_penalizes_backtracking_in_dense_line():
    ctx = _LineMappingContext(
        all_words=[
            TranscriptionWord(text="late", start=24.68, end=24.9),
            TranscriptionWord(text="mid", start=25.86, end=26.5),
            TranscriptionWord(text="future", start=26.6, end=27.12),
        ],
        segments=[],
        word_segment_idx={0: 2, 1: 2, 2: 2},
        language="eng-Latn",
        total_lrc_words=20,
        total_whisper_words=20,
    )
    word = Word("de", start_time=25.457, end_time=25.7)
    whisper_candidates = [
        (TranscriptionWord(text=f"filler{i}", start=20.0 + i, end=20.1 + i), i + 10)
        for i in range(10)
    ]
    whisper_candidates.extend(
        [
            (ctx.all_words[2], 14),
            (ctx.all_words[1], 12),
            (ctx.all_words[0], 6),
        ]
    )

    best_word, best_idx = candidates._select_best_candidate(
        ctx,
        whisper_candidates,
        word,
        line_shift=0.0,
        line_segment=2,
        line_anchor_time=24.612,
        lrc_idx_opt=3,
        prior_matched_word_idx=13,
        line_word_count=10,
        time_drift_threshold=0.8,
        phonetic_similarity_fn=lambda *_: 1.0,
    )

    assert best_idx == 12
    assert best_word.text == "mid"


def test_match_assigned_words_prefers_local_assigned_window_before_full_segment():
    ctx = _LineMappingContext(
        all_words=[
            TranscriptionWord(text="ya", start=21.5, end=22.08),
            TranscriptionWord(text="vi", start=22.08, end=22.16),
            TranscriptionWord(text="que", start=22.16, end=22.32),
            TranscriptionWord(text="estas", start=22.32, end=22.5),
            TranscriptionWord(text="solita,", start=22.5, end=23.02),
            TranscriptionWord(text="acompáñame,", start=23.08, end=23.9),
            TranscriptionWord(text="la", start=24.1, end=24.6),
            TranscriptionWord(text="me", start=27.48, end=27.58),
        ],
        segments=[],
        word_segment_idx={idx: 2 for idx in range(8)},
        language="eng-Latn",
        total_lrc_words=6,
        total_whisper_words=8,
    )
    line = Line(
        words=[
            Word("Ya", start_time=22.458, end_time=22.7),
            Word("vi", start_time=22.788, end_time=23.0),
            Word("que", start_time=23.117, end_time=23.3),
            Word("estás", start_time=23.447, end_time=23.7),
            Word("solita,", start_time=23.777, end_time=24.0),
            Word("acompáñame", start_time=24.106, end_time=24.612),
        ]
    )
    lrc_index_by_loc = {(0, idx): idx for idx in range(len(line.words))}
    lrc_assignments = {
        0: [0],
        1: [0],
        2: [1],
        3: [2],
        4: [3],
        5: [4],
    }
    line_matches = []
    line_match_intervals = {}
    line_match_word_indices = {}
    line_last_idx_ref = [None]

    matching._match_assigned_words(
        ctx,
        line_idx=0,
        line=line,
        lrc_index_by_loc=lrc_index_by_loc,
        lrc_assignments=lrc_assignments,
        line_segment=2,
        line_anchor_time=22.458,
        line_shift=0.0,
        line_matches=line_matches,
        line_match_intervals=line_match_intervals,
        line_match_word_indices=line_match_word_indices,
        line_last_idx_ref=line_last_idx_ref,
        filter_and_order_candidates_fn=wmp._filter_and_order_candidates,
        select_best_candidate_fn=wmp._select_best_candidate,
        register_word_match_fn=wmp._register_word_match,
    )

    assert line_match_word_indices == {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
    assert 7 not in ctx.used_word_indices
    assert line_last_idx_ref[0] == 5


def test_assemble_mapped_line_releases_trailing_outlier_match_from_cursor():
    ctx = _LineMappingContext(
        all_words=[
            TranscriptionWord(text="ya", start=21.5, end=22.08),
            TranscriptionWord(text="vi", start=22.08, end=22.16),
            TranscriptionWord(text="solita", start=23.08, end=23.9),
            TranscriptionWord(text="me", start=27.48, end=27.58),
        ],
        segments=[],
        word_segment_idx={0: 2, 1: 2, 2: 2, 3: 2},
        language="eng-Latn",
        total_lrc_words=10,
        total_whisper_words=10,
        used_word_indices={0, 1, 2, 3},
    )
    line = Line(
        words=[
            Word("Ya", start_time=22.458, end_time=22.7),
            Word("vi", start_time=22.788, end_time=23.0),
            Word("solita,", start_time=23.777, end_time=24.0),
            Word("acompáñame", start_time=24.106, end_time=24.612),
        ]
    )
    line_matches = [
        (0, (21.5, 22.08)),
        (1, (22.08, 22.16)),
        (2, (23.08, 23.9)),
        (3, (27.48, 27.58)),
    ]
    line_match_intervals = dict(line_matches)
    line_match_word_indices = {0: 0, 1: 1, 2: 2, 3: 3}
    line_last_idx_ref = [3]

    mapped = wmp._assemble_mapped_line(
        ctx,
        line_idx=8,
        line=line,
        line_matches=line_matches,
        line_match_intervals=line_match_intervals,
        line_match_word_indices=line_match_word_indices,
        line_anchor_time=22.458,
        line_segment=2,
        line_last_idx_ref=line_last_idx_ref,
        next_original_start=24.68,
    )

    assert mapped.end_time == 24.622
    assert line_last_idx_ref[0] == 2
    assert ctx.next_word_idx_start == 3
    assert ctx.used_word_indices == {0, 1, 2}


def test_skip_collapsed_assigned_match_for_late_fragment_anchor():
    ctx = _LineMappingContext(
        all_words=[
            TranscriptionWord(text="It's", start=38.44, end=39.24),
            TranscriptionWord(text="funny", start=39.24, end=39.44),
            TranscriptionWord(text="how", start=39.44, end=39.68),
        ],
        segments=[],
        word_segment_idx={0: 7, 1: 8, 2: 8},
        language="eng-Latn",
        total_lrc_words=8,
        total_whisper_words=3,
    )
    line = Line(
        words=[
            Word("It's", start_time=33.1, end_time=33.45),
            Word("funny", start_time=33.45, end_time=33.7),
            Word("how", start_time=33.7, end_time=33.95),
            Word("we", start_time=33.95, end_time=34.25),
            Word("animate", start_time=34.25, end_time=35.75),
            Word("colorful", start_time=36.15, end_time=36.9),
            Word("objects", start_time=36.9, end_time=37.6),
            Word("saved", start_time=37.6, end_time=38.35),
        ]
    )
    lrc_index_by_loc = {(2, idx): idx for idx in range(len(line.words))}
    lrc_assignments = {idx: [min(idx, 2)] for idx in range(len(line.words))}

    assert matching._should_skip_collapsed_assigned_match(
        ctx=ctx,
        line_idx=2,
        line=line,
        lrc_index_by_loc=lrc_index_by_loc,
        lrc_assignments=lrc_assignments,
    )
