import numpy as np
import pytest

from y2karaoke.core.components.alignment.timing_models import (
    AudioFeatures,
    TranscriptionSegment,
    TranscriptionWord,
)
from y2karaoke.core.components.whisper import whisper_integration as wi
from y2karaoke.core.components.whisper import whisper_integration_align as wialign
from y2karaoke.core.components.whisper.whisper_integration_baseline import (
    _restore_implausibly_short_lines,
)
from y2karaoke.core.models import Line, Word


def test_restore_weak_evidence_large_start_shifts_restores_to_baseline():
    mapped = [
        Line(words=[Word(text="alpha", start_time=20.0, end_time=21.0)]),
        Line(words=[Word(text="beta", start_time=30.0, end_time=31.0)]),
    ]
    baseline = [
        Line(words=[Word(text="alpha", start_time=10.0, end_time=11.0)]),
        Line(words=[Word(text="beta", start_time=30.0, end_time=31.0)]),
    ]
    whisper_words = [
        TranscriptionWord(text="[VOCAL]", start=20.1, end=20.4, probability=0.9),
        TranscriptionWord(text="[VOCAL]", start=20.6, end=20.9, probability=0.9),
        TranscriptionWord(text="beta", start=30.0, end=30.5, probability=0.9),
    ]

    repaired, restored = wialign._restore_weak_evidence_large_start_shifts(
        mapped,
        baseline,
        whisper_words,
        min_shift_sec=1.1,
        min_support_words=2,
        support_window_sec=1.0,
    )

    assert restored == 1
    assert repaired[0].start_time == pytest.approx(10.0)
    assert repaired[1].start_time == pytest.approx(30.0)


def test_restore_weak_evidence_large_start_shifts_keeps_supported_shift():
    mapped = [
        Line(words=[Word(text="alpha", start_time=20.0, end_time=21.0)]),
    ]
    baseline = [
        Line(words=[Word(text="alpha", start_time=10.0, end_time=11.0)]),
    ]
    whisper_words = [
        TranscriptionWord(text="alpha", start=19.85, end=19.95, probability=0.8),
        TranscriptionWord(text="hear", start=19.8, end=20.0, probability=0.8),
        TranscriptionWord(text="words", start=20.2, end=20.4, probability=0.8),
        TranscriptionWord(text="[VOCAL]", start=20.6, end=20.9, probability=0.8),
    ]

    repaired, restored = wialign._restore_weak_evidence_large_start_shifts(
        mapped,
        baseline,
        whisper_words,
        min_shift_sec=1.1,
        min_support_words=3,
        support_window_sec=1.0,
    )

    assert restored == 0
    assert repaired[0].start_time == pytest.approx(20.0)


def test_restore_weak_evidence_large_start_shifts_restores_low_confidence_window():
    mapped = [
        Line(words=[Word(text="alpha", start_time=20.0, end_time=21.0)]),
        Line(words=[Word(text="beta", start_time=22.0, end_time=23.0)]),
    ]
    baseline = [
        Line(words=[Word(text="alpha", start_time=10.0, end_time=11.0)]),
        Line(words=[Word(text="beta", start_time=22.0, end_time=23.0)]),
    ]
    whisper_words = [
        TranscriptionWord(text="i", start=19.4, end=19.5, probability=0.2),
        TranscriptionWord(text="hear", start=19.8, end=20.0, probability=0.3),
        TranscriptionWord(text="words", start=20.2, end=20.4, probability=0.45),
        TranscriptionWord(text="beta", start=22.0, end=22.5, probability=0.9),
    ]

    repaired, restored = wialign._restore_weak_evidence_large_start_shifts(
        mapped,
        baseline,
        whisper_words,
        min_shift_sec=1.1,
        min_support_words=3,
        support_window_sec=1.0,
    )

    assert restored == 1
    assert repaired[0].start_time == pytest.approx(10.0)


def test_align_pipeline_restores_single_implausibly_short_line_before_rollback():
    lines = [
        Line(words=[Word(text="prev", start_time=10.0, end_time=11.0)]),
        Line(
            words=[
                Word(text="So", start_time=20.0, end_time=20.5),
                Word(text="I", start_time=20.5, end_time=21.0),
                Word(text="hit", start_time=21.0, end_time=21.5),
                Word(text="the", start_time=21.5, end_time=22.0),
            ]
        ),
    ]
    collapsed = [
        lines[0],
        Line(
            words=[
                Word(text="So", start_time=28.0, end_time=28.0),
                Word(text="I", start_time=28.0, end_time=28.0),
                Word(text="hit", start_time=28.0, end_time=28.0),
                Word(text="the", start_time=28.0, end_time=28.1),
            ]
        ),
    ]
    whisper_words = [
        TranscriptionWord(text="prev", start=10.0, end=10.5, probability=0.9),
        TranscriptionWord(text="So", start=27.8, end=27.9, probability=0.95),
        TranscriptionWord(text="I", start=28.0, end=28.05, probability=0.95),
        TranscriptionWord(text="hit", start=28.1, end=28.15, probability=0.95),
    ]
    segments = [
        TranscriptionSegment(
            start=10.0, end=10.5, text="prev", words=[whisper_words[0]]
        )
    ]

    mapped, corrections, _metrics = wialign.align_lrc_text_to_whisper_timings_impl(
        lines,
        vocals_path="vocals.wav",
        language="en",
        model_size="base",
        aggressive=False,
        temperature=0.0,
        min_similarity=0.15,
        audio_features=None,
        lenient_vocal_activity_threshold=0.3,
        lenient_activity_bonus=0.4,
        low_word_confidence_threshold=0.5,
        transcribe_vocals_fn=lambda *_a, **_k: (segments, whisper_words, "en", "base"),
        extract_audio_features_fn=lambda *_a, **_k: None,
        dedupe_whisper_segments_fn=lambda s: s,
        trim_whisper_transcription_by_lyrics_fn=lambda s, w, _t: (s, w, None),
        fill_vocal_activity_gaps_fn=lambda w, _a, _t, segments=None: (w, segments),
        dedupe_whisper_words_fn=lambda w: w,
        filter_low_confidence_whisper_words_fn=lambda w, _t: w,
        extract_lrc_words_all_fn=lambda in_lines: [
            {"text": wd.text, "line_idx": li, "word_idx": wi}
            for li, line in enumerate(in_lines)
            for wi, wd in enumerate(line.words)
        ],
        build_phoneme_tokens_from_lrc_words_fn=lambda _w, _l: [1, 2, 3],
        build_phoneme_tokens_from_whisper_words_fn=lambda _w, _l: [1, 2, 3],
        build_syllable_tokens_from_phonemes_fn=lambda _p: [1],
        build_segment_text_overlap_assignments_fn=lambda _lw, _aw, _s: {0: [0]},
        build_phoneme_dtw_path_fn=lambda *_a, **_k: [],
        build_word_assignments_from_phoneme_path_fn=lambda *_a, **_k: {},
        build_block_segmented_syllable_assignments_fn=lambda *_a, **_k: {},
        map_lrc_words_to_whisper_fn=lambda *_a, **_k: (collapsed, 1, 0.2, {0}),
        shift_repeated_lines_to_next_whisper_fn=lambda ml, _aw: ml,
        enforce_monotonic_line_starts_whisper_fn=lambda ml, _aw: ml,
        resolve_line_overlaps_fn=lambda ml: ml,
        extend_line_to_trailing_whisper_matches_fn=lambda ml, _aw: ml,
        pull_late_lines_to_matching_segments_fn=lambda ml, _s, _lang: ml,
        retime_short_interjection_lines_fn=lambda ml, _s: ml,
        snap_first_word_to_whisper_onset_fn=lambda ml, _aw, **_kw: ml,
        interpolate_unmatched_lines_fn=lambda ml, _set: ml,
        refine_unmatched_lines_with_onsets_fn=lambda ml, _set, _vp: ml,
        pull_lines_forward_for_continuous_vocals_fn=lambda ml, _af: (ml, 0),
        run_mapped_line_postpasses_fn=lambda **kwargs: (
            kwargs["mapped_lines"],
            kwargs["corrections"],
        ),
        constrain_line_starts_to_baseline_fn=lambda ml, _bl: ml,
        should_rollback_short_line_degradation_fn=lambda *_a, **_k: (False, 0, 0),
        restore_implausibly_short_lines_fn=_restore_implausibly_short_lines,
        clone_lines_for_fallback_fn=lambda in_lines: in_lines,
        min_segment_overlap_coverage=0.4,
        logger=wi.logger,
    )

    assert mapped[1].text == lines[1].text
    assert mapped[1].start_time == pytest.approx(lines[1].start_time)
    assert any("short compressed lines" in msg for msg in corrections)


def test_align_pipeline_uses_whisperx_on_low_dtw_coverage(monkeypatch):
    lines = [
        Line(words=[Word(text=f"w{i}", start_time=float(i), end_time=float(i) + 0.5)])
        for i in range(20)
    ]
    whisper_words = [
        TranscriptionWord(
            text=f"w{i}", start=10.0 + 0.2 * i, end=10.1 + 0.2 * i, probability=0.9
        )
        for i in range(90)
    ]
    segments = [
        TranscriptionSegment(
            start=10.0 + 4.0 * i,
            end=14.0 + 4.0 * i,
            text=f"segment{i}",
            words=whisper_words[i * 18 : (i + 1) * 18],
        )
        for i in range(5)
    ]
    forced_lines = [
        Line(words=[Word(text="w0", start_time=10.0, end_time=10.5)]),
    ] + lines[1:]

    monkeypatch.setattr(
        "y2karaoke.core.components.whisper.whisper_integration_align.align_lines_with_whisperx",
        lambda *_args, **_kwargs: (
            forced_lines,
            {"forced_line_coverage": 0.8, "forced_word_coverage": 0.8},
        ),
    )

    mapped, corrections, metrics = wialign.align_lrc_text_to_whisper_timings_impl(
        lines,
        vocals_path="vocals.wav",
        language="en",
        model_size="base",
        aggressive=False,
        temperature=0.0,
        min_similarity=0.15,
        audio_features=None,
        lenient_vocal_activity_threshold=0.3,
        lenient_activity_bonus=0.4,
        low_word_confidence_threshold=0.5,
        transcribe_vocals_fn=lambda *_a, **_k: (segments, whisper_words, "en", "base"),
        extract_audio_features_fn=lambda *_a, **_k: None,
        dedupe_whisper_segments_fn=lambda s: s,
        trim_whisper_transcription_by_lyrics_fn=lambda s, w, _t: (s, w, None),
        fill_vocal_activity_gaps_fn=lambda w, _a, _t, segments=None: (w, segments),
        dedupe_whisper_words_fn=lambda w: w,
        filter_low_confidence_whisper_words_fn=lambda w, _t: w,
        extract_lrc_words_all_fn=lambda in_lines: [
            {"text": wd.text, "line_idx": li, "word_idx": wi}
            for li, line in enumerate(in_lines)
            for wi, wd in enumerate(line.words)
        ],
        build_phoneme_tokens_from_lrc_words_fn=lambda _w, _l: [1, 2, 3],
        build_phoneme_tokens_from_whisper_words_fn=lambda _w, _l: [1, 2, 3],
        build_syllable_tokens_from_phonemes_fn=lambda _p: [1],
        build_segment_text_overlap_assignments_fn=lambda _lw, _aw, _s: {0: [0]},
        build_phoneme_dtw_path_fn=lambda *_a, **_k: [],
        build_word_assignments_from_phoneme_path_fn=lambda *_a, **_k: {},
        build_block_segmented_syllable_assignments_fn=lambda *_a, **_k: {},
        map_lrc_words_to_whisper_fn=lambda *_a, **_k: (lines, 1, 0.2, {0}),
        shift_repeated_lines_to_next_whisper_fn=lambda ml, _aw: ml,
        enforce_monotonic_line_starts_whisper_fn=lambda ml, _aw: ml,
        resolve_line_overlaps_fn=lambda ml: ml,
        extend_line_to_trailing_whisper_matches_fn=lambda ml, _aw: ml,
        pull_late_lines_to_matching_segments_fn=lambda ml, _s, _lang: ml,
        retime_short_interjection_lines_fn=lambda ml, _s: ml,
        snap_first_word_to_whisper_onset_fn=lambda ml, _aw: ml,
        interpolate_unmatched_lines_fn=lambda ml, _set: ml,
        refine_unmatched_lines_with_onsets_fn=lambda ml, _set, _vp: ml,
        pull_lines_forward_for_continuous_vocals_fn=lambda ml, _af: (ml, 0),
        run_mapped_line_postpasses_fn=lambda **kwargs: (
            kwargs["mapped_lines"],
            kwargs["corrections"],
        ),
        constrain_line_starts_to_baseline_fn=lambda ml, _bl: ml,
        should_rollback_short_line_degradation_fn=lambda *_a, **_k: (False, 0, 0),
        restore_implausibly_short_lines_fn=lambda _bl, al: (al, 0),
        clone_lines_for_fallback_fn=lambda in_lines: in_lines,
        min_segment_overlap_coverage=0.4,
        logger=wi.logger,
    )

    assert mapped == forced_lines
    assert metrics.get("whisperx_forced", 0.0) == pytest.approx(1.0)
    assert any("low DTW mapping coverage" in msg for msg in corrections)


def test_should_apply_baseline_constraint_skips_for_strong_global_shift():
    mapped = [
        Line(words=[Word(text="a", start_time=20.0, end_time=21.0)]),
        Line(words=[Word(text="b", start_time=30.0, end_time=31.0)]),
    ]
    baseline = [
        Line(words=[Word(text="a", start_time=10.0, end_time=11.0)]),
        Line(words=[Word(text="b", start_time=20.0, end_time=21.0)]),
    ]
    apply, median_shift = wialign._should_apply_baseline_constraint(
        mapped,
        baseline,
        matched_ratio=0.8,
        line_coverage=0.8,
        min_global_shift_sec=2.5,
    )
    assert apply is False
    assert median_shift == pytest.approx(10.0)


def test_align_pipeline_restores_pairwise_inversion_outlier_from_baseline():
    lines = [
        Line(words=[Word(text="a", start_time=10.0, end_time=11.0)]),
        Line(words=[Word(text="b", start_time=12.0, end_time=13.0)]),
    ]
    aligned = [
        Line(words=[Word(text="a", start_time=15.5, end_time=16.5)]),
        Line(words=[Word(text="b", start_time=12.1, end_time=13.1)]),
    ]
    whisper_words = [
        TranscriptionWord(text="a", start=15.5, end=16.5, probability=0.95),
        TranscriptionWord(text="b", start=12.1, end=13.1, probability=0.95),
    ]
    segments = [
        TranscriptionSegment(start=12.0, end=17.0, text="ab", words=whisper_words),
    ]

    mapped, corrections, _metrics = wialign.align_lrc_text_to_whisper_timings_impl(
        lines,
        vocals_path="vocals.wav",
        language="en",
        model_size="base",
        aggressive=False,
        temperature=0.0,
        min_similarity=0.15,
        audio_features=None,
        lenient_vocal_activity_threshold=0.3,
        lenient_activity_bonus=0.4,
        low_word_confidence_threshold=0.5,
        transcribe_vocals_fn=lambda *_a, **_k: (segments, whisper_words, "en", "base"),
        extract_audio_features_fn=lambda *_a, **_k: None,
        dedupe_whisper_segments_fn=lambda s: s,
        trim_whisper_transcription_by_lyrics_fn=lambda s, w, _t: (s, w, None),
        fill_vocal_activity_gaps_fn=lambda w, _a, _t, segments=None: (w, segments),
        dedupe_whisper_words_fn=lambda w: w,
        filter_low_confidence_whisper_words_fn=lambda w, _t: w,
        extract_lrc_words_all_fn=lambda in_lines: [
            {"text": wd.text, "line_idx": li, "word_idx": wi}
            for li, line in enumerate(in_lines)
            for wi, wd in enumerate(line.words)
        ],
        build_phoneme_tokens_from_lrc_words_fn=lambda _w, _l: [1, 2],
        build_phoneme_tokens_from_whisper_words_fn=lambda _w, _l: [1, 2],
        build_syllable_tokens_from_phonemes_fn=lambda _p: [1],
        build_segment_text_overlap_assignments_fn=lambda _lw, _aw, _s: {0: [0], 1: [1]},
        build_phoneme_dtw_path_fn=lambda *_a, **_k: [],
        build_word_assignments_from_phoneme_path_fn=lambda *_a, **_k: {},
        build_block_segmented_syllable_assignments_fn=lambda *_a, **_k: {},
        map_lrc_words_to_whisper_fn=lambda *_a, **_k: (aligned, 2, 0.9, {0, 1}),
        shift_repeated_lines_to_next_whisper_fn=lambda ml, _aw: ml,
        enforce_monotonic_line_starts_whisper_fn=lambda ml, _aw: ml,
        resolve_line_overlaps_fn=lambda ml: ml,
        extend_line_to_trailing_whisper_matches_fn=lambda ml, _aw: ml,
        pull_late_lines_to_matching_segments_fn=lambda ml, _s, _lang: ml,
        retime_short_interjection_lines_fn=lambda ml, _s: ml,
        snap_first_word_to_whisper_onset_fn=lambda ml, _aw, **_kw: ml,
        interpolate_unmatched_lines_fn=lambda ml, _set: ml,
        refine_unmatched_lines_with_onsets_fn=lambda ml, _set, _vp: ml,
        pull_lines_forward_for_continuous_vocals_fn=lambda ml, _af: (ml, 0),
        run_mapped_line_postpasses_fn=lambda **kwargs: (
            kwargs["mapped_lines"],
            kwargs["corrections"],
        ),
        constrain_line_starts_to_baseline_fn=lambda ml, _bl: ml,
        should_rollback_short_line_degradation_fn=lambda *_a, **_k: (False, 0, 0),
        restore_implausibly_short_lines_fn=lambda bl, al: (al, 0),
        clone_lines_for_fallback_fn=lambda in_lines: in_lines,
        min_segment_overlap_coverage=0.4,
        logger=wi.logger,
    )

    assert mapped[0].start_time == pytest.approx(10.0)
    assert mapped[1].start_time == pytest.approx(12.1)
    assert any("DTW-phonetic mapped" in msg for msg in corrections)


def test_align_pipeline_applies_carryover_shift_after_final_restores():
    lines = [
        Line(words=[Word(text="prev", start_time=150.59, end_time=156.41)]),
        Line(
            words=[
                Word(text="No,", start_time=156.57, end_time=156.9),
                Word(text="I", start_time=156.9, end_time=157.2),
                Word(text="can't", start_time=157.2, end_time=157.5),
                Word(text="sleep", start_time=157.5, end_time=157.8),
                Word(text="until", start_time=157.8, end_time=158.2),
                Word(text="I", start_time=158.2, end_time=158.5),
                Word(text="feel", start_time=158.5, end_time=158.8),
                Word(text="your", start_time=158.8, end_time=159.1),
                Word(text="touch", start_time=159.1, end_time=159.56),
            ]
        ),
        Line(words=[Word(text="next", start_time=160.79, end_time=161.2)]),
    ]
    whisper_words = [
        TranscriptionWord(text="prev", start=150.59, end=151.0, probability=0.9),
    ]
    segments = [
        TranscriptionSegment(
            start=150.0, end=160.0, text="segment", words=whisper_words
        ),
    ]
    audio_features = AudioFeatures(
        onset_times=np.array([156.69, 157.11, 157.29, 157.43], dtype=float),
        silence_regions=[],
        vocal_start=0.0,
        vocal_end=200.0,
        duration=200.0,
        energy_envelope=np.array([], dtype=float),
        energy_times=np.array([], dtype=float),
    )

    mapped, corrections, _metrics = wialign.align_lrc_text_to_whisper_timings_impl(
        lines,
        vocals_path="vocals.wav",
        language="en",
        model_size="base",
        aggressive=False,
        temperature=0.0,
        min_similarity=0.15,
        audio_features=audio_features,
        lenient_vocal_activity_threshold=0.3,
        lenient_activity_bonus=0.4,
        low_word_confidence_threshold=0.5,
        transcribe_vocals_fn=lambda *_a, **_k: (segments, whisper_words, "en", "base"),
        extract_audio_features_fn=lambda *_a, **_k: audio_features,
        dedupe_whisper_segments_fn=lambda s: s,
        trim_whisper_transcription_by_lyrics_fn=lambda s, w, _t: (s, w, None),
        fill_vocal_activity_gaps_fn=lambda w, _a, _t, segments=None: (w, segments),
        dedupe_whisper_words_fn=lambda w: w,
        filter_low_confidence_whisper_words_fn=lambda w, _t: w,
        extract_lrc_words_all_fn=lambda in_lines: [
            {"text": wd.text, "line_idx": li, "word_idx": wi}
            for li, line in enumerate(in_lines)
            for wi, wd in enumerate(line.words)
        ],
        build_phoneme_tokens_from_lrc_words_fn=lambda _w, _l: [1, 2, 3],
        build_phoneme_tokens_from_whisper_words_fn=lambda _w, _l: [1, 2, 3],
        build_syllable_tokens_from_phonemes_fn=lambda _p: [1],
        build_segment_text_overlap_assignments_fn=lambda _lw, _aw, _s: {0: [0]},
        build_phoneme_dtw_path_fn=lambda *_a, **_k: [],
        build_word_assignments_from_phoneme_path_fn=lambda *_a, **_k: {},
        build_block_segmented_syllable_assignments_fn=lambda *_a, **_k: {},
        map_lrc_words_to_whisper_fn=lambda *_a, **_k: (lines, 1, 0.2, {0}),
        shift_repeated_lines_to_next_whisper_fn=lambda ml, _aw: ml,
        enforce_monotonic_line_starts_whisper_fn=lambda ml, _aw: ml,
        resolve_line_overlaps_fn=lambda ml: ml,
        extend_line_to_trailing_whisper_matches_fn=lambda ml, _aw: ml,
        pull_late_lines_to_matching_segments_fn=lambda ml, _s, _lang: ml,
        retime_short_interjection_lines_fn=lambda ml, _s: ml,
        snap_first_word_to_whisper_onset_fn=lambda ml, _aw, **_kw: ml,
        interpolate_unmatched_lines_fn=lambda ml, _set: ml,
        refine_unmatched_lines_with_onsets_fn=lambda ml, _set, _vp: ml,
        pull_lines_forward_for_continuous_vocals_fn=lambda ml, _af: (ml, 0),
        run_mapped_line_postpasses_fn=lambda **kwargs: (
            kwargs["mapped_lines"],
            kwargs["corrections"],
        ),
        constrain_line_starts_to_baseline_fn=lambda ml, _bl: ml,
        should_rollback_short_line_degradation_fn=lambda *_a, **_k: (False, 0, 0),
        restore_implausibly_short_lines_fn=lambda _bl, al: (al, 0),
        clone_lines_for_fallback_fn=lambda in_lines: in_lines,
        min_segment_overlap_coverage=0.4,
        logger=wi.logger,
    )

    assert mapped[1].start_time == pytest.approx(157.11)
    assert any("prior-phrase carryover" in msg for msg in corrections)


def test_align_pipeline_applies_carryover_shift_with_small_positive_gap():
    lines = [
        Line(words=[Word(text="prev", start_time=29.83, end_time=32.41)]),
        Line(
            words=[
                Word(text="Maybe", start_time=32.62, end_time=33.0),
                Word(text="you", start_time=33.0, end_time=33.4),
                Word(text="can", start_time=33.4, end_time=33.8),
                Word(text="show", start_time=33.8, end_time=34.2),
                Word(text="me", start_time=34.2, end_time=34.6),
                Word(text="how", start_time=34.6, end_time=35.0),
                Word(text="to", start_time=35.0, end_time=35.4),
                Word(text="love,", start_time=35.4, end_time=35.9),
                Word(text="maybe", start_time=35.9, end_time=36.42),
            ]
        ),
        Line(words=[Word(text="next", start_time=38.24, end_time=40.58)]),
    ]
    whisper_words = [
        TranscriptionWord(text="prev", start=30.64, end=31.64, probability=0.9),
    ]
    segments = [
        TranscriptionSegment(start=29.0, end=41.0, text="segment", words=whisper_words),
    ]
    audio_features = AudioFeatures(
        onset_times=np.array([33.58, 34.02], dtype=float),
        silence_regions=[],
        vocal_start=0.0,
        vocal_end=200.0,
        duration=200.0,
        energy_envelope=np.array([], dtype=float),
        energy_times=np.array([], dtype=float),
    )

    mapped, corrections, _metrics = wialign.align_lrc_text_to_whisper_timings_impl(
        lines,
        vocals_path="vocals.wav",
        language="en",
        model_size="base",
        aggressive=False,
        temperature=0.0,
        min_similarity=0.15,
        audio_features=audio_features,
        lenient_vocal_activity_threshold=0.3,
        lenient_activity_bonus=0.4,
        low_word_confidence_threshold=0.5,
        transcribe_vocals_fn=lambda *_a, **_k: (segments, whisper_words, "en", "base"),
        extract_audio_features_fn=lambda *_a, **_k: audio_features,
        dedupe_whisper_segments_fn=lambda s: s,
        trim_whisper_transcription_by_lyrics_fn=lambda s, w, _t: (s, w, None),
        fill_vocal_activity_gaps_fn=lambda w, _a, _t, segments=None: (w, segments),
        dedupe_whisper_words_fn=lambda w: w,
        filter_low_confidence_whisper_words_fn=lambda w, _t: w,
        extract_lrc_words_all_fn=lambda in_lines: [
            {"text": wd.text, "line_idx": li, "word_idx": wi}
            for li, line in enumerate(in_lines)
            for wi, wd in enumerate(line.words)
        ],
        build_phoneme_tokens_from_lrc_words_fn=lambda _w, _l: [1, 2, 3],
        build_phoneme_tokens_from_whisper_words_fn=lambda _w, _l: [1, 2, 3],
        build_syllable_tokens_from_phonemes_fn=lambda _p: [1],
        build_segment_text_overlap_assignments_fn=lambda _lw, _aw, _s: {0: [0]},
        build_phoneme_dtw_path_fn=lambda *_a, **_k: [],
        build_word_assignments_from_phoneme_path_fn=lambda *_a, **_k: {},
        build_block_segmented_syllable_assignments_fn=lambda *_a, **_k: {},
        map_lrc_words_to_whisper_fn=lambda *_a, **_k: (lines, 1, 0.2, {0}),
        shift_repeated_lines_to_next_whisper_fn=lambda ml, _aw: ml,
        enforce_monotonic_line_starts_whisper_fn=lambda ml, _aw: ml,
        resolve_line_overlaps_fn=lambda ml: ml,
        extend_line_to_trailing_whisper_matches_fn=lambda ml, _aw: ml,
        pull_late_lines_to_matching_segments_fn=lambda ml, _s, _lang: ml,
        retime_short_interjection_lines_fn=lambda ml, _s: ml,
        snap_first_word_to_whisper_onset_fn=lambda ml, _aw, **_kw: ml,
        interpolate_unmatched_lines_fn=lambda ml, _set: ml,
        refine_unmatched_lines_with_onsets_fn=lambda ml, _set, _vp: ml,
        pull_lines_forward_for_continuous_vocals_fn=lambda ml, _af: (ml, 0),
        run_mapped_line_postpasses_fn=lambda **kwargs: (
            kwargs["mapped_lines"],
            kwargs["corrections"],
        ),
        constrain_line_starts_to_baseline_fn=lambda ml, _bl: ml,
        should_rollback_short_line_degradation_fn=lambda *_a, **_k: (False, 0, 0),
        restore_implausibly_short_lines_fn=lambda _bl, al: (al, 0),
        clone_lines_for_fallback_fn=lambda in_lines: in_lines,
        min_segment_overlap_coverage=0.4,
        logger=wi.logger,
    )

    assert mapped[1].start_time == pytest.approx(33.58)
    assert any("prior-phrase carryover" in msg for msg in corrections)


def test_align_pipeline_reanchors_unsupported_i_said_line_to_later_onset():
    lines = [
        Line(words=[Word(text="prev", start_time=145.0, end_time=149.0)]),
        Line(
            words=[
                Word(text="I", start_time=150.0, end_time=150.3),
                Word(text="said,", start_time=150.3, end_time=150.8),
                Word(text="ooh,", start_time=150.8, end_time=151.4),
                Word(text="I'm", start_time=151.4, end_time=152.2),
                Word(text="blinded", start_time=152.2, end_time=154.2),
                Word(text="by", start_time=154.2, end_time=154.9),
                Word(text="the", start_time=154.9, end_time=155.4),
                Word(text="lights", start_time=155.4, end_time=156.0),
            ]
        ),
        Line(
            words=[
                Word(text="No,", start_time=156.8, end_time=157.1),
                Word(text="I", start_time=157.1, end_time=157.4),
            ]
        ),
    ]
    whisper_words = [
        TranscriptionWord(text="[VOCAL]", start=150.2, end=150.3, probability=0.9),
    ]
    segments = [
        TranscriptionSegment(
            start=145.0, end=160.0, text="segment", words=whisper_words
        ),
    ]
    audio_features = AudioFeatures(
        onset_times=np.array([150.95, 151.6], dtype=float),
        silence_regions=[],
        vocal_start=0.0,
        vocal_end=200.0,
        duration=200.0,
        energy_envelope=np.array([], dtype=float),
        energy_times=np.array([], dtype=float),
    )

    mapped, corrections, _metrics = wialign.align_lrc_text_to_whisper_timings_impl(
        lines,
        vocals_path="vocals.wav",
        language="en",
        model_size="base",
        aggressive=False,
        temperature=0.0,
        min_similarity=0.15,
        audio_features=audio_features,
        lenient_vocal_activity_threshold=0.3,
        lenient_activity_bonus=0.4,
        low_word_confidence_threshold=0.5,
        transcribe_vocals_fn=lambda *_a, **_k: (segments, whisper_words, "en", "base"),
        extract_audio_features_fn=lambda *_a, **_k: audio_features,
        dedupe_whisper_segments_fn=lambda s: s,
        trim_whisper_transcription_by_lyrics_fn=lambda s, w, _t: (s, w, None),
        fill_vocal_activity_gaps_fn=lambda w, _a, _t, segments=None: (w, segments),
        dedupe_whisper_words_fn=lambda w: w,
        filter_low_confidence_whisper_words_fn=lambda w, _t: w,
        extract_lrc_words_all_fn=lambda in_lines: [
            {"text": wd.text, "line_idx": li, "word_idx": wi}
            for li, line in enumerate(in_lines)
            for wi, wd in enumerate(line.words)
        ],
        build_phoneme_tokens_from_lrc_words_fn=lambda _w, _l: [1, 2, 3],
        build_phoneme_tokens_from_whisper_words_fn=lambda _w, _l: [1, 2, 3],
        build_syllable_tokens_from_phonemes_fn=lambda _p: [1],
        build_segment_text_overlap_assignments_fn=lambda _lw, _aw, _s: {0: [0]},
        build_phoneme_dtw_path_fn=lambda *_a, **_k: [],
        build_word_assignments_from_phoneme_path_fn=lambda *_a, **_k: {},
        build_block_segmented_syllable_assignments_fn=lambda *_a, **_k: {},
        map_lrc_words_to_whisper_fn=lambda *_a, **_k: (lines, 1, 0.2, {0}),
        shift_repeated_lines_to_next_whisper_fn=lambda ml, _aw: ml,
        enforce_monotonic_line_starts_whisper_fn=lambda ml, _aw: ml,
        resolve_line_overlaps_fn=lambda ml: ml,
        extend_line_to_trailing_whisper_matches_fn=lambda ml, _aw: ml,
        pull_late_lines_to_matching_segments_fn=lambda ml, _s, _lang: ml,
        retime_short_interjection_lines_fn=lambda ml, _s: ml,
        snap_first_word_to_whisper_onset_fn=lambda ml, _aw, **_kw: ml,
        interpolate_unmatched_lines_fn=lambda ml, _set: ml,
        refine_unmatched_lines_with_onsets_fn=lambda ml, _set, _vp: ml,
        pull_lines_forward_for_continuous_vocals_fn=lambda ml, _af: (ml, 0),
        run_mapped_line_postpasses_fn=lambda **kwargs: (
            kwargs["mapped_lines"],
            kwargs["corrections"],
        ),
        constrain_line_starts_to_baseline_fn=lambda ml, _bl: ml,
        should_rollback_short_line_degradation_fn=lambda *_a, **_k: (False, 0, 0),
        restore_implausibly_short_lines_fn=lambda _bl, al: (al, 0),
        clone_lines_for_fallback_fn=lambda in_lines: in_lines,
        min_segment_overlap_coverage=0.4,
        logger=wi.logger,
    )

    assert mapped[1].start_time == pytest.approx(150.95)
    assert mapped[1].end_time == pytest.approx(156.0)
    assert any("unsupported 'I said' line" in msg for msg in corrections)


def test_align_pipeline_extends_unsupported_parenthetical_tail():
    lines = [
        Line(
            words=[
                Word(text="Will", start_time=145.46, end_time=145.88),
                Word(text="never", start_time=145.91, end_time=146.33),
                Word(text="let", start_time=146.35, end_time=146.78),
                Word(text="you", start_time=146.80, end_time=147.22),
                Word(text="go", start_time=147.24, end_time=147.67),
                Word(text="this", start_time=147.69, end_time=148.12),
                Word(text="time", start_time=148.14, end_time=148.56),
                Word(text="(ooh)", start_time=148.58, end_time=149.01),
            ]
        ),
        Line(
            words=[
                Word(text="I", start_time=151.05, end_time=151.2),
                Word(text="said,", start_time=151.2, end_time=151.6),
                Word(text="ooh,", start_time=151.6, end_time=152.0),
                Word(text="I'm", start_time=152.0, end_time=152.4),
                Word(text="blinded", start_time=152.4, end_time=154.2),
                Word(text="by", start_time=154.2, end_time=154.9),
                Word(text="the", start_time=154.9, end_time=155.3),
                Word(text="lights", start_time=155.3, end_time=156.41),
            ]
        ),
    ]
    whisper_words = [
        TranscriptionWord(text="[VOCAL]", start=145.6, end=145.7, probability=0.9),
    ]
    segments = [
        TranscriptionSegment(
            start=145.0, end=160.0, text="segment", words=whisper_words
        ),
    ]
    audio_features = AudioFeatures(
        onset_times=np.array([], dtype=float),
        silence_regions=[],
        vocal_start=0.0,
        vocal_end=200.0,
        duration=200.0,
        energy_envelope=np.array([], dtype=float),
        energy_times=np.array([], dtype=float),
    )

    mapped, corrections, _metrics = wialign.align_lrc_text_to_whisper_timings_impl(
        lines,
        vocals_path="vocals.wav",
        language="en",
        model_size="base",
        aggressive=False,
        temperature=0.0,
        min_similarity=0.15,
        audio_features=audio_features,
        lenient_vocal_activity_threshold=0.3,
        lenient_activity_bonus=0.4,
        low_word_confidence_threshold=0.5,
        transcribe_vocals_fn=lambda *_a, **_k: (segments, whisper_words, "en", "base"),
        extract_audio_features_fn=lambda *_a, **_k: audio_features,
        dedupe_whisper_segments_fn=lambda s: s,
        trim_whisper_transcription_by_lyrics_fn=lambda s, w, _t: (s, w, None),
        fill_vocal_activity_gaps_fn=lambda w, _a, _t, segments=None: (w, segments),
        dedupe_whisper_words_fn=lambda w: w,
        filter_low_confidence_whisper_words_fn=lambda w, _t: w,
        extract_lrc_words_all_fn=lambda in_lines: [
            {"text": wd.text, "line_idx": li, "word_idx": wi}
            for li, line in enumerate(in_lines)
            for wi, wd in enumerate(line.words)
        ],
        build_phoneme_tokens_from_lrc_words_fn=lambda _w, _l: [1, 2, 3],
        build_phoneme_tokens_from_whisper_words_fn=lambda _w, _l: [1, 2, 3],
        build_syllable_tokens_from_phonemes_fn=lambda _p: [1],
        build_segment_text_overlap_assignments_fn=lambda _lw, _aw, _s: {0: [0]},
        build_phoneme_dtw_path_fn=lambda *_a, **_k: [],
        build_word_assignments_from_phoneme_path_fn=lambda *_a, **_k: {},
        build_block_segmented_syllable_assignments_fn=lambda *_a, **_k: {},
        map_lrc_words_to_whisper_fn=lambda *_a, **_k: (lines, 1, 0.2, {0}),
        shift_repeated_lines_to_next_whisper_fn=lambda ml, _aw: ml,
        enforce_monotonic_line_starts_whisper_fn=lambda ml, _aw: ml,
        resolve_line_overlaps_fn=lambda ml: ml,
        extend_line_to_trailing_whisper_matches_fn=lambda ml, _aw: ml,
        pull_late_lines_to_matching_segments_fn=lambda ml, _s, _lang: ml,
        retime_short_interjection_lines_fn=lambda ml, _s: ml,
        snap_first_word_to_whisper_onset_fn=lambda ml, _aw, **_kw: ml,
        interpolate_unmatched_lines_fn=lambda ml, _set: ml,
        refine_unmatched_lines_with_onsets_fn=lambda ml, _set, _vp: ml,
        pull_lines_forward_for_continuous_vocals_fn=lambda ml, _af: (ml, 0),
        run_mapped_line_postpasses_fn=lambda **kwargs: (
            kwargs["mapped_lines"],
            kwargs["corrections"],
        ),
        constrain_line_starts_to_baseline_fn=lambda ml, _bl: ml,
        should_rollback_short_line_degradation_fn=lambda *_a, **_k: (False, 0, 0),
        restore_implausibly_short_lines_fn=lambda _bl, al: (al, 0),
        clone_lines_for_fallback_fn=lambda in_lines: in_lines,
        min_segment_overlap_coverage=0.4,
        logger=wi.logger,
    )

    assert mapped[0].words[-1].start_time == pytest.approx(148.58)
    assert mapped[0].words[-1].end_time == pytest.approx(150.8)
    assert any("parenthetical tail" in msg for msg in corrections)


def test_should_apply_baseline_constraint_keeps_for_weak_coverage():
    mapped = [
        Line(words=[Word(text="a", start_time=20.0, end_time=21.0)]),
    ]
    baseline = [
        Line(words=[Word(text="a", start_time=10.0, end_time=11.0)]),
    ]
    apply, median_shift = wialign._should_apply_baseline_constraint(
        mapped,
        baseline,
        matched_ratio=0.3,
        line_coverage=0.4,
        min_global_shift_sec=2.5,
    )
    assert apply is True
    assert median_shift == pytest.approx(10.0)


def test_should_apply_baseline_constraint_keeps_for_extreme_shift():
    mapped = [
        Line(words=[Word(text="a", start_time=40.0, end_time=41.0)]),
    ]
    baseline = [
        Line(words=[Word(text="a", start_time=10.0, end_time=11.0)]),
    ]
    apply, median_shift = wialign._should_apply_baseline_constraint(
        mapped,
        baseline,
        matched_ratio=0.9,
        line_coverage=0.95,
        min_global_shift_sec=2.5,
        max_global_shift_sec=12.0,
    )
    assert apply is True
    assert median_shift == pytest.approx(30.0)
