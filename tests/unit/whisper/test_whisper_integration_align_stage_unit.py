import numpy as np
import pytest

from y2karaoke.core.components.alignment.timing_models import (
    AudioFeatures,
    TranscriptionSegment,
    TranscriptionWord,
)
from y2karaoke.core.components.whisper import whisper_integration as wi
from y2karaoke.core.components.whisper import whisper_integration_align as wialign
from y2karaoke.core.models import Line, Word


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
    assert mapped[1].end_time == pytest.approx(156.7)
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


def test_align_pipeline_skips_i_said_reanchor_when_already_late_vs_baseline():
    lines = [
        Line(words=[Word(text="prev", start_time=123.67, end_time=129.07)]),
        Line(
            words=[
                Word(text="I", start_time=129.17, end_time=129.77),
                Word(text="said,", start_time=129.77, end_time=130.37),
                Word(text="ooh,", start_time=130.37, end_time=130.97),
                Word(text="I'm", start_time=130.97, end_time=131.57),
                Word(text="drowning", start_time=131.57, end_time=132.17),
                Word(text="in", start_time=132.17, end_time=132.77),
                Word(text="the", start_time=132.77, end_time=133.37),
                Word(text="night", start_time=133.37, end_time=133.97),
            ]
        ),
        Line(
            words=[
                Word(text="Oh,", start_time=135.12, end_time=135.4),
                Word(text="when", start_time=135.4, end_time=135.8),
            ]
        ),
    ]
    baseline = [
        Line(words=[Word(text="prev", start_time=123.67, end_time=129.07)]),
        Line(
            words=[
                Word(text="I", start_time=128.09, end_time=128.82),
                Word(text="said,", start_time=128.82, end_time=129.54),
                Word(text="ooh,", start_time=129.54, end_time=130.27),
                Word(text="I'm", start_time=130.27, end_time=131.0),
                Word(text="drowning", start_time=131.0, end_time=131.73),
                Word(text="in", start_time=131.73, end_time=132.45),
                Word(text="the", start_time=132.45, end_time=133.18),
                Word(text="night", start_time=133.18, end_time=133.91),
            ]
        ),
        Line(
            words=[
                Word(text="Oh,", start_time=133.99, end_time=134.4),
                Word(text="when", start_time=134.4, end_time=134.8),
            ]
        ),
    ]
    whisper_words = [
        TranscriptionWord(text="[VOCAL]", start=120.0, end=120.1, probability=0.9)
    ]
    audio_features = AudioFeatures(
        onset_times=np.array([129.85, 130.01, 130.77], dtype=float),
        silence_regions=[],
        vocal_start=0.0,
        vocal_end=200.0,
        duration=200.0,
        energy_envelope=np.array([], dtype=float),
        energy_times=np.array([], dtype=float),
    )

    mapped, applied = wialign._reanchor_unsupported_i_said_lines_to_later_onset(
        lines,
        baseline,
        whisper_words,
        audio_features,
    )

    assert applied == 0
    assert mapped[1].start_time == pytest.approx(129.17, abs=0.01)


def test_align_pipeline_extends_unsupported_weak_opening_line():
    lines = [
        Line(words=[Word(text="prev", start_time=73.0, end_time=77.1)]),
        Line(
            words=[
                Word(text="Oh,", start_time=77.9, end_time=78.2),
                Word(text="when", start_time=78.2, end_time=78.5),
                Word(text="I'm", start_time=78.5, end_time=78.8),
                Word(text="like", start_time=78.8, end_time=79.1),
                Word(text="this,", start_time=79.1, end_time=79.4),
                Word(text="you're", start_time=79.4, end_time=79.7),
                Word(text="the", start_time=79.7, end_time=80.0),
                Word(text="one", start_time=80.0, end_time=80.3),
                Word(text="I", start_time=80.3, end_time=80.6),
                Word(text="trust", start_time=80.6, end_time=81.25),
            ]
        ),
        Line(words=[Word(text="next", start_time=82.23, end_time=83.0)]),
    ]
    whisper_words = [
        TranscriptionWord(text="[VOCAL]", start=70.0, end=70.1, probability=0.9),
    ]
    segments = [
        TranscriptionSegment(start=70.0, end=84.0, text="segment", words=whisper_words),
    ]
    audio_features = AudioFeatures(
        onset_times=np.array([78.2, 79.0], dtype=float),
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

    assert mapped[1].end_time == pytest.approx(81.93, abs=0.01)
    assert any("weak-opening line" in msg for msg in corrections)


def test_restore_consistently_late_runs_from_baseline_restores_dense_run():
    baseline = [
        Line(
            words=[
                Word(text=t, start_time=54.10 + i * 0.5, end_time=54.50 + i * 0.5)
                for i, t in enumerate(["a", "b", "c", "d"])
            ]
        ),
        Line(
            words=[
                Word(text=t, start_time=56.15 + i * 0.4, end_time=56.45 + i * 0.4)
                for i, t in enumerate(["a", "b", "c", "d"])
            ]
        ),
        Line(
            words=[
                Word(text=t, start_time=57.80 + i * 0.45, end_time=58.10 + i * 0.45)
                for i, t in enumerate(["a", "b", "c", "d"])
            ]
        ),
        Line(
            words=[
                Word(text=t, start_time=59.80 + i * 0.4, end_time=60.10 + i * 0.4)
                for i, t in enumerate(["a", "b", "c", "d"])
            ]
        ),
    ]
    mapped = [
        Line(
            words=[
                Word(text=t, start_time=55.67 + i * 0.45, end_time=56.05 + i * 0.45)
                for i, t in enumerate(["a", "b", "c", "d"])
            ]
        ),
        Line(
            words=[
                Word(text=t, start_time=57.51 + i * 0.55, end_time=57.90 + i * 0.55)
                for i, t in enumerate(["a", "b", "c", "d"])
            ]
        ),
        Line(
            words=[
                Word(text=t, start_time=59.87 + i * 0.43, end_time=60.24 + i * 0.43)
                for i, t in enumerate(["a", "b", "c", "d"])
            ]
        ),
        Line(
            words=[
                Word(text=t, start_time=61.71 + i * 0.42, end_time=62.08 + i * 0.42)
                for i, t in enumerate(["a", "b", "c", "d"])
            ]
        ),
    ]

    repaired, restored = wialign._restore_consistently_late_runs_from_baseline(
        mapped,
        baseline,
    )

    assert restored == 4
    assert repaired[0].start_time == pytest.approx(54.10)
    assert repaired[3].start_time == pytest.approx(59.80)


def test_restore_consistently_late_runs_from_baseline_skips_inconsistent_run():
    baseline = [
        Line(
            words=[
                Word(text=t, start_time=10.0 + i * 0.2, end_time=10.15 + i * 0.2)
                for i, t in enumerate(["a", "b", "c", "d"])
            ]
        ),
        Line(
            words=[
                Word(text=t, start_time=11.0 + i * 0.2, end_time=11.15 + i * 0.2)
                for i, t in enumerate(["a", "b", "c", "d"])
            ]
        ),
        Line(
            words=[
                Word(text=t, start_time=12.0 + i * 0.2, end_time=12.15 + i * 0.2)
                for i, t in enumerate(["a", "b", "c", "d"])
            ]
        ),
        Line(
            words=[
                Word(text=t, start_time=13.0 + i * 0.2, end_time=13.15 + i * 0.2)
                for i, t in enumerate(["a", "b", "c", "d"])
            ]
        ),
    ]
    mapped = [
        Line(
            words=[
                Word(text=t, start_time=10.9 + i * 0.2, end_time=11.05 + i * 0.2)
                for i, t in enumerate(["a", "b", "c", "d"])
            ]
        ),
        Line(
            words=[
                Word(text=t, start_time=12.4 + i * 0.2, end_time=12.55 + i * 0.2)
                for i, t in enumerate(["a", "b", "c", "d"])
            ]
        ),
        Line(
            words=[
                Word(text=t, start_time=13.1 + i * 0.2, end_time=13.25 + i * 0.2)
                for i, t in enumerate(["a", "b", "c", "d"])
            ]
        ),
        Line(
            words=[
                Word(text=t, start_time=13.95 + i * 0.2, end_time=14.10 + i * 0.2)
                for i, t in enumerate(["a", "b", "c", "d"])
            ]
        ),
    ]

    repaired, restored = wialign._restore_consistently_late_runs_from_baseline(
        mapped,
        baseline,
    )

    assert restored == 0
    assert repaired[1].start_time == pytest.approx(12.4)
