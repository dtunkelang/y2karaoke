import numpy as np
import pytest

from y2karaoke.core.components.alignment.timing_models import (
    AudioFeatures,
    TranscriptionSegment,
    TranscriptionWord,
)
from y2karaoke.core.components.whisper import whisper_integration as wi
from y2karaoke.core.components.whisper import whisper_integration_align as wialign
from y2karaoke.core.components.whisper.whisper_runtime_config import (
    WhisperRuntimeConfig,
)
from y2karaoke.core.models import Line, Word


def test_align_pipeline_reanchors_unsupported_interjection_line_to_onsets():
    lines = [
        Line(words=[Word(text="prev", start_time=157.22, end_time=160.21)]),
        Line(
            words=[
                Word(text="(Hey,", start_time=160.79, end_time=161.16),
                Word(text="hey,", start_time=161.18, end_time=161.56),
                Word(text="hey)", start_time=161.58, end_time=161.95),
            ]
        ),
        Line(words=[Word(text="next", start_time=171.94, end_time=173.1)]),
    ]
    whisper_words = [
        TranscriptionWord(text="[VOCAL]", start=150.0, end=150.1, probability=0.9),
    ]
    segments = [
        TranscriptionSegment(
            start=150.0, end=180.0, text="segment", words=whisper_words
        ),
    ]
    audio_features = AudioFeatures(
        onset_times=np.array(
            [161.56, 161.91, 162.31, 162.82, 162.98, 163.17], dtype=float
        ),
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
        runtime_config=WhisperRuntimeConfig(restored_run_onset_shift=False),
        logger=wi.logger,
    )

    assert mapped[1].start_time == pytest.approx(161.56, abs=0.01)
    assert mapped[1].end_time == pytest.approx(163.12, abs=0.02)
    assert any("unsupported interjection line" in msg for msg in corrections)


def test_extend_unsupported_i_said_tails_helper():
    lines = [
        Line(
            words=[
                Word(text="I", start_time=117.73, end_time=118.12),
                Word(text="said,", start_time=118.12, end_time=118.50),
                Word(text="ooh,", start_time=118.50, end_time=118.88),
                Word(text="I'm", start_time=118.88, end_time=119.26),
                Word(text="blinded", start_time=119.26, end_time=120.30),
                Word(text="by", start_time=120.30, end_time=120.75),
                Word(text="the", start_time=120.75, end_time=121.20),
                Word(text="lights", start_time=121.20, end_time=121.97),
            ]
        ),
        Line(
            words=[
                Word(text="No,", start_time=123.67, end_time=124.10),
                Word(text="I", start_time=124.10, end_time=124.45),
                Word(text="can't", start_time=124.45, end_time=124.90),
            ]
        ),
    ]

    mapped, applied = wialign._extend_unsupported_i_said_tails(lines, [])

    assert applied == 1
    assert mapped[0].end_time == pytest.approx(123.45)


def test_extend_sparse_i_said_tails_helper_with_single_nearby_word():
    lines = [
        Line(
            words=[
                Word(text="I", start_time=60.78, end_time=61.468),
                Word(text="said,", start_time=61.504, end_time=62.191),
                Word(text="ooh,", start_time=62.227, end_time=62.915),
                Word(text="I'm", start_time=62.951, end_time=63.639),
                Word(text="blinded", start_time=63.675, end_time=64.363),
                Word(text="by", start_time=64.399, end_time=65.086),
                Word(text="the", start_time=65.123, end_time=65.81),
                Word(text="lights", start_time=65.846, end_time=66.52),
            ]
        ),
        Line(
            words=[
                Word(text="No,", start_time=67.50, end_time=68.016),
                Word(text="I", start_time=68.043, end_time=68.558),
                Word(text="can't", start_time=68.585, end_time=69.10),
            ]
        ),
    ]
    whisper_words = [
        TranscriptionWord(text="been", start=61.42, end=61.66, probability=0.938),
    ]

    mapped, applied = wialign._extend_unsupported_i_said_tails(lines, whisper_words)

    assert applied == 1
    assert mapped[0].end_time == pytest.approx(67.28)


def test_restore_zero_support_parenthetical_late_start_expansions_helper():
    baseline = [
        Line(words=[Word(text="prev", start_time=100.39, end_time=106.43)]),
        Line(
            words=[
                Word(text="The", start_time=107.08, end_time=107.46),
                Word(text="city's", start_time=107.46, end_time=107.83),
                Word(text="cold", start_time=107.83, end_time=108.21),
                Word(text="and", start_time=108.21, end_time=108.58),
                Word(text="empty", start_time=108.58, end_time=108.96),
                Word(text="(oh)", start_time=108.96, end_time=109.336),
            ]
        ),
        Line(
            words=[
                Word(text="No,", start_time=122.84, end_time=123.45),
                Word(text="I", start_time=123.45, end_time=124.06),
                Word(text="can't", start_time=124.06, end_time=124.67),
                Word(text="sleep", start_time=124.67, end_time=125.27),
                Word(text="until", start_time=125.27, end_time=125.88),
                Word(text="I", start_time=125.88, end_time=126.49),
                Word(text="feel", start_time=126.49, end_time=127.10),
                Word(text="your", start_time=127.10, end_time=127.70),
                Word(text="touch", start_time=127.70, end_time=129.07),
            ]
        ),
    ]
    mapped = [
        baseline[0],
        Line(
            words=[
                Word(text="The", start_time=107.92, end_time=108.473),
                Word(text="city's", start_time=108.473, end_time=109.027),
                Word(text="cold", start_time=109.027, end_time=109.58),
                Word(text="and", start_time=109.58, end_time=110.133),
                Word(text="empty", start_time=110.133, end_time=110.687),
                Word(text="(oh)", start_time=110.687, end_time=111.24),
            ]
        ),
        Line(
            words=[
                Word(text="No,", start_time=123.67, end_time=124.27),
                Word(text="I", start_time=124.27, end_time=124.87),
                Word(text="can't", start_time=124.87, end_time=125.47),
                Word(text="sleep", start_time=125.47, end_time=126.07),
                Word(text="until", start_time=126.07, end_time=126.67),
                Word(text="I", start_time=126.67, end_time=127.27),
                Word(text="feel", start_time=127.27, end_time=127.87),
                Word(text="your", start_time=127.87, end_time=128.47),
                Word(text="touch", start_time=128.47, end_time=129.07),
            ]
        ),
    ]

    restored, applied = (
        wialign._restore_zero_support_parenthetical_late_start_expansions(
            mapped,
            baseline,
            [],
        )
    )

    assert applied == 1
    assert restored[1].start_time == pytest.approx(107.08, abs=0.01)
    assert restored[1].end_time == pytest.approx(111.24, abs=0.01)
    assert restored[2].start_time == pytest.approx(123.67, abs=0.01)


def test_align_pipeline_reanchors_sparse_interjection_cluster_with_modest_shift():
    lines = [
        Line(words=[Word(text="prev", start_time=77.9, end_time=81.93)]),
        Line(
            words=[
                Word(text="(Hey,", start_time=82.23, end_time=82.62),
                Word(text="hey,", start_time=82.62, end_time=83.0),
                Word(text="hey)", start_time=83.0, end_time=83.39),
            ]
        ),
        Line(words=[Word(text="next", start_time=94.24, end_time=95.95)]),
    ]
    whisper_words = [
        TranscriptionWord(text="[VOCAL]", start=75.0, end=75.1, probability=0.9),
    ]
    segments = [
        TranscriptionSegment(
            start=75.0, end=100.0, text="segment", words=whisper_words
        ),
    ]
    audio_features = AudioFeatures(
        onset_times=np.array([82.988, 83.685], dtype=float),
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

    assert mapped[1].start_time == pytest.approx(82.988, abs=0.01)
    assert mapped[1].end_time == pytest.approx(84.438, abs=0.03)
    assert any("unsupported interjection line" in msg for msg in corrections)


def test_align_pipeline_reanchors_very_sparse_hey_cluster():
    lines = [
        Line(words=[Word(text="prev", start_time=157.22, end_time=160.21)]),
        Line(
            words=[
                Word(text="(Hey,", start_time=160.79, end_time=161.16),
                Word(text="hey,", start_time=161.18, end_time=161.56),
                Word(text="hey)", start_time=161.58, end_time=161.95),
            ]
        ),
        Line(words=[Word(text="next", start_time=171.94, end_time=173.10)]),
    ]
    whisper_words = [
        TranscriptionWord(text="[VOCAL]", start=150.0, end=150.1, probability=0.9),
    ]
    segments = [
        TranscriptionSegment(
            start=150.0, end=180.0, text="segment", words=whisper_words
        ),
    ]
    audio_features = AudioFeatures(
        onset_times=np.array([162.656, 162.981], dtype=float),
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

    assert mapped[1].start_time == pytest.approx(160.79, abs=0.01)
    assert mapped[1].end_time == pytest.approx(163.488, abs=0.03)
    assert any("unsupported interjection line" in msg for msg in corrections)


def test_align_pipeline_keeps_standalone_sparse_hey_cluster_with_large_gap_before():
    lines = [
        Line(words=[Word(text="prev", start_time=160.79, end_time=162.34)]),
        Line(
            words=[
                Word(text="(Hey,", start_time=171.94, end_time=172.32),
                Word(text="hey,", start_time=172.32, end_time=172.71),
                Word(text="hey)", start_time=172.71, end_time=173.10),
            ]
        ),
        Line(words=[Word(text="next", start_time=184.0, end_time=185.73)]),
    ]
    whisper_words = [
        TranscriptionWord(text="[VOCAL]", start=150.0, end=150.1, probability=0.9),
    ]
    segments = [
        TranscriptionSegment(
            start=150.0, end=190.0, text="segment", words=whisper_words
        ),
    ]
    audio_features = AudioFeatures(
        onset_times=np.array([172.756, 173.728], dtype=float),
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

    assert mapped[1].start_time == pytest.approx(171.94, abs=0.01)
    assert mapped[1].end_time == pytest.approx(173.10, abs=0.01)
    assert not any("unsupported interjection line" in msg for msg in corrections)


def test_align_pipeline_extends_unsupported_line_before_weak_opening():
    lines = [
        Line(words=[Word(text="prev", start_time=123.67, end_time=129.71)]),
        Line(
            words=[
                Word(text="I", start_time=130.01, end_time=130.5),
                Word(text="said,", start_time=130.5, end_time=131.0),
                Word(text="ooh,", start_time=131.0, end_time=131.5),
                Word(text="I'm", start_time=131.5, end_time=132.0),
                Word(text="drowning", start_time=132.0, end_time=132.7),
                Word(text="in", start_time=132.7, end_time=133.0),
                Word(text="the", start_time=133.0, end_time=133.3),
                Word(text="night", start_time=133.3, end_time=133.97),
            ]
        ),
        Line(
            words=[
                Word(text="Oh,", start_time=135.12, end_time=135.4),
                Word(text="when", start_time=135.4, end_time=135.8),
            ]
        ),
    ]
    whisper_words = [
        TranscriptionWord(text="[VOCAL]", start=120.0, end=120.1, probability=0.9),
    ]
    segments = [
        TranscriptionSegment(
            start=120.0, end=140.0, text="segment", words=whisper_words
        ),
    ]
    audio_features = AudioFeatures(
        onset_times=np.array([129.85, 131.56, 132.49], dtype=float),
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

    assert mapped[1].end_time == pytest.approx(135.02, abs=0.01)
    assert any("before weak openings" in msg for msg in corrections)


def test_align_pipeline_does_not_extend_no_line_before_i_said():
    lines = [
        Line(
            words=[
                Word(text="No,", start_time=123.67, end_time=124.15),
                Word(text="I", start_time=124.15, end_time=124.55),
                Word(text="can't", start_time=124.55, end_time=125.05),
                Word(text="sleep", start_time=125.05, end_time=125.55),
                Word(text="until", start_time=125.55, end_time=126.05),
                Word(text="I", start_time=126.05, end_time=126.45),
                Word(text="feel", start_time=126.45, end_time=126.95),
                Word(text="your", start_time=126.95, end_time=127.45),
                Word(text="touch", start_time=127.45, end_time=128.4),
            ]
        ),
        Line(
            words=[
                Word(text="I", start_time=130.01, end_time=130.5),
                Word(text="said,", start_time=130.5, end_time=131.0),
            ]
        ),
    ]

    mapped, applied = wialign._extend_unsupported_weak_opening_lines(lines, [])

    assert applied == 0
    assert mapped[0].end_time == pytest.approx(128.4)


def test_align_pipeline_extends_misaligned_line_before_i_said():
    lines = [
        Line(
            words=[
                Word(text="I", start_time=112.67, end_time=113.0),
                Word(text="can't", start_time=113.0, end_time=113.4),
                Word(text="see", start_time=113.4, end_time=113.8),
                Word(text="clearly", start_time=113.8, end_time=114.3),
                Word(text="when", start_time=114.3, end_time=114.8),
                Word(text="you're", start_time=114.8, end_time=115.2),
                Word(text="gone", start_time=115.2, end_time=115.67),
            ]
        ),
        Line(
            words=[
                Word(text="I", start_time=117.17, end_time=117.5),
                Word(text="said,", start_time=117.5, end_time=118.0),
            ]
        ),
    ]
    whisper_words = [
        TranscriptionWord(text="maybe", start=111.81, end=112.01, probability=0.99),
        TranscriptionWord(text="now", start=112.01, end=112.17, probability=0.99),
        TranscriptionWord(text="hold", start=112.17, end=112.17, probability=0.99),
        TranscriptionWord(text="on", start=112.17, end=112.17, probability=0.99),
        TranscriptionWord(text="tight", start=112.17, end=112.17, probability=0.8),
        TranscriptionWord(text="for", start=112.17, end=112.17, probability=0.99),
        TranscriptionWord(text="one", start=112.17, end=112.17, probability=0.99),
        TranscriptionWord(text="more", start=112.17, end=112.17, probability=0.99),
        TranscriptionWord(text="turn", start=112.17, end=112.17, probability=0.99),
        TranscriptionWord(text="before", start=112.17, end=112.17, probability=0.99),
        TranscriptionWord(text="I", start=112.17, end=112.17, probability=0.99),
        TranscriptionWord(text="guess", start=112.17, end=112.17, probability=0.99),
    ]
    segments = [
        TranscriptionSegment(
            start=111.0, end=119.0, text="segment", words=whisper_words
        ),
    ]
    audio_features = AudioFeatures(
        onset_times=np.array([112.872, 113.22, 113.546], dtype=float),
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
        runtime_config=WhisperRuntimeConfig(restored_run_onset_shift=False),
        logger=wi.logger,
    )

    assert mapped[0].end_time == pytest.approx(116.97, abs=0.01)
    assert any("before unsupported 'I said' lines" in msg for msg in corrections)


def test_align_pipeline_extends_sparse_start_line_with_moderate_overlap_before_i_said():
    lines = [
        Line(
            words=[
                Word(text="I", start_time=56.29, end_time=56.66),
                Word(text="can't", start_time=56.68, end_time=57.05),
                Word(text="see", start_time=57.07, end_time=57.441),
                Word(text="clearly", start_time=57.46, end_time=57.831),
                Word(text="when", start_time=57.85, end_time=58.221),
                Word(text="you're", start_time=58.24, end_time=58.611),
                Word(text="gone", start_time=58.63, end_time=59.001),
            ]
        ),
        Line(
            words=[
                Word(text="I", start_time=60.78, end_time=61.468),
                Word(text="said,", start_time=61.504, end_time=62.191),
                Word(text="ooh,", start_time=62.227, end_time=62.915),
            ]
        ),
    ]
    whisper_words = [
        TranscriptionWord(text="bless", start=55.46, end=55.82, probability=0.231),
        TranscriptionWord(text="me,", start=55.82, end=56.3, probability=0.999),
        TranscriptionWord(text="I", start=56.44, end=56.96, probability=0.879),
        TranscriptionWord(text="guess", start=56.96, end=57.34, probability=0.812),
        TranscriptionWord(text="you", start=57.34, end=57.58, probability=0.93),
        TranscriptionWord(text="clearly", start=57.58, end=58.06, probability=0.853),
        TranscriptionWord(text="didn't", start=58.06, end=58.76, probability=0.875),
        TranscriptionWord(text="get", start=58.76, end=59.04, probability=0.613),
        TranscriptionWord(text="it,", start=59.04, end=59.2, probability=0.489),
        TranscriptionWord(text="God,", start=59.42, end=59.62, probability=0.251),
        TranscriptionWord(text="I've", start=59.62, end=61.42, probability=0.512),
    ]
    segments = [
        TranscriptionSegment(start=55.0, end=67.0, text="segment", words=whisper_words),
    ]
    audio_features = AudioFeatures(
        onset_times=np.array([56.44, 57.58], dtype=float),
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

    assert mapped[0].end_time == pytest.approx(60.58, abs=0.01)
    assert any("before unsupported 'I said' lines" in msg for msg in corrections)
