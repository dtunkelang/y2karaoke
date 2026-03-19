import numpy as np
import pytest
import y2karaoke.core.components.alignment.timing_evaluator as te
from y2karaoke.core.models import Line, Word
import y2karaoke.core.components.whisper.whisper_integration as wi
import y2karaoke.core.components.whisper.whisper_integration_pipeline as wip
from y2karaoke.core.components.whisper.whisper_integration_pipeline import (
    align_lrc_text_to_whisper_timings_impl,
)
from y2karaoke.core.components.whisper.whisper_integration_hooks import (
    AlignmentPassHooks,
)
from y2karaoke.core.components.whisper.whisper_runtime_config import (
    WhisperRuntimeConfig,
)
from y2karaoke.core.components.whisper import whisper_mapping as wm
from y2karaoke.core.components.alignment.timing_models import (
    AudioFeatures,
    TranscriptionWord,
    TranscriptionSegment,
)


def test_align_lrc_text_pipeline_pulls_forward_for_continuous_vocals():
    lines = [
        Line(words=[Word(text="Daddy", start_time=95.0, end_time=97.0)]),
        Line(words=[Word(text="I'm", start_time=120.0, end_time=121.0)]),
    ]
    whisper_words = [
        TranscriptionWord(text="daddy", start=95.1, end=96.0, probability=0.9),
        TranscriptionWord(text="im", start=120.2, end=120.8, probability=0.9),
    ]
    segments = [
        TranscriptionSegment(
            start=95.0, end=97.0, text="daddy", words=[whisper_words[0]]
        ),
        TranscriptionSegment(
            start=120.0, end=121.0, text="im", words=[whisper_words[1]]
        ),
    ]
    audio_features = AudioFeatures(
        onset_times=np.array([96.0, 98.0], dtype=float),
        silence_regions=[],
        vocal_start=0.0,
        vocal_end=130.0,
        duration=130.0,
        energy_envelope=np.array([], dtype=float),
        energy_times=np.array([], dtype=float),
    )

    def fake_pull_forward(mapped_lines, _audio_features):
        shifted = [
            mapped_lines[0],
            Line(
                words=[Word(text="I'm", start_time=98.0, end_time=99.0)],
                singer=mapped_lines[1].singer,
            ),
        ]
        return shifted, 1

    mapped, corrections, metrics = align_lrc_text_to_whisper_timings_impl(
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
        build_word_assignments_from_phoneme_path_fn=lambda *_a, **_k: {0: [0], 1: [1]},
        build_block_segmented_syllable_assignments_fn=lambda *_a, **_k: {
            0: [0],
            1: [1],
        },
        map_lrc_words_to_whisper_fn=lambda *_a, **_k: (lines, 2, 2.0, {0, 1}),
        shift_repeated_lines_to_next_whisper_fn=lambda ml, _aw: ml,
        enforce_monotonic_line_starts_whisper_fn=lambda ml, _aw: ml,
        resolve_line_overlaps_fn=lambda ml: ml,
        extend_line_to_trailing_whisper_matches_fn=lambda ml, _aw: ml,
        pull_late_lines_to_matching_segments_fn=lambda ml, _s, _lang: ml,
        retime_short_interjection_lines_fn=lambda ml, _s: ml,
        snap_first_word_to_whisper_onset_fn=lambda ml, _aw: ml,
        interpolate_unmatched_lines_fn=lambda ml, _set: ml,
        refine_unmatched_lines_with_onsets_fn=lambda ml, _set, _vp: ml,
        pull_lines_forward_for_continuous_vocals_fn=fake_pull_forward,
        normalize_line_word_timings_fn=lambda lines: lines,
        enforce_monotonic_line_starts_fn=lambda lines: lines,
        enforce_non_overlapping_lines_fn=lambda lines: lines,
        logger=wi.logger,
    )

    assert mapped[1].start_time == 98.0
    assert any("continuous vocals" in msg for msg in corrections)
    assert "mapping_stage_sec" in metrics
    assert "mapped_postpasses_sec" in metrics
    assert "alignment_total_sec" in metrics


def test_build_alignment_pass_kwargs_uses_hook_bundle():
    lines = [Line(words=[Word(text="a", start_time=1.0, end_time=1.2)])]
    kwargs = wip._build_alignment_pass_kwargs(
        lines=lines,
        vocals_path="vocals.wav",
        language="en",
        model_size="base",
        temperature=0.0,
        min_similarity=0.15,
        audio_features=None,
        lenient_vocal_activity_threshold=0.3,
        lenient_activity_bonus=0.4,
        low_word_confidence_threshold=0.5,
        runtime_config=WhisperRuntimeConfig(profile="safe"),
        transcribe_vocals_fn=lambda *_a, **_k: ([], [], "en", "base"),
        extract_audio_features_fn=lambda *_a, **_k: None,
        dedupe_whisper_segments_fn=lambda s: s,
        trim_whisper_transcription_by_lyrics_fn=lambda s, w, _t: (s, w, None),
        fill_vocal_activity_gaps_fn=lambda w, _a, _t, segments=None: (w, segments),
        dedupe_whisper_words_fn=lambda w: w,
        extract_lrc_words_all_fn=lambda _in_lines: [],
        build_phoneme_tokens_from_lrc_words_fn=lambda *_a, **_k: [],
        build_phoneme_tokens_from_whisper_words_fn=lambda *_a, **_k: [],
        build_syllable_tokens_from_phonemes_fn=lambda *_a, **_k: [],
        build_segment_text_overlap_assignments_fn=lambda *_a, **_k: {},
        build_phoneme_dtw_path_fn=lambda *_a, **_k: [],
        build_word_assignments_from_phoneme_path_fn=lambda *_a, **_k: {},
        build_block_segmented_syllable_assignments_fn=lambda *_a, **_k: {},
        map_lrc_words_to_whisper_fn=lambda *_a, **_k: (lines, 0, 0.0, set()),
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
        normalize_line_word_timings_fn=lambda lines: lines,
        enforce_monotonic_line_starts_fn=lambda lines: lines,
        enforce_non_overlapping_lines_fn=lambda lines: lines,
        logger=wi.logger,
    )

    assert "hooks" in kwargs
    assert isinstance(kwargs["hooks"], AlignmentPassHooks)
    assert kwargs["runtime_config"] == WhisperRuntimeConfig(profile="safe")
    assert "transcribe_vocals_fn" not in kwargs


def test_align_lrc_text_pipeline_uses_whisperx_for_sparse_transcript(monkeypatch):
    lines = [
        Line(words=[Word(text="a", start_time=1.0, end_time=1.2)]),
        Line(words=[Word(text="b", start_time=2.0, end_time=2.2)]),
    ]
    forced_lines = [
        Line(words=[Word(text="a", start_time=1.5, end_time=1.8)]),
        Line(words=[Word(text="b", start_time=2.5, end_time=2.8)]),
    ]
    whisper_words = [TranscriptionWord(text="a", start=1.4, end=1.8, probability=0.9)]
    segments = [
        TranscriptionSegment(start=1.4, end=1.8, text="a", words=[whisper_words[0]])
    ]

    monkeypatch.setattr(
        "y2karaoke.core.components.whisper.whisper_integration_align.align_lines_with_whisperx",
        lambda *_args, **_kwargs: (
            forced_lines,
            {"forced_line_coverage": 1.0, "forced_word_coverage": 1.0},
        ),
    )

    mapped, corrections, metrics = align_lrc_text_to_whisper_timings_impl(
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
        extract_lrc_words_all_fn=lambda _in_lines: [],
        build_phoneme_tokens_from_lrc_words_fn=lambda *_a, **_k: [],
        build_phoneme_tokens_from_whisper_words_fn=lambda *_a, **_k: [],
        build_syllable_tokens_from_phonemes_fn=lambda *_a, **_k: [],
        build_segment_text_overlap_assignments_fn=lambda *_a, **_k: {},
        build_phoneme_dtw_path_fn=lambda *_a, **_k: [],
        build_word_assignments_from_phoneme_path_fn=lambda *_a, **_k: {},
        build_block_segmented_syllable_assignments_fn=lambda *_a, **_k: {},
        map_lrc_words_to_whisper_fn=lambda *_a, **_k: (lines, 0, 0.0, set()),
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
        normalize_line_word_timings_fn=lambda lines: lines,
        enforce_monotonic_line_starts_fn=lambda lines: lines,
        enforce_non_overlapping_lines_fn=lambda lines: lines,
        logger=wi.logger,
    )

    assert mapped == forced_lines
    assert metrics["whisperx_forced"] == 1.0
    assert any(
        "WhisperX transcript-constrained forced alignment" in msg for msg in corrections
    )


def test_align_lrc_text_pipeline_uses_whisperx_for_tail_shortfall_experiment(
    monkeypatch,
):
    lines = [
        Line(words=[Word(text="a", start_time=10.0, end_time=10.2)]),
        Line(words=[Word(text="b", start_time=195.0, end_time=195.2)]),
    ]
    forced_lines = [
        Line(words=[Word(text="a", start_time=10.5, end_time=10.8)]),
        Line(words=[Word(text="b", start_time=195.5, end_time=195.8)]),
    ]
    whisper_words = [
        TranscriptionWord(
            text=f"w{i}",
            start=100.0 + i * 0.7,
            end=100.2 + i * 0.7,
            probability=0.9,
        )
        for i in range(72)
    ] + [
        TranscriptionWord(
            text=f"tail{i}",
            start=183.0 + i * 0.1,
            end=183.05 + i * 0.1,
            probability=0.9,
        )
        for i in range(8)
    ]
    segments = [
        TranscriptionSegment(
            start=100.0 + i * 4.0,
            end=102.0 + i * 4.0,
            text=f"s{i}",
            words=[],
        )
        for i in range(19)
    ]

    monkeypatch.setattr(
        "y2karaoke.core.components.whisper.whisper_integration_align.align_lines_with_whisperx",
        lambda *_args, **_kwargs: (
            forced_lines,
            {"forced_line_coverage": 1.0, "forced_word_coverage": 1.0},
        ),
    )

    mapped, corrections, metrics = align_lrc_text_to_whisper_timings_impl(
        lines,
        vocals_path="vocals.wav",
        language="fr",
        model_size="base",
        aggressive=False,
        temperature=0.0,
        min_similarity=0.15,
        audio_features=None,
        lenient_vocal_activity_threshold=0.3,
        lenient_activity_bonus=0.4,
        low_word_confidence_threshold=0.5,
        runtime_config=WhisperRuntimeConfig(tail_shortfall_forced_fallback=True),
        transcribe_vocals_fn=lambda *_a, **_k: (segments, whisper_words, "fr", "base"),
        extract_audio_features_fn=lambda *_a, **_k: None,
        dedupe_whisper_segments_fn=lambda s: s,
        trim_whisper_transcription_by_lyrics_fn=lambda s, w, _t: (s, w, None),
        fill_vocal_activity_gaps_fn=lambda w, _a, _t, segments=None: (w, segments),
        dedupe_whisper_words_fn=lambda w: w,
        extract_lrc_words_all_fn=lambda _in_lines: [],
        build_phoneme_tokens_from_lrc_words_fn=lambda *_a, **_k: [],
        build_phoneme_tokens_from_whisper_words_fn=lambda *_a, **_k: [],
        build_syllable_tokens_from_phonemes_fn=lambda *_a, **_k: [],
        build_segment_text_overlap_assignments_fn=lambda *_a, **_k: {},
        build_phoneme_dtw_path_fn=lambda *_a, **_k: [],
        build_word_assignments_from_phoneme_path_fn=lambda *_a, **_k: {},
        build_block_segmented_syllable_assignments_fn=lambda *_a, **_k: {},
        map_lrc_words_to_whisper_fn=lambda *_a, **_k: (lines, 0, 0.0, set()),
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
        normalize_line_word_timings_fn=lambda lines: lines,
        enforce_monotonic_line_starts_fn=lambda lines: lines,
        enforce_non_overlapping_lines_fn=lambda lines: lines,
        logger=wi.logger,
    )

    assert mapped == forced_lines
    assert metrics["whisperx_forced"] == 1.0
    assert any("early Whisper transcript tail shortfall" in msg for msg in corrections)


def test_correct_timing_with_whisper_uses_whisperx_when_sparse(monkeypatch):
    lines = [Line(words=[Word(text="hello", start_time=10.0, end_time=11.0)])]
    forced = [Line(words=[Word(text="hello", start_time=10.3, end_time=11.2)])]
    words = [te.TranscriptionWord(start=10.1, end=10.6, text="hello", probability=0.9)]
    segments = [
        te.TranscriptionSegment(start=10.1, end=10.6, text="hello", words=words)
    ]

    monkeypatch.setattr(
        "y2karaoke.core.components.whisper.whisper_integration_correct.align_lines_with_whisperx",
        lambda *_args, **_kwargs: (
            forced,
            {"forced_line_coverage": 1.0, "forced_word_coverage": 1.0},
        ),
    )

    with wi.use_whisper_integration_hooks(
        transcribe_vocals_fn=lambda *_: (segments, words, "en", "base"),
        extract_audio_features_fn=lambda *_: None,
    ):
        aligned, corrections, metrics = te.correct_timing_with_whisper(
            lines, "vocals.wav"
        )

    assert aligned[0].start_time == pytest.approx(lines[0].start_time, abs=0.05)
    assert metrics["whisperx_forced"] == 1.0
    assert any(
        "WhisperX transcript-constrained forced alignment" in c for c in corrections
    )


def test_correct_timing_with_whisper_rejects_low_coverage_whisperx(monkeypatch):
    lines = [Line(words=[Word(text="hello", start_time=10.0, end_time=11.0)])]
    forced = [Line(words=[Word(text="hello", start_time=15.0, end_time=16.0)])]
    words = [te.TranscriptionWord(start=10.1, end=10.6, text="hello", probability=0.9)]
    segments = [
        te.TranscriptionSegment(start=10.1, end=10.6, text="hello", words=words)
    ]

    monkeypatch.setattr(
        "y2karaoke.core.components.whisper.whisper_integration_correct.align_lines_with_whisperx",
        lambda *_args, **_kwargs: (
            forced,
            {"forced_line_coverage": 0.1, "forced_word_coverage": 0.1},
        ),
    )

    with wi.use_whisper_integration_hooks(
        transcribe_vocals_fn=lambda *_: (segments, words, "en", "base"),
        extract_audio_features_fn=lambda *_: None,
    ):
        aligned, corrections, metrics = te.correct_timing_with_whisper(
            lines, "vocals.wav"
        )

    assert aligned[0].start_time == pytest.approx(lines[0].start_time)
    assert not any(
        "WhisperX transcript-constrained forced alignment" in c for c in corrections
    )
    assert metrics.get("whisperx_forced", 0.0) == 0.0


def test_correct_timing_with_whisper_rolls_back_when_dtw_has_no_evidence():
    lines = [Line(words=[Word(text="hello", start_time=10.0, end_time=11.0)])]
    words = [
        te.TranscriptionWord(
            start=10.0 + 0.1 * i, end=10.05 + 0.1 * i, text=f"w{i}", probability=0.9
        )
        for i in range(81)
    ]
    segments = [
        te.TranscriptionSegment(
            start=10.0 + 2.0 * i,
            end=12.0 + 2.0 * i,
            text=f"s{i}",
            words=words[i * 16 : (i + 1) * 16],
        )
        for i in range(5)
    ]
    misaligned = [Line(words=[Word(text="hello", start_time=35.0, end_time=36.0)])]

    with wi.use_whisper_integration_hooks(
        transcribe_vocals_fn=lambda *_: (segments, words, "en", "base"),
        extract_audio_features_fn=lambda *_: None,
        assess_lrc_quality_fn=lambda *_a, **_k: (0.2, []),
        align_dtw_whisper_with_data_fn=lambda *_a, **_k: (
            misaligned,
            ["dtw-shifted"],
            {"matched_ratio": 0.0, "avg_similarity": 0.0, "line_coverage": 0.0},
            [],
            {},
        ),
    ):
        aligned, corrections, metrics = te.correct_timing_with_whisper(
            lines, "vocals.wav"
        )

    assert aligned[0].start_time == pytest.approx(lines[0].start_time)
    assert metrics["no_evidence_fallback"] == 1.0
    assert len(corrections) == 1
    assert any("insufficient DTW alignment evidence" in c for c in corrections)


def test_align_lrc_text_pipeline_enforces_monotonic_non_overlapping_invariants():
    lines = [
        Line(words=[Word(text="a", start_time=10.0, end_time=11.0)]),
        Line(words=[Word(text="b", start_time=12.0, end_time=13.0)]),
        Line(words=[Word(text="c", start_time=14.0, end_time=15.0)]),
    ]
    whisper_words = [
        TranscriptionWord(text="a", start=10.0, end=11.0, probability=0.9),
        TranscriptionWord(text="b", start=12.0, end=13.0, probability=0.9),
        TranscriptionWord(text="c", start=14.0, end=15.0, probability=0.9),
    ]
    segments = [
        TranscriptionSegment(start=10.0, end=11.0, text="a", words=[whisper_words[0]]),
        TranscriptionSegment(start=12.0, end=13.0, text="b", words=[whisper_words[1]]),
        TranscriptionSegment(start=14.0, end=15.0, text="c", words=[whisper_words[2]]),
    ]
    audio_features = AudioFeatures(
        onset_times=np.array([], dtype=float),
        silence_regions=[],
        vocal_start=0.0,
        vocal_end=20.0,
        duration=20.0,
        energy_envelope=np.array([], dtype=float),
        energy_times=np.array([], dtype=float),
    )

    overlapped = [
        Line(words=[Word(text="a", start_time=10.0, end_time=12.0)]),
        Line(words=[]),
        Line(words=[Word(text="b", start_time=11.5, end_time=12.4)]),
    ]

    mapped, _corrections, _metrics = align_lrc_text_to_whisper_timings_impl(
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
        extract_lrc_words_all_fn=lambda in_lines: [
            {"text": wd.text, "line_idx": li, "word_idx": wi}
            for li, line in enumerate(in_lines)
            for wi, wd in enumerate(line.words)
        ],
        build_phoneme_tokens_from_lrc_words_fn=lambda _w, _l: [1, 2, 3],
        build_phoneme_tokens_from_whisper_words_fn=lambda _w, _l: [1, 2, 3],
        build_syllable_tokens_from_phonemes_fn=lambda _p: [1],
        build_segment_text_overlap_assignments_fn=lambda _lw, _aw, _s: {
            0: [0],
            1: [1],
            2: [2],
        },
        build_phoneme_dtw_path_fn=lambda *_a, **_k: [],
        build_word_assignments_from_phoneme_path_fn=lambda *_a, **_k: {
            0: [0],
            1: [1],
            2: [2],
        },
        build_block_segmented_syllable_assignments_fn=lambda *_a, **_k: {
            0: [0],
            1: [1],
            2: [2],
        },
        map_lrc_words_to_whisper_fn=lambda *_a, **_k: (overlapped, 3, 3.0, {0, 1, 2}),
        shift_repeated_lines_to_next_whisper_fn=lambda ml, _aw: ml,
        enforce_monotonic_line_starts_whisper_fn=lambda ml, _aw: ml,
        resolve_line_overlaps_fn=wm._resolve_line_overlaps,
        extend_line_to_trailing_whisper_matches_fn=lambda ml, _aw: ml,
        pull_late_lines_to_matching_segments_fn=lambda ml, _s, _lang: ml,
        retime_short_interjection_lines_fn=lambda ml, _s: ml,
        snap_first_word_to_whisper_onset_fn=lambda ml, _aw: ml,
        interpolate_unmatched_lines_fn=lambda ml, _set: ml,
        refine_unmatched_lines_with_onsets_fn=lambda ml, _set, _vp: ml,
        pull_lines_forward_for_continuous_vocals_fn=lambda ml, _af: (ml, 0),
        normalize_line_word_timings_fn=lambda lines: lines,
        enforce_monotonic_line_starts_fn=lambda lines: lines,
        enforce_non_overlapping_lines_fn=lambda lines: lines,
        logger=wi.logger,
    )

    non_empty = [line for line in mapped if line.words]
    for idx in range(len(non_empty) - 1):
        assert non_empty[idx].end_time <= non_empty[idx + 1].start_time
    for line in non_empty:
        for word in line.words:
            assert word.end_time - word.start_time >= 0.06


def test_align_lrc_text_pipeline_falls_back_to_block_dtw_for_moderate_overlap():
    lines = [
        Line(words=[Word(text="a", start_time=1.0, end_time=1.2)]),
        Line(words=[Word(text="b", start_time=2.0, end_time=2.2)]),
        Line(words=[Word(text="c", start_time=3.0, end_time=3.2)]),
        Line(words=[Word(text="d", start_time=4.0, end_time=4.2)]),
        Line(words=[Word(text="e", start_time=5.0, end_time=5.2)]),
    ]
    whisper_words = [
        TranscriptionWord(text="a", start=1.0, end=1.2, probability=0.9),
        TranscriptionWord(text="b", start=2.0, end=2.2, probability=0.9),
        TranscriptionWord(text="c", start=3.0, end=3.2, probability=0.9),
        TranscriptionWord(text="d", start=4.0, end=4.2, probability=0.9),
        TranscriptionWord(text="e", start=5.0, end=5.2, probability=0.9),
    ]
    segments = [
        TranscriptionSegment(start=1.0, end=2.2, text="a b", words=whisper_words[:2]),
        TranscriptionSegment(start=3.0, end=5.2, text="c d e", words=whisper_words[2:]),
    ]
    audio_features = AudioFeatures(
        onset_times=np.array([], dtype=float),
        silence_regions=[],
        vocal_start=0.0,
        vocal_end=10.0,
        duration=10.0,
        energy_envelope=np.array([], dtype=float),
        energy_times=np.array([], dtype=float),
    )
    calls = {"block": 0}

    def build_block_assignments(*_a, **_k):
        calls["block"] += 1
        return {0: [0], 1: [1], 2: [2], 3: [3], 4: [4]}

    mapped, _corrections, _metrics = align_lrc_text_to_whisper_timings_impl(
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
        extract_lrc_words_all_fn=lambda in_lines: [
            {"text": wd.text, "line_idx": li, "word_idx": wi}
            for li, line in enumerate(in_lines)
            for wi, wd in enumerate(line.words)
        ],
        build_phoneme_tokens_from_lrc_words_fn=lambda _w, _l: [1, 2, 3, 4, 5],
        build_phoneme_tokens_from_whisper_words_fn=lambda _w, _l: [1, 2, 3, 4, 5],
        build_syllable_tokens_from_phonemes_fn=lambda _p: [1, 2],
        # 2/5 coverage (40%) should now trigger DTW fallback path.
        build_segment_text_overlap_assignments_fn=lambda _lw, _aw, _s: {0: [0], 1: [1]},
        build_phoneme_dtw_path_fn=lambda *_a, **_k: [],
        build_word_assignments_from_phoneme_path_fn=lambda *_a, **_k: {},
        build_block_segmented_syllable_assignments_fn=build_block_assignments,
        map_lrc_words_to_whisper_fn=lambda *_a, **_k: (lines, 5, 5.0, {0, 1, 2, 3, 4}),
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
        normalize_line_word_timings_fn=lambda lines: lines,
        enforce_monotonic_line_starts_fn=lambda lines: lines,
        enforce_non_overlapping_lines_fn=lambda lines: lines,
        logger=wi.logger,
    )

    assert calls["block"] == 1
    assert len(mapped) == 5


def test_align_lrc_text_pipeline_filters_low_confidence_whisper_words():
    lines = [Line(words=[Word(text="hello", start_time=1.0, end_time=1.2)])]
    whisper_words = []
    for idx in range(30):
        prob = 0.2 if idx < 8 else 0.95
        whisper_words.append(
            TranscriptionWord(
                text=f"w{idx}",
                start=1.0 + idx * 0.1,
                end=1.05 + idx * 0.1,
                probability=prob,
            )
        )
    segments = [
        TranscriptionSegment(start=1.0, end=4.0, text="s", words=whisper_words),
    ]
    audio_features = AudioFeatures(
        onset_times=np.array([], dtype=float),
        silence_regions=[],
        vocal_start=0.0,
        vocal_end=10.0,
        duration=10.0,
        energy_envelope=np.array([], dtype=float),
        energy_times=np.array([], dtype=float),
    )
    observed = {"word_count": None}

    def capture_map(_lines, _lrc_words, all_words, *_rest):
        observed["word_count"] = len(all_words)
        return _lines, 1, 1.0, {0}

    _mapped, _corrections, _metrics = align_lrc_text_to_whisper_timings_impl(
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
        extract_lrc_words_all_fn=lambda in_lines: [
            {"text": wd.text, "line_idx": li, "word_idx": wi}
            for li, line in enumerate(in_lines)
            for wi, wd in enumerate(line.words)
        ],
        build_phoneme_tokens_from_lrc_words_fn=lambda _w, _l: [1],
        build_phoneme_tokens_from_whisper_words_fn=lambda _w, _l: [1],
        build_syllable_tokens_from_phonemes_fn=lambda _p: [1],
        build_segment_text_overlap_assignments_fn=lambda _lw, _aw, _s: {0: [0]},
        build_phoneme_dtw_path_fn=lambda *_a, **_k: [],
        build_word_assignments_from_phoneme_path_fn=lambda *_a, **_k: {0: [0]},
        build_block_segmented_syllable_assignments_fn=lambda *_a, **_k: {0: [0]},
        map_lrc_words_to_whisper_fn=capture_map,
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
        normalize_line_word_timings_fn=lambda lines: lines,
        enforce_monotonic_line_starts_fn=lambda lines: lines,
        enforce_non_overlapping_lines_fn=lambda lines: lines,
        logger=wi.logger,
    )

    assert observed["word_count"] == 22


def test_align_lrc_text_pipeline_preserves_line_count_under_repeated_reset_like_jitter():
    lines = [
        Line(words=[Word(text="Lately", start_time=10.0, end_time=10.8)]),
        Line(words=[Word(text="Dreaming", start_time=10.9, end_time=11.6)]),
        Line(words=[Word(text="But", start_time=11.7, end_time=12.2)]),
        Line(words=[Word(text="Said", start_time=12.3, end_time=12.8)]),
    ]
    whisper_words = [
        TranscriptionWord(text="Lately", start=10.0, end=10.3, probability=0.9),
        TranscriptionWord(text="Dreaming", start=10.4, end=10.8, probability=0.9),
        TranscriptionWord(text="But", start=10.9, end=11.2, probability=0.9),
        TranscriptionWord(text="Said", start=11.3, end=11.6, probability=0.9),
        # Repeat-cycle reset style sequence
        TranscriptionWord(text="Lately", start=11.7, end=12.0, probability=0.9),
        TranscriptionWord(text="Dreaming", start=12.1, end=12.4, probability=0.9),
        TranscriptionWord(text="But", start=12.5, end=12.7, probability=0.9),
        TranscriptionWord(text="Said", start=12.8, end=13.0, probability=0.9),
    ]
    segments = [
        TranscriptionSegment(start=10.0, end=13.0, text="verse", words=whisper_words)
    ]
    audio_features = AudioFeatures(
        onset_times=np.array([], dtype=float),
        silence_regions=[],
        vocal_start=0.0,
        vocal_end=20.0,
        duration=20.0,
        energy_envelope=np.array([], dtype=float),
        energy_times=np.array([], dtype=float),
    )

    jittered = [
        Line(words=[Word(text="Lately", start_time=10.2, end_time=10.9)]),
        Line(words=[Word(text="Dreaming", start_time=9.9, end_time=10.5)]),
        Line(words=[Word(text="But", start_time=10.0, end_time=10.6)]),
        Line(words=[Word(text="Said", start_time=12.2, end_time=12.9)]),
    ]

    mapped, _corrections, _metrics = align_lrc_text_to_whisper_timings_impl(
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
        extract_lrc_words_all_fn=lambda in_lines: [
            {"text": wd.text, "line_idx": li, "word_idx": wi}
            for li, line in enumerate(in_lines)
            for wi, wd in enumerate(line.words)
        ],
        build_phoneme_tokens_from_lrc_words_fn=lambda _w, _l: [1, 2, 3, 4],
        build_phoneme_tokens_from_whisper_words_fn=lambda _w, _l: [1, 2, 3, 4],
        build_syllable_tokens_from_phonemes_fn=lambda _p: [1, 2],
        build_segment_text_overlap_assignments_fn=lambda _lw, _aw, _s: {
            0: [0],
            1: [1],
            2: [2],
            3: [3],
        },
        build_phoneme_dtw_path_fn=lambda *_a, **_k: [],
        build_word_assignments_from_phoneme_path_fn=lambda *_a, **_k: {
            0: [0],
            1: [1],
            2: [2],
            3: [3],
        },
        build_block_segmented_syllable_assignments_fn=lambda *_a, **_k: {
            0: [0],
            1: [1],
            2: [2],
            3: [3],
        },
        map_lrc_words_to_whisper_fn=lambda *_a, **_k: (jittered, 4, 4.0, {0, 1, 2, 3}),
        shift_repeated_lines_to_next_whisper_fn=wm._shift_repeated_lines_to_next_whisper,
        enforce_monotonic_line_starts_whisper_fn=wm._enforce_monotonic_line_starts_whisper,
        resolve_line_overlaps_fn=wm._resolve_line_overlaps,
        extend_line_to_trailing_whisper_matches_fn=lambda ml, _aw: ml,
        pull_late_lines_to_matching_segments_fn=lambda ml, _s, _lang: ml,
        retime_short_interjection_lines_fn=lambda ml, _s: ml,
        snap_first_word_to_whisper_onset_fn=lambda ml, _aw, **_kw: ml,
        interpolate_unmatched_lines_fn=lambda ml, _set: ml,
        refine_unmatched_lines_with_onsets_fn=lambda ml, _set, _vp: ml,
        pull_lines_forward_for_continuous_vocals_fn=lambda ml, _af: (ml, 0),
        normalize_line_word_timings_fn=lambda lines: lines,
        enforce_monotonic_line_starts_fn=lambda lines: lines,
        enforce_non_overlapping_lines_fn=lambda lines: lines,
        logger=wi.logger,
    )

    assert len(mapped) == len(lines)
    assert all(line.words for line in mapped)
    starts = [line.start_time for line in mapped]
    assert starts == sorted(starts)
