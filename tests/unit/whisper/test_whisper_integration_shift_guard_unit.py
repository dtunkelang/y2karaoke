import pytest

from y2karaoke.core.components.alignment.timing_models import (
    TranscriptionSegment,
    TranscriptionWord,
)
from y2karaoke.core.components.whisper import whisper_integration as wi
from y2karaoke.core.components.whisper import whisper_integration_align as wialign
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
        TranscriptionWord(text="i", start=19.4, end=19.5, probability=0.8),
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
