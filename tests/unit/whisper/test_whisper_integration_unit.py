import numpy as np
import pytest
from typing import Any, cast
import y2karaoke.core.components.alignment.timing_evaluator as te
from y2karaoke.core.models import Line, Word
import y2karaoke.core.components.whisper.whisper_integration as wi
import y2karaoke.core.components.whisper.whisper_integration_pipeline as wip
from y2karaoke.core.components.whisper.whisper_integration_pipeline import (
    align_lrc_text_to_whisper_timings_impl,
)
from y2karaoke.core.components.whisper import whisper_mapping as wm
from y2karaoke.core.components.whisper import whisper_integration_baseline as wib
from y2karaoke.core.components.whisper import whisper_integration_transcribe as witx
from y2karaoke.core.components.alignment.timing_models import (
    AudioFeatures,
    TranscriptionWord,
    TranscriptionSegment,
)

wi_any = cast(Any, wi)


def test_whisper_lang_to_epitran():
    assert wi_any._whisper_lang_to_epitran("en") == "eng-Latn"
    assert wi_any._whisper_lang_to_epitran("fr") == "fra-Latn"
    assert wi_any._whisper_lang_to_epitran("zh") == "cmn-Hans"
    assert wi_any._whisper_lang_to_epitran("unknown") == "eng-Latn"


def test_assess_lrc_quality():
    lines = [
        Line(words=[Word(text="hello", start_time=10.0, end_time=11.0)]),
        Line(words=[Word(text="world", start_time=20.0, end_time=21.0)]),
    ]
    whisper_words = [
        TranscriptionWord(text="hello", start=10.1, end=11.1, probability=1.0),
        TranscriptionWord(text="world", start=25.0, end=26.0, probability=1.0),
    ]
    # Tolerance 1.5s. 10.0 vs 10.1 is fine. 20.0 vs 25.0 is too far.
    quality, assessments = wi._assess_lrc_quality(
        lines, whisper_words, "eng-Latn", tolerance=1.5
    )
    assert quality == 0.5
    # assessments is List[Tuple[int, float, float]] (line_idx, lrc_time, best_whisper_time)
    assert assessments[0][0] == 0
    assert assessments[1][0] == 1


def test_trim_whisper_transcription_by_lyrics():
    segments = [
        TranscriptionSegment(start=0, end=5, text="intro", words=[]),
        TranscriptionSegment(start=10, end=15, text="first line", words=[]),
        TranscriptionSegment(start=20, end=25, text="outro", words=[]),
    ]
    words = [
        TranscriptionWord(text="intro", start=0, end=5, probability=1.0),
        TranscriptionWord(text="first", start=10, end=12, probability=1.0),
        TranscriptionWord(text="line", start=12, end=15, probability=1.0),
        TranscriptionWord(text="outro", start=20, end=25, probability=1.0),
    ]
    line_texts = ["first line"]

    trimmed_segs, trimmed_words, end_time = wi._trim_whisper_transcription_by_lyrics(
        segments, words, line_texts
    )

    # It should keep segments up to the one matching "first line" + buffer
    # And potentially some after.
    assert end_time > 0
    assert len(trimmed_segs) <= len(segments)


def test_trim_whisper_transcription_skips_when_match_far_from_tail():
    segments = [
        TranscriptionSegment(start=0, end=5, text="intro", words=[]),
        TranscriptionSegment(start=10, end=15, text="target line", words=[]),
        TranscriptionSegment(start=200, end=205, text="very late words", words=[]),
    ]
    words = [
        TranscriptionWord(text="intro", start=0, end=5, probability=1.0),
        TranscriptionWord(text="target", start=10, end=12, probability=1.0),
        TranscriptionWord(text="line", start=12, end=15, probability=1.0),
        TranscriptionWord(text="late", start=200, end=205, probability=1.0),
    ]
    line_texts = ["target line"]

    trimmed_segs, trimmed_words, end_time = wi._trim_whisper_transcription_by_lyrics(
        segments, words, line_texts
    )

    assert end_time is None
    assert len(trimmed_segs) == len(segments)
    assert len(trimmed_words) == len(words)


def test_build_word_to_segment_index():
    segments = [
        TranscriptionSegment(
            start=0,
            end=10,
            text="s1",
            words=[
                TranscriptionWord(text="w1", start=1, end=2, probability=1.0),
                TranscriptionWord(text="w2", start=3, end=4, probability=1.0),
            ],
        ),
        TranscriptionSegment(
            start=10,
            end=20,
            text="s2",
            words=[
                TranscriptionWord(text="w3", start=11, end=12, probability=1.0),
            ],
        ),
    ]
    all_words = (segments[0].words or []) + (segments[1].words or [])
    # wi._build_word_to_segment_index(all_words, segments)
    idx_map = wi_any._build_word_to_segment_index(all_words, segments)
    assert idx_map[0] == 0
    assert idx_map[1] == 0
    assert idx_map[2] == 1


def test_find_segment_for_time():
    segments = [
        TranscriptionSegment(start=0, end=10, text="s1", words=[]),
        TranscriptionSegment(start=15, end=25, text="s2", words=[]),
    ]
    # wi._find_segment_for_time(time, segments)
    assert wi_any._find_segment_for_time(5.0, segments) == 0
    assert wi_any._find_segment_for_time(20.0, segments) == 1


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

    mapped, corrections, _metrics = align_lrc_text_to_whisper_timings_impl(
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
        logger=wi.logger,
    )

    assert mapped[1].start_time == 98.0
    assert any("continuous vocals" in msg for msg in corrections)


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
        logger=wi.logger,
    )

    assert mapped == forced_lines
    assert metrics["whisperx_forced"] == 1.0
    assert any(
        "WhisperX transcript-constrained forced alignment" in msg for msg in corrections
    )


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
        logger=wi.logger,
    )

    assert observed["word_count"] == 22


def test_should_rollback_short_line_degradation_triggers():
    original = [
        Line(
            words=[
                Word(text="a", start_time=0.0, end_time=0.4),
                Word(text="b", start_time=0.4, end_time=0.8),
                Word(text="c", start_time=0.8, end_time=1.2),
            ]
        )
        for _ in range(12)
    ]
    degraded = [
        Line(
            words=[
                Word(text="a", start_time=0.0, end_time=0.05),
                Word(text="b", start_time=0.05, end_time=0.1),
                Word(text="c", start_time=0.1, end_time=0.15),
            ]
        )
        for _ in range(12)
    ]

    rollback, before, after = wip._should_rollback_short_line_degradation(
        original, degraded
    )

    assert rollback
    assert before == 0
    assert after == 12


def test_should_rollback_short_line_degradation_ignores_small_change():
    original = [
        Line(
            words=[
                Word(text="a", start_time=0.0, end_time=0.2),
                Word(text="b", start_time=0.2, end_time=0.4),
                Word(text="c", start_time=0.4, end_time=0.6),
            ]
        )
        for _ in range(20)
    ]
    slightly_worse = list(original)
    slightly_worse[0] = Line(
        words=[
            Word(text="a", start_time=0.0, end_time=0.05),
            Word(text="b", start_time=0.05, end_time=0.1),
            Word(text="c", start_time=0.1, end_time=0.15),
        ]
    )
    slightly_worse[1] = Line(
        words=[
            Word(text="a", start_time=0.7, end_time=0.75),
            Word(text="b", start_time=0.75, end_time=0.8),
            Word(text="c", start_time=0.8, end_time=0.85),
        ]
    )

    rollback, before, after = wip._should_rollback_short_line_degradation(
        original, slightly_worse
    )

    assert not rollback
    assert before == 0
    assert after == 2


def test_should_accept_whisperx_upgrade_accepts_sane_shape():
    base_words = [
        TranscriptionWord(text="a", start=0.2, end=0.4, probability=0.9),
        TranscriptionWord(text="b", start=8.0, end=8.2, probability=0.9),
    ]
    base_segments = [TranscriptionSegment(start=0.2, end=8.2, text="ab", words=[])]
    upgraded_words = [
        witx._WhisperxWord(
            start=float(i) * 0.12,
            end=float(i) * 0.12 + 0.08,
            text=f"w{i}",
            probability=0.9,
        )
        for i in range(120)
    ]
    upgraded_segments = [
        witx._WhisperxSegment(
            start=float(i),
            end=float(i) + 0.9,
            text=f"s{i}",
            words=upgraded_words[i * 20 : (i + 1) * 20],
        )
        for i in range(6)
    ]

    accepted = witx._should_accept_whisperx_upgrade(
        base_segments=base_segments,
        base_words=base_words,
        upgraded_segments=upgraded_segments,
        upgraded_words=upgraded_words,
        logger=wi.logger,
    )

    assert accepted is True


def test_should_accept_whisperx_upgrade_rejects_excessive_overlap():
    base_words = [
        TranscriptionWord(text="a", start=0.0, end=0.2, probability=0.9),
        TranscriptionWord(text="b", start=5.0, end=5.2, probability=0.9),
    ]
    base_segments = [TranscriptionSegment(start=0.0, end=5.2, text="ab", words=[])]
    upgraded_words = []
    for i in range(120):
        if i == 0:
            start = 0.0
        else:
            start = upgraded_words[-1].start + 0.03
        end = start + 0.08
        if i % 10 == 0 and i > 0:
            start -= 0.08
            end -= 0.08
        upgraded_words.append(
            witx._WhisperxWord(start=start, end=end, text=f"w{i}", probability=0.9)
        )
    upgraded_segments = [
        witx._WhisperxSegment(
            start=float(i),
            end=float(i) + 1.0,
            text=f"s{i}",
            words=upgraded_words[i * 20 : (i + 1) * 20],
        )
        for i in range(6)
    ]

    accepted = witx._should_accept_whisperx_upgrade(
        base_segments=base_segments,
        base_words=base_words,
        upgraded_segments=upgraded_segments,
        upgraded_words=upgraded_words,
        logger=wi.logger,
    )

    assert accepted is False


def test_should_accept_whisperx_upgrade_rejects_shorter_span():
    base_words = [
        TranscriptionWord(text="a", start=2.0, end=2.2, probability=0.9),
        TranscriptionWord(text="b", start=20.0, end=20.4, probability=0.9),
    ]
    base_segments = [TranscriptionSegment(start=2.0, end=20.4, text="ab", words=[])]
    upgraded_words = [
        witx._WhisperxWord(
            start=float(i) * 0.1,
            end=float(i) * 0.1 + 0.07,
            text=f"w{i}",
            probability=0.9,
        )
        for i in range(120)
    ]
    upgraded_segments = [
        witx._WhisperxSegment(
            start=float(i) * 2.0,
            end=float(i) * 2.0 + 1.0,
            text=f"s{i}",
            words=upgraded_words[i * 20 : (i + 1) * 20],
        )
        for i in range(6)
    ]

    accepted = witx._should_accept_whisperx_upgrade(
        base_segments=base_segments,
        base_words=base_words,
        upgraded_segments=upgraded_segments,
        upgraded_words=upgraded_words,
        logger=wi.logger,
    )

    assert accepted is False


def test_normalize_whisperx_segments_enforces_monotonic_words():
    segments = [
        witx._WhisperxSegment(
            start=0.0,
            end=1.0,
            text="a",
            words=[
                witx._WhisperxWord(start=0.10, end=0.25, text="w1", probability=0.9),
                witx._WhisperxWord(start=0.21, end=0.30, text="w2", probability=0.9),
            ],
        ),
        witx._WhisperxSegment(
            start=0.9,
            end=1.3,
            text="b",
            words=[
                witx._WhisperxWord(start=0.28, end=0.40, text="w3", probability=0.9),
            ],
        ),
    ]

    normalized_segments, normalized_words = witx._normalize_whisperx_segments(segments)

    assert len(normalized_segments) == 2
    assert len(normalized_words) == 3
    starts = [w.start for w in normalized_words]
    ends = [w.end for w in normalized_words]
    assert starts == sorted(starts)
    assert all(ends[i] <= starts[i + 1] for i in range(len(starts) - 1))
    assert normalized_segments[0].start == pytest.approx(normalized_words[0].start)
    assert normalized_segments[-1].end == pytest.approx(normalized_words[-1].end)


def test_constrain_line_starts_to_baseline_skips_large_shift():
    mapped = [
        Line(words=[Word(text="hello", start_time=20.0, end_time=21.5)]),
    ]
    baseline = [
        Line(words=[Word(text="hello", start_time=12.0, end_time=13.5)]),
    ]

    constrained = wib._constrain_line_starts_to_baseline(
        mapped, baseline, max_shift_sec=2.5
    )

    assert constrained[0].start_time == pytest.approx(20.0)
    assert constrained[0].end_time == pytest.approx(21.5)


def test_constrain_line_starts_to_baseline_applies_small_shift():
    mapped = [
        Line(words=[Word(text="hello", start_time=10.0, end_time=11.0)]),
    ]
    baseline = [
        Line(words=[Word(text="hello", start_time=11.2, end_time=12.2)]),
    ]

    constrained = wib._constrain_line_starts_to_baseline(
        mapped, baseline, max_shift_sec=2.5
    )

    assert constrained[0].start_time == pytest.approx(11.2)


def test_restore_implausibly_short_lines_restores_newly_compressed():
    baseline = [
        Line(
            words=[
                Word(text="one", start_time=10.0, end_time=10.4),
                Word(text="two", start_time=10.4, end_time=10.8),
                Word(text="three", start_time=10.8, end_time=11.2),
            ]
        )
    ]
    aligned = [
        Line(
            words=[
                Word(text="one", start_time=10.0, end_time=10.05),
                Word(text="two", start_time=10.05, end_time=10.1),
                Word(text="three", start_time=10.1, end_time=10.15),
            ]
        )
    ]

    repaired, restored = wib._restore_implausibly_short_lines(baseline, aligned)

    assert restored == 1
    assert repaired[0].start_time == pytest.approx(baseline[0].start_time)
    assert repaired[0].end_time == pytest.approx(baseline[0].end_time)


def test_restore_implausibly_short_lines_keeps_legitimate_short_baseline():
    baseline = [
        Line(
            words=[
                Word(text="a", start_time=1.0, end_time=1.03),
                Word(text="b", start_time=1.03, end_time=1.06),
                Word(text="c", start_time=1.06, end_time=1.09),
            ]
        )
    ]
    aligned = [
        Line(
            words=[
                Word(text="a", start_time=2.0, end_time=2.03),
                Word(text="b", start_time=2.03, end_time=2.06),
                Word(text="c", start_time=2.06, end_time=2.09),
            ]
        )
    ]

    repaired, restored = wib._restore_implausibly_short_lines(baseline, aligned)

    assert restored == 0
    assert repaired[0].start_time == pytest.approx(aligned[0].start_time)
