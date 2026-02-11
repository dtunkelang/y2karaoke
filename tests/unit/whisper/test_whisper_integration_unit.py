import numpy as np
from typing import Any, cast
from y2karaoke.core.models import Line, Word
import y2karaoke.core.components.whisper.whisper_integration as wi
import y2karaoke.core.components.whisper.whisper_integration_pipeline as wip
from y2karaoke.core.components.whisper.whisper_integration_pipeline import (
    align_lrc_text_to_whisper_timings_impl,
)
from y2karaoke.core.components.whisper import whisper_mapping as wm
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
