from y2karaoke.core.models import Line, Word
import y2karaoke.core.components.whisper.whisper_integration as wi
import y2karaoke.core.components.whisper.whisper_integration_pipeline as wip
import y2karaoke.core.components.whisper.whisper_integration_pipeline_align as wipa


def _dummy_align_call(lines):
    return wip.align_lrc_text_to_whisper_timings_impl(
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
        logger=wi.logger,
    )


def test_align_lrc_text_pipeline_applies_aggressive_retry_when_improved(monkeypatch):
    lines = [
        Line(words=[Word(text=f"w{i}", start_time=float(i), end_time=float(i) + 0.4)])
        for i in range(30)
    ]
    calls = {"normal": 0, "aggressive": 0}

    def fake_align(_lines, *_args, **_kwargs):
        aggressive = bool(_args[3])
        if aggressive:
            calls["aggressive"] += 1
            return (
                _lines,
                ["retry"],
                {
                    "matched_ratio": 0.86,
                    "line_coverage": 0.9,
                    "phonetic_similarity_coverage": 0.5,
                },
            )
        calls["normal"] += 1
        return (
            _lines,
            ["base"],
            {
                "matched_ratio": 0.81,
                "line_coverage": 0.84,
                "phonetic_similarity_coverage": 0.41,
            },
        )

    monkeypatch.setattr(wipa, "_align_lrc_text_to_whisper_timings_impl", fake_align)
    mapped, corrections, metrics = _dummy_align_call(lines)

    assert len(mapped) == len(lines)
    assert calls == {"normal": 1, "aggressive": 1}
    assert metrics["aggressive_retry_applied"] == 1.0
    assert any("Accepted aggressive Whisper retry" in msg for msg in corrections)


def test_align_lrc_text_pipeline_keeps_base_when_aggressive_retry_not_better(
    monkeypatch,
):
    lines = [
        Line(words=[Word(text=f"w{i}", start_time=float(i), end_time=float(i) + 0.4)])
        for i in range(30)
    ]

    def fake_align(_lines, *_args, **_kwargs):
        aggressive = bool(_args[3])
        if aggressive:
            return (
                _lines,
                ["retry"],
                {
                    "matched_ratio": 0.80,
                    "line_coverage": 0.82,
                    "phonetic_similarity_coverage": 0.42,
                },
            )
        return (
            _lines,
            ["base"],
            {
                "matched_ratio": 0.81,
                "line_coverage": 0.84,
                "phonetic_similarity_coverage": 0.41,
            },
        )

    monkeypatch.setattr(wipa, "_align_lrc_text_to_whisper_timings_impl", fake_align)
    _mapped, corrections, metrics = _dummy_align_call(lines)

    assert corrections == ["base"]
    assert metrics["aggressive_retry_attempted"] == 1.0
    assert metrics["aggressive_retry_applied"] == 0.0
