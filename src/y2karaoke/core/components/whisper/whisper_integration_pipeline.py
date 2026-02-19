"""Pipeline implementations for Whisper integration entry points."""

from typing import Any, Callable, Dict, List, Optional, Tuple

from ....utils.lex_lookup_installer import ensure_local_lex_lookup
from ... import models
from ... import phonetic_utils
from ..alignment import timing_models
from .whisper_integration_filters import (
    _filter_low_confidence_whisper_words as _filter_low_confidence_whisper_words_impl,
)

_MIN_SEGMENT_OVERLAP_COVERAGE = 0.45


def _clone_lines_for_fallback(lines: List[models.Line]) -> List[models.Line]:
    """Deep-copy lines so rollback logic can safely restore original timing."""
    return [
        models.Line(
            words=[
                models.Word(
                    text=w.text,
                    start_time=w.start_time,
                    end_time=w.end_time,
                    singer=w.singer,
                )
                for w in line.words
            ],
            singer=line.singer,
        )
        for line in lines
    ]


def _implausibly_short_multiword_count(lines: List[models.Line]) -> int:
    """Count suspiciously compressed multi-word lines."""
    count = 0
    for line in lines:
        word_count = len(line.words)
        if word_count < 3:
            continue
        duration = line.end_time - line.start_time
        if duration <= 0:
            count += 1
            continue
        if duration < 0.5 and (duration / max(word_count, 1)) < 0.14:
            count += 1
    return count


def _should_rollback_short_line_degradation(
    original_lines: List[models.Line],
    aligned_lines: List[models.Line],
) -> Tuple[bool, int, int]:
    """Detect when Whisper introduces widespread short-line compression artifacts."""
    before = _implausibly_short_multiword_count(original_lines)
    after = _implausibly_short_multiword_count(aligned_lines)
    added = after - before
    min_added = max(3, int(0.06 * max(len(aligned_lines), 1)))
    should_rollback = added >= min_added and after >= max(4, before * 2)
    return should_rollback, before, after


def _constrain_line_starts_to_baseline(
    mapped_lines: List[models.Line],
    baseline_lines: List[models.Line],
    *,
    min_gap: float = 0.01,
) -> List[models.Line]:
    """Force mapped line starts to baseline (LRC) starts while preserving within-line shape."""
    constrained: List[models.Line] = []
    for idx, line in enumerate(mapped_lines):
        if idx >= len(baseline_lines) or not line.words:
            constrained.append(line)
            continue

        baseline = baseline_lines[idx]
        if not baseline.words:
            constrained.append(line)
            continue

        target_start = baseline.start_time
        shift = target_start - line.start_time
        shifted_words = [
            models.Word(
                text=w.text,
                start_time=w.start_time + shift,
                end_time=w.end_time + shift,
                singer=w.singer,
            )
            for w in line.words
        ]
        shifted_line = models.Line(words=shifted_words, singer=line.singer)

        next_baseline_start = None
        for nxt in baseline_lines[idx + 1 :]:
            if nxt.words:
                next_baseline_start = nxt.start_time
                break

        if next_baseline_start is not None and shifted_line.end_time > (
            next_baseline_start - min_gap
        ):
            available = max(0.1, (next_baseline_start - min_gap) - target_start)
            current = max(0.1, shifted_line.end_time - target_start)
            scale = min(1.0, available / current)
            compressed_words = []
            for w in shifted_line.words:
                ws = target_start + (w.start_time - target_start) * scale
                we = target_start + (w.end_time - target_start) * scale
                if we < ws:
                    we = ws
                compressed_words.append(
                    models.Word(
                        text=w.text,
                        start_time=ws,
                        end_time=we,
                        singer=w.singer,
                    )
                )
            shifted_line = models.Line(words=compressed_words, singer=line.singer)

        constrained.append(shifted_line)

    return constrained


def _filter_low_confidence_whisper_words(
    words: List[timing_models.TranscriptionWord],
    threshold: float,
    *,
    min_keep_ratio: float = 0.6,
    min_keep_words: int = 20,
) -> List[timing_models.TranscriptionWord]:
    return _filter_low_confidence_whisper_words_impl(
        words,
        threshold,
        min_keep_ratio=min_keep_ratio,
        min_keep_words=min_keep_words,
    )


def _enforce_mapped_line_stage_invariants(
    lines_in: List[models.Line],
    all_words: List[timing_models.TranscriptionWord],
    *,
    enforce_monotonic_line_starts_whisper_fn: Callable[..., Any],
    resolve_line_overlaps_fn: Callable[..., Any],
) -> List[models.Line]:
    """Settle monotonic/overlap constraints across dense mapped lyric sections."""
    out = lines_in
    # Run twice to settle chained neighbor constraints in repeated sections.
    for _ in range(2):
        out = enforce_monotonic_line_starts_whisper_fn(out, all_words)
        out = resolve_line_overlaps_fn(out)
    return out


def _run_mapped_line_postpasses(
    *,
    mapped_lines: List[models.Line],
    mapped_lines_set: set[int],
    all_words: List[timing_models.TranscriptionWord],
    transcription: List[timing_models.TranscriptionSegment],
    audio_features: Optional[timing_models.AudioFeatures],
    vocals_path: str,
    epitran_lang: str,
    corrections: List[str],
    interpolate_unmatched_lines_fn: Callable[..., Any],
    refine_unmatched_lines_with_onsets_fn: Callable[..., Any],
    shift_repeated_lines_to_next_whisper_fn: Callable[..., Any],
    extend_line_to_trailing_whisper_matches_fn: Callable[..., Any],
    pull_late_lines_to_matching_segments_fn: Callable[..., Any],
    retime_short_interjection_lines_fn: Callable[..., Any],
    snap_first_word_to_whisper_onset_fn: Callable[..., Any],
    pull_lines_forward_for_continuous_vocals_fn: Callable[..., Any],
    enforce_monotonic_line_starts_whisper_fn: Callable[..., Any],
    resolve_line_overlaps_fn: Callable[..., Any],
) -> Tuple[List[models.Line], List[str]]:
    mapped_lines = interpolate_unmatched_lines_fn(mapped_lines, mapped_lines_set)
    mapped_lines = _enforce_mapped_line_stage_invariants(
        mapped_lines,
        all_words,
        enforce_monotonic_line_starts_whisper_fn=enforce_monotonic_line_starts_whisper_fn,
        resolve_line_overlaps_fn=resolve_line_overlaps_fn,
    )

    mapped_lines = refine_unmatched_lines_with_onsets_fn(
        mapped_lines,
        mapped_lines_set,
        vocals_path,
    )
    mapped_lines = _enforce_mapped_line_stage_invariants(
        mapped_lines,
        all_words,
        enforce_monotonic_line_starts_whisper_fn=enforce_monotonic_line_starts_whisper_fn,
        resolve_line_overlaps_fn=resolve_line_overlaps_fn,
    )

    mapped_lines = shift_repeated_lines_to_next_whisper_fn(mapped_lines, all_words)
    mapped_lines = _enforce_mapped_line_stage_invariants(
        mapped_lines,
        all_words,
        enforce_monotonic_line_starts_whisper_fn=enforce_monotonic_line_starts_whisper_fn,
        resolve_line_overlaps_fn=resolve_line_overlaps_fn,
    )
    mapped_lines = extend_line_to_trailing_whisper_matches_fn(
        mapped_lines,
        all_words,
    )
    mapped_lines = _enforce_mapped_line_stage_invariants(
        mapped_lines,
        all_words,
        enforce_monotonic_line_starts_whisper_fn=enforce_monotonic_line_starts_whisper_fn,
        resolve_line_overlaps_fn=resolve_line_overlaps_fn,
    )
    mapped_lines = pull_late_lines_to_matching_segments_fn(
        mapped_lines,
        transcription,
        epitran_lang,
    )
    mapped_lines = _enforce_mapped_line_stage_invariants(
        mapped_lines,
        all_words,
        enforce_monotonic_line_starts_whisper_fn=enforce_monotonic_line_starts_whisper_fn,
        resolve_line_overlaps_fn=resolve_line_overlaps_fn,
    )
    mapped_lines = retime_short_interjection_lines_fn(
        mapped_lines,
        transcription,
    )
    mapped_lines = _enforce_mapped_line_stage_invariants(
        mapped_lines,
        all_words,
        enforce_monotonic_line_starts_whisper_fn=enforce_monotonic_line_starts_whisper_fn,
        resolve_line_overlaps_fn=resolve_line_overlaps_fn,
    )
    mapped_lines = snap_first_word_to_whisper_onset_fn(
        mapped_lines,
        all_words,
    )
    mapped_lines = _enforce_mapped_line_stage_invariants(
        mapped_lines,
        all_words,
        enforce_monotonic_line_starts_whisper_fn=enforce_monotonic_line_starts_whisper_fn,
        resolve_line_overlaps_fn=resolve_line_overlaps_fn,
    )
    if audio_features is not None:
        mapped_lines, continuous_fixes = pull_lines_forward_for_continuous_vocals_fn(
            mapped_lines,
            audio_features,
        )
        if continuous_fixes:
            corrections.append(
                f"Pulled {continuous_fixes} line(s) forward for continuous vocals"
            )
    mapped_lines = _enforce_mapped_line_stage_invariants(
        mapped_lines,
        all_words,
        enforce_monotonic_line_starts_whisper_fn=enforce_monotonic_line_starts_whisper_fn,
        resolve_line_overlaps_fn=resolve_line_overlaps_fn,
    )
    mapped_lines = pull_late_lines_to_matching_segments_fn(
        mapped_lines,
        transcription,
        epitran_lang,
    )
    mapped_lines = _enforce_mapped_line_stage_invariants(
        mapped_lines,
        all_words,
        enforce_monotonic_line_starts_whisper_fn=enforce_monotonic_line_starts_whisper_fn,
        resolve_line_overlaps_fn=resolve_line_overlaps_fn,
    )
    try:
        mapped_lines = snap_first_word_to_whisper_onset_fn(
            mapped_lines,
            all_words,
            max_shift=2.5,
        )
    except TypeError:
        mapped_lines = snap_first_word_to_whisper_onset_fn(mapped_lines, all_words)
    mapped_lines = _enforce_mapped_line_stage_invariants(
        mapped_lines,
        all_words,
        enforce_monotonic_line_starts_whisper_fn=enforce_monotonic_line_starts_whisper_fn,
        resolve_line_overlaps_fn=resolve_line_overlaps_fn,
    )
    if audio_features is not None:
        mapped_lines, late_audio_fixes = pull_lines_forward_for_continuous_vocals_fn(
            mapped_lines,
            audio_features,
        )
        if late_audio_fixes:
            corrections.append(
                f"Applied {late_audio_fixes} late audio onset/silence adjustment(s)"
            )
        mapped_lines = _enforce_mapped_line_stage_invariants(
            mapped_lines,
            all_words,
            enforce_monotonic_line_starts_whisper_fn=enforce_monotonic_line_starts_whisper_fn,
            resolve_line_overlaps_fn=resolve_line_overlaps_fn,
        )

    return mapped_lines, corrections


def transcribe_vocals_impl(
    vocals_path: str,
    language: Optional[str],
    model_size: str,
    aggressive: bool,
    temperature: float,
    *,
    get_whisper_cache_path_fn: Callable[..., Optional[str]],
    find_best_cached_whisper_model_fn: Callable[..., Optional[Tuple[str, str]]],
    load_whisper_cache_fn: Callable[..., Optional[Tuple[Any, Any, str]]],
    save_whisper_cache_fn: Callable[..., None],
    load_whisper_model_class_fn: Callable[[], Any],
    logger,
) -> Tuple[
    List[timing_models.TranscriptionSegment],
    List[timing_models.TranscriptionWord],
    str,
    str,
]:
    """Transcribe vocals with disk cache and return timing models."""
    cache_path = get_whisper_cache_path_fn(
        vocals_path, model_size, language, aggressive, temperature
    )
    cached_model = model_size
    if cache_path:
        best_cached = find_best_cached_whisper_model_fn(
            vocals_path, language, aggressive, model_size, temperature
        )
        if best_cached:
            cache_path, cached_model = best_cached
        cached = load_whisper_cache_fn(cache_path)
        if cached:
            segments, all_words, detected_lang = cached
            return segments, all_words, detected_lang, cached_model

    try:
        whisper_model_class = load_whisper_model_class_fn()
    except ImportError:
        logger.warning("faster-whisper not installed, cannot transcribe")
        return [], [], "", model_size

    try:
        logger.info(f"Loading Whisper model ({model_size})...")
        model = whisper_model_class(model_size, device="cpu", compute_type="int8")

        logger.info(f"Transcribing vocals{f' in {language}' if language else ''}...")
        transcribe_kwargs: Dict[str, object] = {
            "language": language,
            "word_timestamps": True,
            "vad_filter": True,
            "temperature": temperature,
        }
        if aggressive:
            transcribe_kwargs.update(
                {
                    "vad_filter": False,
                    "no_speech_threshold": 1.0,
                    "log_prob_threshold": -2.0,
                }
            )
        segments, info = model.transcribe(vocals_path, **transcribe_kwargs)

        result = []
        all_words = []
        for seg in segments:
            seg_words = []
            if seg.words:
                for w in seg.words:
                    tw = timing_models.TranscriptionWord(
                        start=w.start,
                        end=w.end,
                        text=w.word.strip(),
                        probability=w.probability,
                    )
                    seg_words.append(tw)
                    all_words.append(tw)
            result.append(
                timing_models.TranscriptionSegment(
                    start=seg.start,
                    end=seg.end,
                    text=seg.text.strip(),
                    words=seg_words,
                )
            )

        detected_lang = info.language
        logger.info(
            f"Transcribed {len(result)} segments, {len(all_words)} words (language: {detected_lang})"
        )

        if cache_path:
            save_whisper_cache_fn(
                cache_path,
                result,
                all_words,
                detected_lang,
                model_size,
                aggressive,
                temperature,
            )

        return result, all_words, detected_lang, model_size

    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return [], [], "", model_size


def align_lrc_text_to_whisper_timings_impl(  # noqa: C901
    lines: List[models.Line],
    vocals_path: str,
    language: Optional[str],
    model_size: str,
    aggressive: bool,
    temperature: float,
    min_similarity: float,
    audio_features: Optional[timing_models.AudioFeatures],
    lenient_vocal_activity_threshold: float,
    lenient_activity_bonus: float,
    low_word_confidence_threshold: float,
    *,
    transcribe_vocals_fn: Callable[..., Tuple[Any, Any, str, str]],
    extract_audio_features_fn: Callable[..., Optional[timing_models.AudioFeatures]],
    dedupe_whisper_segments_fn: Callable[..., Any],
    trim_whisper_transcription_by_lyrics_fn: Callable[..., Any],
    fill_vocal_activity_gaps_fn: Callable[..., Any],
    dedupe_whisper_words_fn: Callable[..., Any],
    extract_lrc_words_all_fn: Callable[..., Any],
    build_phoneme_tokens_from_lrc_words_fn: Callable[..., Any],
    build_phoneme_tokens_from_whisper_words_fn: Callable[..., Any],
    build_syllable_tokens_from_phonemes_fn: Callable[..., Any],
    build_segment_text_overlap_assignments_fn: Callable[..., Any],
    build_phoneme_dtw_path_fn: Callable[..., Any],
    build_word_assignments_from_phoneme_path_fn: Callable[..., Any],
    build_block_segmented_syllable_assignments_fn: Callable[..., Any],
    map_lrc_words_to_whisper_fn: Callable[..., Any],
    shift_repeated_lines_to_next_whisper_fn: Callable[..., Any],
    enforce_monotonic_line_starts_whisper_fn: Callable[..., Any],
    resolve_line_overlaps_fn: Callable[..., Any],
    extend_line_to_trailing_whisper_matches_fn: Callable[..., Any],
    pull_late_lines_to_matching_segments_fn: Callable[..., Any],
    retime_short_interjection_lines_fn: Callable[..., Any],
    snap_first_word_to_whisper_onset_fn: Callable[..., Any],
    interpolate_unmatched_lines_fn: Callable[..., Any],
    refine_unmatched_lines_with_onsets_fn: Callable[..., Any],
    pull_lines_forward_for_continuous_vocals_fn: Callable[..., Any],
    logger,
) -> Tuple[List[models.Line], List[str], Dict[str, float]]:
    """Align LRC text to Whisper timings using phonetic DTW (timings fixed)."""
    _ = min_similarity
    _ = lenient_activity_bonus
    baseline_lines = _clone_lines_for_fallback(lines)

    ensure_local_lex_lookup()
    transcription, all_words, detected_lang, used_model = transcribe_vocals_fn(
        vocals_path, language, model_size, aggressive, temperature
    )
    if not audio_features:
        audio_features = extract_audio_features_fn(vocals_path)
    transcription = dedupe_whisper_segments_fn(transcription)
    transcription = dedupe_whisper_segments_fn(transcription)
    line_texts = [line.text for line in lines if line.text.strip()]
    transcription, all_words, trimmed_end = trim_whisper_transcription_by_lyrics_fn(
        transcription, all_words, line_texts
    )
    if trimmed_end:
        logger.info(
            "Truncated Whisper transcript to %.2f s (last matching lyric).", trimmed_end
        )

    if not transcription or not all_words:
        logger.warning("No transcription available, skipping Whisper timing map")
        return lines, [], {}

    if audio_features:
        all_words, filled_segments = fill_vocal_activity_gaps_fn(
            all_words,
            audio_features,
            lenient_vocal_activity_threshold,
            segments=transcription,
        )
        if filled_segments is not None:
            transcription = filled_segments

    before_low_conf_filter = len(all_words)
    all_words = _filter_low_confidence_whisper_words(
        all_words,
        low_word_confidence_threshold,
    )
    if len(all_words) != before_low_conf_filter:
        logger.debug(
            "Filtered low-confidence Whisper words: %d -> %d (threshold=%.2f)",
            before_low_conf_filter,
            len(all_words),
            low_word_confidence_threshold,
        )

    all_words = dedupe_whisper_words_fn(all_words)

    epitran_lang = phonetic_utils._whisper_lang_to_epitran(detected_lang)
    logger.debug(
        f"Using epitran language: {epitran_lang} (from Whisper: {detected_lang})"
    )

    lrc_words = extract_lrc_words_all_fn(lines)
    if not lrc_words:
        return lines, [], {}

    logger.debug(
        f"DTW-phonetic: Pre-computing IPA for {len(all_words)} Whisper words..."
    )
    for ww in all_words:
        phonetic_utils._get_ipa(ww.text, epitran_lang)
    for lw in lrc_words:
        phonetic_utils._get_ipa(lw["text"], epitran_lang)

    logger.debug(
        f"DTW-phonetic: Preparing phoneme sequences for {len(lrc_words)} lyrics "
        f"words and {len(all_words)} Whisper words..."
    )
    lrc_phonemes = build_phoneme_tokens_from_lrc_words_fn(lrc_words, epitran_lang)
    whisper_phonemes = build_phoneme_tokens_from_whisper_words_fn(
        all_words, epitran_lang
    )

    lrc_syllables = build_syllable_tokens_from_phonemes_fn(lrc_phonemes)
    whisper_syllables = build_syllable_tokens_from_phonemes_fn(whisper_phonemes)

    lrc_assignments = build_segment_text_overlap_assignments_fn(
        lrc_words,
        all_words,
        transcription,
    )
    seg_coverage = len(lrc_assignments) / len(lrc_words) if lrc_words else 0
    if seg_coverage < _MIN_SEGMENT_OVERLAP_COVERAGE:
        logger.debug(
            "Segment overlap coverage %.0f%% below %.0f%% threshold, falling back to DTW",
            seg_coverage * 100,
            _MIN_SEGMENT_OVERLAP_COVERAGE * 100,
        )
        if not lrc_syllables or not whisper_syllables:
            if not lrc_phonemes or not whisper_phonemes:
                logger.warning("No phoneme/syllable data; skipping mapping")
                return lines, [], {}
            path = build_phoneme_dtw_path_fn(
                lrc_phonemes,
                whisper_phonemes,
                epitran_lang,
            )
            lrc_assignments = build_word_assignments_from_phoneme_path_fn(
                path, lrc_phonemes, whisper_phonemes
            )
        else:
            lrc_assignments = build_block_segmented_syllable_assignments_fn(
                lrc_words,
                all_words,
                lrc_syllables,
                whisper_syllables,
                epitran_lang,
            )

    corrections: List[str] = []
    mapped_lines, mapped_count, total_similarity, mapped_lines_set = (
        map_lrc_words_to_whisper_fn(
            lines,
            lrc_words,
            all_words,
            lrc_assignments,
            epitran_lang,
            transcription,
        )
    )
    mapped_lines, corrections = _run_mapped_line_postpasses(
        mapped_lines=mapped_lines,
        mapped_lines_set=mapped_lines_set,
        all_words=all_words,
        transcription=transcription,
        audio_features=audio_features,
        vocals_path=vocals_path,
        epitran_lang=epitran_lang,
        corrections=corrections,
        interpolate_unmatched_lines_fn=interpolate_unmatched_lines_fn,
        refine_unmatched_lines_with_onsets_fn=refine_unmatched_lines_with_onsets_fn,
        shift_repeated_lines_to_next_whisper_fn=shift_repeated_lines_to_next_whisper_fn,
        extend_line_to_trailing_whisper_matches_fn=extend_line_to_trailing_whisper_matches_fn,
        pull_late_lines_to_matching_segments_fn=pull_late_lines_to_matching_segments_fn,
        retime_short_interjection_lines_fn=retime_short_interjection_lines_fn,
        snap_first_word_to_whisper_onset_fn=snap_first_word_to_whisper_onset_fn,
        pull_lines_forward_for_continuous_vocals_fn=pull_lines_forward_for_continuous_vocals_fn,
        enforce_monotonic_line_starts_whisper_fn=enforce_monotonic_line_starts_whisper_fn,
        resolve_line_overlaps_fn=resolve_line_overlaps_fn,
    )

    # Policy constraint: keep line starts anchored to LRC timings.
    mapped_lines = _constrain_line_starts_to_baseline(mapped_lines, baseline_lines)

    matched_ratio = mapped_count / len(lrc_words) if lrc_words else 0.0
    avg_similarity = total_similarity / mapped_count if mapped_count else 0.0
    line_coverage = (
        len(mapped_lines_set) / sum(1 for line in lines if line.words) if lines else 0.0
    )

    metrics: Dict[str, Any] = {
        "matched_ratio": matched_ratio,
        "word_coverage": matched_ratio,
        "avg_similarity": avg_similarity,
        "line_coverage": line_coverage,
        "phonetic_similarity_coverage": matched_ratio * avg_similarity,
        "high_similarity_ratio": avg_similarity,
        "exact_match_ratio": 0.0,
        "unmatched_ratio": 1.0 - matched_ratio,
        "dtw_used": 1.0,
        "dtw_mode": 1.0,
        "whisper_model": used_model,
    }

    if mapped_count:
        corrections.append(f"DTW-phonetic mapped {mapped_count} word(s) to Whisper")
    rollback, short_before, short_after = _should_rollback_short_line_degradation(
        baseline_lines, mapped_lines
    )
    if rollback:
        logger.warning(
            "Rolling back Whisper map: implausibly short multi-word lines worsened (%d -> %d)",
            short_before,
            short_after,
        )
        corrections.append(
            "Ignored Whisper timing map due to short-line compression artifacts"
        )
        return baseline_lines, corrections, metrics
    return mapped_lines, corrections, metrics


def correct_timing_with_whisper_impl(  # noqa: C901
    lines: List[models.Line],
    vocals_path: str,
    language: Optional[str],
    model_size: str,
    aggressive: bool,
    temperature: float,
    trust_lrc_threshold: float,
    correct_lrc_threshold: float,
    force_dtw: bool,
    audio_features: Optional[timing_models.AudioFeatures],
    lenient_vocal_activity_threshold: float,
    lenient_activity_bonus: float,
    low_word_confidence_threshold: float,
    *,
    transcribe_vocals_fn: Callable[..., Tuple[Any, Any, str, str]],
    extract_audio_features_fn: Callable[..., Optional[timing_models.AudioFeatures]],
    trim_whisper_transcription_by_lyrics_fn: Callable[..., Any],
    fill_vocal_activity_gaps_fn: Callable[..., Any],
    assess_lrc_quality_fn: Callable[..., Any],
    align_hybrid_lrc_whisper_fn: Callable[..., Any],
    align_dtw_whisper_with_data_fn: Callable[..., Any],
    retime_lines_from_dtw_alignments_fn: Callable[..., Any],
    merge_first_two_lines_if_segment_matches_fn: Callable[..., Any],
    retime_adjacent_lines_to_whisper_window_fn: Callable[..., Any],
    retime_adjacent_lines_to_segment_window_fn: Callable[..., Any],
    pull_next_line_into_segment_window_fn: Callable[..., Any],
    pull_lines_near_segment_end_fn: Callable[..., Any],
    pull_next_line_into_same_segment_fn: Callable[..., Any],
    merge_lines_to_whisper_segments_fn: Callable[..., Any],
    tighten_lines_to_whisper_segments_fn: Callable[..., Any],
    pull_lines_to_best_segments_fn: Callable[..., Any],
    fix_ordering_violations_fn: Callable[..., Any],
    normalize_line_word_timings_fn: Callable[..., Any],
    enforce_monotonic_line_starts_fn: Callable[..., Any],
    enforce_non_overlapping_lines_fn: Callable[..., Any],
    merge_short_following_line_into_segment_fn: Callable[..., Any],
    clamp_repeated_line_duration_fn: Callable[..., Any],
    drop_duplicate_lines_fn: Callable[..., Any],
    drop_duplicate_lines_by_timing_fn: Callable[..., Any],
    pull_lines_forward_for_continuous_vocals_fn: Callable[..., Any],
    logger,
) -> Tuple[List[models.Line], List[str], Dict[str, float]]:
    """Correct lyric timing by combining quality gates and Whisper alignments."""
    baseline_lines = _clone_lines_for_fallback(lines)
    transcription, all_words, detected_lang, _model = transcribe_vocals_fn(
        vocals_path, language, model_size, aggressive, temperature
    )
    if not audio_features:
        audio_features = extract_audio_features_fn(vocals_path)

    line_texts = [line.text for line in lines if line.text.strip()]
    transcription, all_words, trimmed_end = trim_whisper_transcription_by_lyrics_fn(
        transcription, all_words, line_texts
    )
    if trimmed_end:
        logger.info(
            "Truncated Whisper transcript to %.2f s (last matching lyric).", trimmed_end
        )

    if not transcription:
        logger.warning("No transcription available, skipping Whisper alignment")
        return lines, [], {}

    if audio_features:
        all_words, filled_segments = fill_vocal_activity_gaps_fn(
            all_words,
            audio_features,
            lenient_vocal_activity_threshold,
            segments=transcription,
        )
        if filled_segments is not None:
            transcription = filled_segments

    epitran_lang = phonetic_utils._whisper_lang_to_epitran(detected_lang)
    logger.debug(
        f"Using epitran language: {epitran_lang} (from Whisper: {detected_lang})"
    )

    logger.debug(f"Pre-computing IPA for {len(all_words)} Whisper words...")
    for w in all_words:
        phonetic_utils._get_ipa(w.text, epitran_lang)

    quality, assessments = assess_lrc_quality_fn(
        lines, all_words, epitran_lang, tolerance=1.5
    )
    logger.info(f"LRC timing quality: {quality:.0%} of lines within 1.5s of Whisper")
    _ = assessments

    metrics: Dict[str, float] = {}
    if not force_dtw and quality >= 0.7:
        logger.info("LRC timing is good, using targeted corrections only")
        aligned_lines, alignments = align_hybrid_lrc_whisper_fn(
            lines,
            transcription,
            all_words,
            language=epitran_lang,
            trust_threshold=trust_lrc_threshold,
            correct_threshold=correct_lrc_threshold,
        )
    elif not force_dtw and quality >= 0.4:
        logger.info("LRC timing is mixed, using hybrid Whisper alignment")
        aligned_lines, alignments = align_hybrid_lrc_whisper_fn(
            lines,
            transcription,
            all_words,
            language=epitran_lang,
            trust_threshold=trust_lrc_threshold,
            correct_threshold=correct_lrc_threshold,
        )
    else:
        logger.info("LRC timing is poor, using DTW global alignment")
        aligned_lines, alignments, metrics, lrc_words, alignments_map = (
            align_dtw_whisper_with_data_fn(
                lines,
                all_words,
                language=epitran_lang,
                silence_regions=(
                    audio_features.silence_regions if audio_features else None
                ),
                audio_features=audio_features,
                lenient_vocal_activity_threshold=lenient_vocal_activity_threshold,
                lenient_activity_bonus=lenient_activity_bonus,
                low_word_confidence_threshold=low_word_confidence_threshold,
            )
        )
        metrics["dtw_used"] = 1.0

        matched_ratio = metrics.get("matched_ratio", 0.0)
        avg_similarity = metrics.get("avg_similarity", 0.0)
        line_coverage = metrics.get("line_coverage", 0.0)
        confidence_ok = (
            matched_ratio >= 0.6 and avg_similarity >= 0.5 and line_coverage >= 0.6
        )

        if confidence_ok and lrc_words and alignments_map:
            dtw_lines, dtw_fixes = retime_lines_from_dtw_alignments_fn(
                lines, lrc_words, alignments_map
            )
            aligned_lines = dtw_lines
            alignments.extend(dtw_fixes)
            metrics["dtw_confidence_passed"] = 1.0
        else:
            metrics["dtw_confidence_passed"] = 0.0
            alignments.append(
                "DTW confidence gating failed; keeping conservative word shifts"
            )

    if quality < 0.4 or force_dtw:
        aligned_lines, alignments = _apply_low_quality_segment_postpasses(
            aligned_lines=aligned_lines,
            alignments=alignments,
            transcription=transcription,
            epitran_lang=epitran_lang,
            merge_first_two_lines_if_segment_matches_fn=merge_first_two_lines_if_segment_matches_fn,
            retime_adjacent_lines_to_whisper_window_fn=retime_adjacent_lines_to_whisper_window_fn,
            retime_adjacent_lines_to_segment_window_fn=retime_adjacent_lines_to_segment_window_fn,
            pull_next_line_into_segment_window_fn=pull_next_line_into_segment_window_fn,
            pull_lines_near_segment_end_fn=pull_lines_near_segment_end_fn,
            pull_next_line_into_same_segment_fn=pull_next_line_into_same_segment_fn,
            merge_lines_to_whisper_segments_fn=merge_lines_to_whisper_segments_fn,
            tighten_lines_to_whisper_segments_fn=tighten_lines_to_whisper_segments_fn,
            pull_lines_to_best_segments_fn=pull_lines_to_best_segments_fn,
        )

    aligned_lines, alignments = _finalize_whisper_line_set(
        source_lines=lines,
        aligned_lines=aligned_lines,
        alignments=alignments,
        transcription=transcription,
        epitran_lang=epitran_lang,
        force_dtw=force_dtw,
        audio_features=audio_features,
        fix_ordering_violations_fn=fix_ordering_violations_fn,
        normalize_line_word_timings_fn=normalize_line_word_timings_fn,
        enforce_monotonic_line_starts_fn=enforce_monotonic_line_starts_fn,
        enforce_non_overlapping_lines_fn=enforce_non_overlapping_lines_fn,
        pull_lines_near_segment_end_fn=pull_lines_near_segment_end_fn,
        merge_short_following_line_into_segment_fn=merge_short_following_line_into_segment_fn,
        clamp_repeated_line_duration_fn=clamp_repeated_line_duration_fn,
        drop_duplicate_lines_fn=drop_duplicate_lines_fn,
        drop_duplicate_lines_by_timing_fn=drop_duplicate_lines_by_timing_fn,
        pull_lines_forward_for_continuous_vocals_fn=pull_lines_forward_for_continuous_vocals_fn,
    )

    # Policy constraint: keep line starts anchored to LRC timings.
    aligned_lines = _constrain_line_starts_to_baseline(aligned_lines, baseline_lines)

    rollback, short_before, short_after = _should_rollback_short_line_degradation(
        baseline_lines, aligned_lines
    )
    if rollback:
        logger.warning(
            "Rolling back Whisper corrections: implausibly short multi-word lines worsened (%d -> %d)",
            short_before,
            short_after,
        )
        alignments.append(
            "Rolled back Whisper timing due to short-line compression artifacts"
        )
        aligned_lines = baseline_lines

    if alignments:
        logger.info(f"Whisper hybrid alignment: {len(alignments)} lines corrected")

    return aligned_lines, alignments, metrics


def _apply_low_quality_segment_postpasses(
    *,
    aligned_lines: List[models.Line],
    alignments: List[str],
    transcription: List[timing_models.TranscriptionSegment],
    epitran_lang: str,
    merge_first_two_lines_if_segment_matches_fn: Callable[..., Any],
    retime_adjacent_lines_to_whisper_window_fn: Callable[..., Any],
    retime_adjacent_lines_to_segment_window_fn: Callable[..., Any],
    pull_next_line_into_segment_window_fn: Callable[..., Any],
    pull_lines_near_segment_end_fn: Callable[..., Any],
    pull_next_line_into_same_segment_fn: Callable[..., Any],
    merge_lines_to_whisper_segments_fn: Callable[..., Any],
    tighten_lines_to_whisper_segments_fn: Callable[..., Any],
    pull_lines_to_best_segments_fn: Callable[..., Any],
) -> Tuple[List[models.Line], List[str]]:
    aligned_lines, merged_first = merge_first_two_lines_if_segment_matches_fn(
        aligned_lines, transcription, epitran_lang
    )
    if merged_first:
        alignments.append("Merged first two lines via Whisper segment")
    aligned_lines, pair_retimed = retime_adjacent_lines_to_whisper_window_fn(
        aligned_lines, transcription, epitran_lang
    )
    if pair_retimed:
        alignments.append(
            f"Retimed {pair_retimed} adjacent line pair(s) to Whisper window"
        )
    aligned_lines, pair_windowed = retime_adjacent_lines_to_segment_window_fn(
        aligned_lines, transcription, epitran_lang
    )
    if pair_windowed:
        alignments.append(
            f"Retimed {pair_windowed} adjacent line pair(s) to Whisper segment window"
        )
    aligned_lines, pulled_next = pull_next_line_into_segment_window_fn(
        aligned_lines, transcription, epitran_lang
    )
    if pulled_next:
        alignments.append(f"Pulled {pulled_next} line(s) into adjacent segment window")
    aligned_lines, pulled_near_end = pull_lines_near_segment_end_fn(
        aligned_lines, transcription, epitran_lang
    )
    if pulled_near_end:
        alignments.append(f"Pulled {pulled_near_end} line(s) near segment ends")
    aligned_lines, pulled_same = pull_next_line_into_same_segment_fn(
        aligned_lines, transcription
    )
    if pulled_same:
        alignments.append(f"Pulled {pulled_same} line(s) into same segment")
    aligned_lines, pair_retimed_after = retime_adjacent_lines_to_whisper_window_fn(
        aligned_lines,
        transcription,
        epitran_lang,
        max_window_duration=4.5,
        max_start_offset=1.0,
    )
    if pair_retimed_after:
        alignments.append(
            f"Retimed {pair_retimed_after} adjacent line pair(s) after pulls"
        )
    aligned_lines, merged = merge_lines_to_whisper_segments_fn(
        aligned_lines, transcription, epitran_lang
    )
    if merged:
        alignments.append(f"Merged {merged} line pair(s) via Whisper segments")
    aligned_lines, tightened = tighten_lines_to_whisper_segments_fn(
        aligned_lines, transcription, epitran_lang
    )
    if tightened:
        alignments.append(f"Tightened {tightened} line(s) to Whisper segments")
    aligned_lines, pulled = pull_lines_to_best_segments_fn(
        aligned_lines, transcription, epitran_lang
    )
    if pulled:
        alignments.append(f"Pulled {pulled} line(s) to Whisper segments")
    return aligned_lines, alignments


def _finalize_whisper_line_set(
    *,
    source_lines: List[models.Line],
    aligned_lines: List[models.Line],
    alignments: List[str],
    transcription: List[timing_models.TranscriptionSegment],
    epitran_lang: str,
    force_dtw: bool,
    audio_features: Optional[timing_models.AudioFeatures],
    fix_ordering_violations_fn: Callable[..., Any],
    normalize_line_word_timings_fn: Callable[..., Any],
    enforce_monotonic_line_starts_fn: Callable[..., Any],
    enforce_non_overlapping_lines_fn: Callable[..., Any],
    pull_lines_near_segment_end_fn: Callable[..., Any],
    merge_short_following_line_into_segment_fn: Callable[..., Any],
    clamp_repeated_line_duration_fn: Callable[..., Any],
    drop_duplicate_lines_fn: Callable[..., Any],
    drop_duplicate_lines_by_timing_fn: Callable[..., Any],
    pull_lines_forward_for_continuous_vocals_fn: Callable[..., Any],
) -> Tuple[List[models.Line], List[str]]:
    aligned_lines, alignments = fix_ordering_violations_fn(
        source_lines, aligned_lines, alignments
    )
    aligned_lines = normalize_line_word_timings_fn(aligned_lines)
    aligned_lines = enforce_monotonic_line_starts_fn(aligned_lines)
    aligned_lines = enforce_non_overlapping_lines_fn(aligned_lines)

    if force_dtw:
        aligned_lines, pulled_near_end = pull_lines_near_segment_end_fn(
            aligned_lines, transcription, epitran_lang
        )
        if pulled_near_end:
            alignments.append(
                f"Pulled {pulled_near_end} line(s) near segment ends (post-order)"
            )
        aligned_lines, merged_short = merge_short_following_line_into_segment_fn(
            aligned_lines, transcription
        )
        if merged_short:
            alignments.append(
                f"Merged {merged_short} short line(s) into prior segments"
            )
        aligned_lines, clamped_repeat = clamp_repeated_line_duration_fn(aligned_lines)
        if clamped_repeat:
            alignments.append(f"Clamped {clamped_repeat} repeated line(s) duration")

    aligned_lines, deduped = drop_duplicate_lines_fn(
        aligned_lines, transcription, epitran_lang
    )
    if deduped:
        alignments.append(f"Dropped {deduped} duplicate line(s)")
    before_drop = len(aligned_lines)
    aligned_lines = [line for line in aligned_lines if line.words]
    if len(aligned_lines) != before_drop:
        alignments.append("Dropped empty lines after Whisper merges")
    aligned_lines, timing_deduped = drop_duplicate_lines_by_timing_fn(aligned_lines)
    if timing_deduped:
        alignments.append(
            f"Dropped {timing_deduped} duplicate line(s) by timing overlap"
        )

    if audio_features is not None:
        aligned_lines, continuous_fixes = pull_lines_forward_for_continuous_vocals_fn(
            aligned_lines, audio_features
        )
        if continuous_fixes:
            alignments.append(
                f"Pulled {continuous_fixes} line(s) forward for continuous vocals"
            )

    aligned_lines = enforce_monotonic_line_starts_fn(aligned_lines)
    aligned_lines = enforce_non_overlapping_lines_fn(aligned_lines)
    return aligned_lines, alignments
