"""DTW-based LRC-to-Whisper alignment orchestration for integration pipeline."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from ....utils.lex_lookup_installer import ensure_local_lex_lookup
from ... import models, phonetic_utils
from ..alignment import timing_models
from .whisper_forced_alignment import align_lines_with_whisperx


def _line_set_end(lines: List[models.Line]) -> float:
    end_time = 0.0
    for line in lines:
        if line.words:
            end_time = max(end_time, line.end_time)
    return end_time


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
    extract_lrc_words_all_fn: Callable[..., Any],
    build_phoneme_tokens_from_lrc_words_fn: Callable[..., Any],
    build_phoneme_tokens_from_whisper_words_fn: Callable[..., Any],
    build_syllable_tokens_from_phonemes_fn: Callable[..., Any],
    build_segment_text_overlap_assignments_fn: Callable[..., Any],
    build_phoneme_dtw_path_fn: Callable[..., Any],
    build_word_assignments_from_phoneme_path_fn: Callable[..., Any],
    build_block_segmented_syllable_assignments_fn: Callable[..., Any],
    map_lrc_words_to_whisper_fn: Callable[..., Any],
    dedupe_whisper_words_fn: Callable[..., Any],
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
    run_mapped_line_postpasses_fn: Callable[..., Any],
    constrain_line_starts_to_baseline_fn: Callable[..., Any],
    should_rollback_short_line_degradation_fn: Callable[..., Any],
    restore_implausibly_short_lines_fn: Callable[..., Any],
    clone_lines_for_fallback_fn: Callable[..., Any],
    filter_low_confidence_whisper_words_fn: Callable[..., Any],
    min_segment_overlap_coverage: float,
    logger,
) -> Tuple[List[models.Line], List[str], Dict[str, float]]:
    """Align LRC text to Whisper timings via DTW-style phonetic assignment."""
    _ = min_similarity  # reserved for potential future tuning hooks
    _ = lenient_activity_bonus  # consumed by downstream scoring in related paths

    baseline_lines = clone_lines_for_fallback_fn(lines)
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

    sparse_whisper_output = len(all_words) < 80 or len(transcription) <= 4
    if sparse_whisper_output:
        forced = align_lines_with_whisperx(lines, vocals_path, language, logger)
        if forced is not None:
            forced_lines, forced_metrics = forced
            rollback, short_before, short_after = (
                should_rollback_short_line_degradation_fn(baseline_lines, forced_lines)
            )
            if rollback:
                repaired_lines, restored_count = restore_implausibly_short_lines_fn(
                    baseline_lines, forced_lines
                )
                repaired_rollback, _, repaired_after = (
                    should_rollback_short_line_degradation_fn(
                        baseline_lines, repaired_lines
                    )
                )
                if restored_count > 0 and not repaired_rollback:
                    logger.info(
                        "Kept WhisperX forced alignment after restoring %d short baseline line(s) (%d -> %d)",
                        restored_count,
                        short_after,
                        repaired_after,
                    )
                    forced_lines = repaired_lines
                    rollback = False
            if not rollback:
                forced_payload: Dict[str, Any] = {
                    "matched_ratio": float(
                        forced_metrics.get("forced_word_coverage", 0.0)
                    ),
                    "word_coverage": float(
                        forced_metrics.get("forced_word_coverage", 0.0)
                    ),
                    "avg_similarity": 1.0,
                    "line_coverage": float(
                        forced_metrics.get("forced_line_coverage", 0.0)
                    ),
                    "phonetic_similarity_coverage": float(
                        forced_metrics.get("forced_word_coverage", 0.0)
                    ),
                    "high_similarity_ratio": 1.0,
                    "exact_match_ratio": 0.0,
                    "unmatched_ratio": 1.0
                    - float(forced_metrics.get("forced_word_coverage", 0.0)),
                    "dtw_used": 0.0,
                    "dtw_mode": 0.0,
                    "whisperx_forced": 1.0,
                    "whisper_model": used_model,
                }
                return (
                    forced_lines,
                    [
                        "Applied WhisperX transcript-constrained forced alignment due to sparse Whisper transcript"
                    ],
                    forced_payload,
                )
            logger.warning(
                "Discarded WhisperX forced alignment due to short-line degradation (%d -> %d)",
                short_before,
                short_after,
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
    all_words = filter_low_confidence_whisper_words_fn(
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
        "Using epitran language: %s (from Whisper: %s)", epitran_lang, detected_lang
    )

    lrc_words = extract_lrc_words_all_fn(lines)
    if not lrc_words:
        return lines, [], {}

    logger.debug(
        "DTW-phonetic: Pre-computing IPA for %d Whisper words...", len(all_words)
    )
    for ww in all_words:
        phonetic_utils._get_ipa(ww.text, epitran_lang)
    for lw in lrc_words:
        phonetic_utils._get_ipa(lw["text"], epitran_lang)

    logger.debug(
        "DTW-phonetic: Preparing phoneme sequences for %d lyrics words and %d Whisper words...",
        len(lrc_words),
        len(all_words),
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
    if seg_coverage < min_segment_overlap_coverage:
        logger.debug(
            "Segment overlap coverage %.0f%% below %.0f%% threshold, falling back to DTW",
            seg_coverage * 100,
            min_segment_overlap_coverage * 100,
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
    mapped_lines, corrections = run_mapped_line_postpasses_fn(
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

    matched_ratio = mapped_count / len(lrc_words) if lrc_words else 0.0
    avg_similarity = total_similarity / mapped_count if mapped_count else 0.0
    line_coverage = (
        len(mapped_lines_set) / sum(1 for line in lines if line.words) if lines else 0.0
    )

    whisper_end = max((w.end for w in all_words), default=0.0)
    baseline_end = _line_set_end(baseline_lines)
    mapped_end = _line_set_end(mapped_lines)
    baseline_timeline_ratio = baseline_end / whisper_end if whisper_end > 0.0 else 1.0
    mapped_timeline_ratio = mapped_end / whisper_end if whisper_end > 0.0 else 1.0
    mapped_lines = constrain_line_starts_to_baseline_fn(mapped_lines, baseline_lines)

    try:
        mapped_lines = snap_first_word_to_whisper_onset_fn(
            mapped_lines,
            all_words,
            max_shift=2.5,
        )
    except TypeError:
        mapped_lines = snap_first_word_to_whisper_onset_fn(mapped_lines, all_words)
    mapped_lines = constrain_line_starts_to_baseline_fn(mapped_lines, baseline_lines)

    metrics: Dict[str, Any] = {
        "matched_ratio": matched_ratio,
        "word_coverage": matched_ratio,
        "avg_similarity": avg_similarity,
        "line_coverage": line_coverage,
        "baseline_timeline_ratio": baseline_timeline_ratio,
        "mapped_timeline_ratio": mapped_timeline_ratio,
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
    rollback, short_before, short_after = should_rollback_short_line_degradation_fn(
        baseline_lines, mapped_lines
    )
    if rollback:
        repaired_lines, restored_count = restore_implausibly_short_lines_fn(
            baseline_lines, mapped_lines
        )
        repaired_rollback, _, repaired_after = (
            should_rollback_short_line_degradation_fn(baseline_lines, repaired_lines)
        )
        if restored_count > 0 and not repaired_rollback:
            logger.info(
                "Recovered Whisper map by restoring %d short baseline line(s) (%d -> %d)",
                restored_count,
                short_after,
                repaired_after,
            )
            corrections.append(
                f"Restored {restored_count} short compressed lines from baseline timing"
            )
            return repaired_lines, corrections, metrics
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
