"""Hybrid/DTW correction orchestration for Whisper timing integration."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

from ....utils.lex_lookup_installer import ensure_local_lex_lookup
from ... import models, phonetic_utils
from ..alignment import timing_models
from .whisper_forced_alignment import align_lines_with_whisperx

_MIN_FORCED_WORD_COVERAGE = 0.2
_MIN_FORCED_LINE_COVERAGE = 0.2


@dataclass(frozen=True)
class _WhisperCorrectionDecisionConfig:
    quality_good_threshold: float = 0.7
    quality_mixed_threshold: float = 0.4
    dtw_confidence_matched_ratio_min: float = 0.6
    dtw_confidence_avg_similarity_min: float = 0.5
    dtw_confidence_line_coverage_min: float = 0.6
    low_confidence_matched_ratio_max: float = 0.25
    low_confidence_line_coverage_max: float = 0.25
    low_confidence_avg_similarity_max: float = 0.4
    no_evidence_matched_ratio_max: float = 0.05
    no_evidence_line_coverage_max: float = 0.05
    no_evidence_avg_similarity_max: float = 0.1
    no_evidence_quality_max: float = 0.4


def _default_correction_config() -> _WhisperCorrectionDecisionConfig:
    profile = os.getenv("Y2K_WHISPER_PROFILE", "default").strip().lower()
    if profile == "safe":
        return _WhisperCorrectionDecisionConfig(
            quality_good_threshold=0.75,
            quality_mixed_threshold=0.45,
            dtw_confidence_matched_ratio_min=0.65,
            dtw_confidence_avg_similarity_min=0.55,
            dtw_confidence_line_coverage_min=0.65,
            low_confidence_matched_ratio_max=0.2,
            low_confidence_line_coverage_max=0.2,
            low_confidence_avg_similarity_max=0.35,
            no_evidence_matched_ratio_max=0.04,
            no_evidence_line_coverage_max=0.04,
            no_evidence_avg_similarity_max=0.08,
            no_evidence_quality_max=0.45,
        )
    if profile == "aggressive":
        return _WhisperCorrectionDecisionConfig(
            quality_good_threshold=0.65,
            quality_mixed_threshold=0.35,
            dtw_confidence_matched_ratio_min=0.55,
            dtw_confidence_avg_similarity_min=0.45,
            dtw_confidence_line_coverage_min=0.55,
            low_confidence_matched_ratio_max=0.3,
            low_confidence_line_coverage_max=0.3,
            low_confidence_avg_similarity_max=0.45,
            no_evidence_matched_ratio_max=0.06,
            no_evidence_line_coverage_max=0.06,
            no_evidence_avg_similarity_max=0.12,
            no_evidence_quality_max=0.35,
        )
    return _WhisperCorrectionDecisionConfig()


def _line_set_end(lines: List[models.Line]) -> float:
    end_time = 0.0
    for line in lines:
        if line.words:
            end_time = max(end_time, line.end_time)
    return end_time


def _call_apply_low_quality_postpasses_with_metrics(
    apply_low_quality_segment_postpasses_fn: Callable[..., Any],
    *,
    aligned_lines: List[models.Line],
    alignments: List[str],
    transcription: List[timing_models.TranscriptionSegment],
    epitran_lang: str,
    metrics: Dict[str, float],
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
    kwargs = dict(
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
        stage_metrics=metrics,
    )
    try:
        return apply_low_quality_segment_postpasses_fn(**kwargs)
    except TypeError:
        kwargs.pop("stage_metrics", None)
        return apply_low_quality_segment_postpasses_fn(**kwargs)


def _call_finalize_line_set_with_metrics(
    finalize_whisper_line_set_fn: Callable[..., Any],
    *,
    source_lines: List[models.Line],
    aligned_lines: List[models.Line],
    alignments: List[str],
    transcription: List[timing_models.TranscriptionSegment],
    epitran_lang: str,
    force_dtw: bool,
    audio_features: Optional[timing_models.AudioFeatures],
    metrics: Dict[str, float],
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
    kwargs = dict(
        source_lines=source_lines,
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
        stage_metrics=metrics,
    )
    try:
        return finalize_whisper_line_set_fn(**kwargs)
    except TypeError:
        kwargs.pop("stage_metrics", None)
        return finalize_whisper_line_set_fn(**kwargs)


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
    apply_low_quality_segment_postpasses_fn: Callable[..., Any],
    finalize_whisper_line_set_fn: Callable[..., Any],
    constrain_line_starts_to_baseline_fn: Callable[..., Any],
    should_rollback_short_line_degradation_fn: Callable[..., Any],
    restore_implausibly_short_lines_fn: Callable[..., Any],
    clone_lines_for_fallback_fn: Callable[..., Any],
    logger,
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
) -> Tuple[List[models.Line], List[str], Dict[str, float]]:
    """Correct lyric timing by combining quality gates and Whisper alignments."""
    config = _default_correction_config()
    baseline_lines = clone_lines_for_fallback_fn(lines)
    ensure_local_lex_lookup()
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

    sparse_whisper_output = len(all_words) < 80 or len(transcription) <= 4
    if sparse_whisper_output:
        forced = align_lines_with_whisperx(lines, vocals_path, language, logger)
        if forced is not None:
            aligned_lines, forced_metrics = forced
            forced_word_coverage = float(
                forced_metrics.get("forced_word_coverage", 0.0)
            )
            forced_line_coverage = float(
                forced_metrics.get("forced_line_coverage", 0.0)
            )
            if (
                forced_word_coverage < _MIN_FORCED_WORD_COVERAGE
                or forced_line_coverage < _MIN_FORCED_LINE_COVERAGE
            ):
                logger.warning(
                    (
                        "Discarded WhisperX forced alignment due to low forced coverage "
                        "(word=%.2f line=%.2f)"
                    ),
                    forced_word_coverage,
                    forced_line_coverage,
                )
                forced = None
            if forced is None:
                pass
            else:
                aligned_lines = constrain_line_starts_to_baseline_fn(
                    aligned_lines, baseline_lines
                )
                rollback, short_before, short_after = (
                    should_rollback_short_line_degradation_fn(
                        baseline_lines, aligned_lines
                    )
                )
                if rollback:
                    repaired_lines, restored_count = restore_implausibly_short_lines_fn(
                        baseline_lines, aligned_lines
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
                        aligned_lines = repaired_lines
                        rollback = False
                if not rollback:
                    return (
                        aligned_lines,
                        [
                            "Applied WhisperX transcript-constrained forced alignment due to sparse Whisper transcript"
                        ],
                        {
                            "matched_ratio": forced_word_coverage,
                            "avg_similarity": 1.0,
                            "line_coverage": forced_line_coverage,
                            "unmatched_ratio": 1.0 - forced_word_coverage,
                            "whisperx_forced": 1.0,
                        },
                    )
                logger.warning(
                    "Discarded WhisperX forced alignment due to short-line degradation (%d -> %d)",
                    short_before,
                    short_after,
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
        "Using epitran language: %s (from Whisper: %s)", epitran_lang, detected_lang
    )

    logger.debug("Pre-computing IPA for %d Whisper words...", len(all_words))
    for word in all_words:
        phonetic_utils._get_ipa(word.text, epitran_lang)

    quality, assessments = assess_lrc_quality_fn(
        lines, all_words, epitran_lang, tolerance=1.5
    )
    logger.info(
        "LRC timing quality: %.0f%% of lines within 1.5s of Whisper", quality * 100
    )
    _ = assessments

    metrics: Dict[str, float] = {}
    if not force_dtw and quality >= config.quality_good_threshold:
        logger.info("LRC timing is good, using targeted corrections only")
        aligned_lines, alignments = align_hybrid_lrc_whisper_fn(
            lines,
            transcription,
            all_words,
            language=epitran_lang,
            trust_threshold=trust_lrc_threshold,
            correct_threshold=correct_lrc_threshold,
        )
    elif not force_dtw and quality >= config.quality_mixed_threshold:
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
            matched_ratio >= config.dtw_confidence_matched_ratio_min
            and avg_similarity >= config.dtw_confidence_avg_similarity_min
            and line_coverage >= config.dtw_confidence_line_coverage_min
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

    low_confidence_alignment = False
    if quality < config.quality_mixed_threshold and not force_dtw:
        matched_ratio = float(metrics.get("matched_ratio", 0.0) or 0.0)
        line_coverage = float(metrics.get("line_coverage", 0.0) or 0.0)
        avg_similarity = float(metrics.get("avg_similarity", 0.0) or 0.0)
        low_confidence_alignment = (
            matched_ratio < config.low_confidence_matched_ratio_max
            and line_coverage < config.low_confidence_line_coverage_max
            and avg_similarity < config.low_confidence_avg_similarity_max
        )
        if low_confidence_alignment:
            alignments.append(
                "Skipped aggressive low-quality segment postpasses due to very low DTW confidence"
            )

    if (
        quality < config.quality_mixed_threshold or force_dtw
    ) and not low_confidence_alignment:
        aligned_lines, alignments = _call_apply_low_quality_postpasses_with_metrics(
            apply_low_quality_segment_postpasses_fn,
            aligned_lines=aligned_lines,
            alignments=alignments,
            transcription=transcription,
            epitran_lang=epitran_lang,
            metrics=metrics,
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

    aligned_lines, alignments = _call_finalize_line_set_with_metrics(
        finalize_whisper_line_set_fn,
        source_lines=lines,
        aligned_lines=aligned_lines,
        alignments=alignments,
        transcription=transcription,
        epitran_lang=epitran_lang,
        force_dtw=force_dtw,
        audio_features=audio_features,
        metrics=metrics,
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

    whisper_end = max((w.end for w in all_words), default=0.0)
    baseline_end = _line_set_end(baseline_lines)
    baseline_timeline_ratio = baseline_end / whisper_end if whisper_end > 0.0 else 1.0
    matched_ratio = float(metrics.get("matched_ratio", 0.0) or 0.0)
    avg_similarity = float(metrics.get("avg_similarity", 0.0) or 0.0)
    line_coverage = float(metrics.get("line_coverage", 0.0) or 0.0)
    aligned_end = _line_set_end(aligned_lines)
    aligned_timeline_ratio = aligned_end / whisper_end if whisper_end > 0.0 else 1.0
    aligned_lines = constrain_line_starts_to_baseline_fn(aligned_lines, baseline_lines)
    metrics["baseline_timeline_ratio"] = baseline_timeline_ratio
    metrics["aligned_timeline_ratio"] = aligned_timeline_ratio

    no_whisper_evidence = (
        quality < config.no_evidence_quality_max
        and not force_dtw
        and float(metrics.get("whisperx_forced", 0.0) or 0.0) < 0.5
        and matched_ratio < config.no_evidence_matched_ratio_max
        and line_coverage < config.no_evidence_line_coverage_max
        and avg_similarity < config.no_evidence_avg_similarity_max
    )
    if no_whisper_evidence:
        discarded = len(alignments)
        logger.warning(
            (
                "Rolling back Whisper corrections: insufficient DTW evidence "
                "(matched=%.2f line_coverage=%.2f avg_similarity=%.2f)"
            ),
            matched_ratio,
            line_coverage,
            avg_similarity,
        )
        if discarded > 0:
            alignments = [
                "Rolled back Whisper timing due to insufficient DTW alignment evidence "
                f"(discarded {discarded} tentative correction(s))"
            ]
        else:
            alignments = [
                "Rolled back Whisper timing due to insufficient DTW alignment evidence"
            ]
        aligned_lines = baseline_lines
        metrics["no_evidence_fallback"] = 1.0

    rollback, short_before, short_after = should_rollback_short_line_degradation_fn(
        baseline_lines, aligned_lines
    )
    if rollback:
        repaired_lines, restored_count = restore_implausibly_short_lines_fn(
            baseline_lines, aligned_lines
        )
        repaired_rollback, _, repaired_after = (
            should_rollback_short_line_degradation_fn(baseline_lines, repaired_lines)
        )
        if restored_count > 0 and not repaired_rollback:
            logger.info(
                "Recovered Whisper corrections by restoring %d short baseline line(s) (%d -> %d)",
                restored_count,
                short_after,
                repaired_after,
            )
            alignments.append(
                f"Restored {restored_count} short compressed lines from baseline timing"
            )
            aligned_lines = repaired_lines
        else:
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
        logger.info("Whisper hybrid alignment: %d lines corrected", len(alignments))

    return aligned_lines, alignments, metrics
