"""Whisper-based transcription and alignment for lyrics."""

from typing import List, Optional, Tuple, Dict, Any

from ..utils.logging import get_logger
from ..utils.lex_lookup_installer import ensure_local_lex_lookup
from .audio_analysis import (
    extract_audio_features,
)
from . import models
from . import timing_models
from . import phonetic_utils
from .whisper_integration_aliases import ALIAS_EXPORTS, build_aliases

logger = get_logger(__name__)

_ALIASES = build_aliases()
globals().update(_ALIASES)

_get_whisper_cache_path = _ALIASES["_get_whisper_cache_path"]
_find_best_cached_whisper_model = _ALIASES["_find_best_cached_whisper_model"]
_load_whisper_cache = _ALIASES["_load_whisper_cache"]
_save_whisper_cache = _ALIASES["_save_whisper_cache"]
align_lyrics_to_transcription = _ALIASES["align_lyrics_to_transcription"]
align_dtw_whisper = _ALIASES["align_dtw_whisper"]
align_words_to_whisper = _ALIASES["align_words_to_whisper"]
_dedupe_whisper_segments = _ALIASES["_dedupe_whisper_segments"]
_trim_whisper_transcription_by_lyrics = _ALIASES[
    "_trim_whisper_transcription_by_lyrics"
]
_fill_vocal_activity_gaps = _ALIASES["_fill_vocal_activity_gaps"]
_dedupe_whisper_words = _ALIASES["_dedupe_whisper_words"]
_extract_lrc_words_all = _ALIASES["_extract_lrc_words_all"]
_build_phoneme_tokens_from_lrc_words = _ALIASES["_build_phoneme_tokens_from_lrc_words"]
_build_phoneme_tokens_from_whisper_words = _ALIASES[
    "_build_phoneme_tokens_from_whisper_words"
]
_build_syllable_tokens_from_phonemes = _ALIASES["_build_syllable_tokens_from_phonemes"]
_build_segment_text_overlap_assignments = _ALIASES[
    "_build_segment_text_overlap_assignments"
]
_build_phoneme_dtw_path = _ALIASES["_build_phoneme_dtw_path"]
_build_word_assignments_from_phoneme_path = _ALIASES[
    "_build_word_assignments_from_phoneme_path"
]
_build_block_segmented_syllable_assignments = _ALIASES[
    "_build_block_segmented_syllable_assignments"
]
_map_lrc_words_to_whisper = _ALIASES["_map_lrc_words_to_whisper"]
_shift_repeated_lines_to_next_whisper = _ALIASES[
    "_shift_repeated_lines_to_next_whisper"
]
_enforce_monotonic_line_starts_whisper = _ALIASES[
    "_enforce_monotonic_line_starts_whisper"
]
_resolve_line_overlaps = _ALIASES["_resolve_line_overlaps"]
_interpolate_unmatched_lines = _ALIASES["_interpolate_unmatched_lines"]
_refine_unmatched_lines_with_onsets = _ALIASES["_refine_unmatched_lines_with_onsets"]
_assess_lrc_quality = _ALIASES["_assess_lrc_quality"]
align_hybrid_lrc_whisper = _ALIASES["align_hybrid_lrc_whisper"]
_align_dtw_whisper_with_data = _ALIASES["_align_dtw_whisper_with_data"]
_extract_alignments_from_path = _ALIASES["_extract_alignments_from_path"]
_compute_phonetic_costs = _ALIASES["_compute_phonetic_costs"]
_apply_dtw_alignments = _ALIASES["_apply_dtw_alignments"]
_apply_offset_to_line = _ALIASES["_apply_offset_to_line"]
_calculate_drift_correction = _ALIASES["_calculate_drift_correction"]
_find_best_whisper_match = _ALIASES["_find_best_whisper_match"]
_retime_lines_from_dtw_alignments = _ALIASES["_retime_lines_from_dtw_alignments"]
_merge_first_two_lines_if_segment_matches = _ALIASES[
    "_merge_first_two_lines_if_segment_matches"
]
_retime_adjacent_lines_to_whisper_window = _ALIASES[
    "_retime_adjacent_lines_to_whisper_window"
]
_retime_adjacent_lines_to_segment_window = _ALIASES[
    "_retime_adjacent_lines_to_segment_window"
]
_pull_next_line_into_segment_window = _ALIASES["_pull_next_line_into_segment_window"]
_pull_lines_near_segment_end = _ALIASES["_pull_lines_near_segment_end"]
_pull_next_line_into_same_segment = _ALIASES["_pull_next_line_into_same_segment"]
_merge_lines_to_whisper_segments = _ALIASES["_merge_lines_to_whisper_segments"]
_tighten_lines_to_whisper_segments = _ALIASES["_tighten_lines_to_whisper_segments"]
_pull_lines_to_best_segments = _ALIASES["_pull_lines_to_best_segments"]
_fix_ordering_violations = _ALIASES["_fix_ordering_violations"]
_normalize_line_word_timings = _ALIASES["_normalize_line_word_timings"]
_enforce_monotonic_line_starts = _ALIASES["_enforce_monotonic_line_starts"]
_enforce_non_overlapping_lines = _ALIASES["_enforce_non_overlapping_lines"]
_merge_short_following_line_into_segment = _ALIASES[
    "_merge_short_following_line_into_segment"
]
_clamp_repeated_line_duration = _ALIASES["_clamp_repeated_line_duration"]
_drop_duplicate_lines = _ALIASES["_drop_duplicate_lines"]
_drop_duplicate_lines_by_timing = _ALIASES["_drop_duplicate_lines_by_timing"]
_pull_lines_forward_for_continuous_vocals = _ALIASES[
    "_pull_lines_forward_for_continuous_vocals"
]

__all__ = [
    "transcribe_vocals",
    "correct_timing_with_whisper",
    "align_lrc_text_to_whisper_timings",
] + ALIAS_EXPORTS


def transcribe_vocals(
    vocals_path: str,
    language: Optional[str] = None,
    model_size: str = "base",
    aggressive: bool = False,
    temperature: float = 0.0,
) -> Tuple[
    List[timing_models.TranscriptionSegment],
    List[timing_models.TranscriptionWord],
    str,
    str,
]:
    """Transcribe vocals using Whisper.

    Results are cached to disk alongside the vocals file to avoid
    expensive re-transcription on subsequent runs.

    Args:
        vocals_path: Path to vocals audio file
        language: Language code (e.g., 'fr', 'en'). Auto-detected if None.
        model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
        temperature: Temperature for transcription (default: 0.0)

    Returns:
        Tuple of (list of timing_models.TranscriptionSegment, list of all timing_models.TranscriptionWord,
        detected language code, whisper model size used)
    """
    # Check cache first
    cache_path = _get_whisper_cache_path(
        vocals_path, model_size, language, aggressive, temperature
    )
    cached_model = model_size
    if cache_path:
        best_cached = _find_best_cached_whisper_model(
            vocals_path, language, aggressive, model_size, temperature
        )
        if best_cached:
            cache_path, cached_model = best_cached
        cached = _load_whisper_cache(cache_path)
        if cached:
            segments, all_words, detected_lang = cached
            return segments, all_words, detected_lang, cached_model

    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError:
        logger.warning("faster-whisper not installed, cannot transcribe")
        return [], [], "", model_size

    try:
        logger.info(f"Loading Whisper model ({model_size})...")
        model = WhisperModel(model_size, device="cpu", compute_type="int8")

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

        # Convert to list of timing_models.TranscriptionSegment with words
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

        # Save to cache
        if cache_path:
            _save_whisper_cache(
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


_TIME_DRIFT_THRESHOLD = 0.8


def align_lrc_text_to_whisper_timings(  # noqa: C901
    lines: List[models.Line],
    vocals_path: str,
    language: Optional[str] = None,
    model_size: str = "base",
    aggressive: bool = False,
    temperature: float = 0.0,
    min_similarity: float = 0.15,
    audio_features: Optional[timing_models.AudioFeatures] = None,
    lenient_vocal_activity_threshold: float = 0.3,
    lenient_activity_bonus: float = 0.4,
    low_word_confidence_threshold: float = 0.5,
) -> Tuple[List[models.Line], List[str], Dict[str, float]]:
    """Align LRC text to Whisper timings using phonetic DTW (timings fixed).

    This maps each LRC word onto a Whisper word timestamp without changing
    the Whisper timing. The alignment is purely phonetic and monotonic.
    """
    ensure_local_lex_lookup()
    transcription, all_words, detected_lang, used_model = transcribe_vocals(
        vocals_path, language, model_size, aggressive, temperature
    )
    if not audio_features:
        audio_features = extract_audio_features(vocals_path)
    transcription = _dedupe_whisper_segments(transcription)
    transcription = _dedupe_whisper_segments(transcription)
    line_texts = [line.text for line in lines if line.text.strip()]
    transcription, all_words, trimmed_end = _trim_whisper_transcription_by_lyrics(
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
        all_words, filled_segments = _fill_vocal_activity_gaps(
            all_words,
            audio_features,
            lenient_vocal_activity_threshold,
            segments=transcription,
        )
        if filled_segments is not None:
            transcription = filled_segments

    all_words = _dedupe_whisper_words(all_words)

    epitran_lang = phonetic_utils._whisper_lang_to_epitran(detected_lang)
    logger.debug(
        f"Using epitran language: {epitran_lang} (from Whisper: {detected_lang})"
    )

    lrc_words = _extract_lrc_words_all(lines)
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
    lrc_phonemes = _build_phoneme_tokens_from_lrc_words(lrc_words, epitran_lang)
    whisper_phonemes = _build_phoneme_tokens_from_whisper_words(all_words, epitran_lang)

    lrc_syllables = _build_syllable_tokens_from_phonemes(lrc_phonemes)
    whisper_syllables = _build_syllable_tokens_from_phonemes(whisper_phonemes)

    # Use segment-level text overlap for robust line→segment mapping,
    # then fall back to syllable DTW only if segment overlap is poor.
    lrc_assignments = _build_segment_text_overlap_assignments(
        lrc_words,
        all_words,
        transcription,
    )
    seg_coverage = len(lrc_assignments) / len(lrc_words) if lrc_words else 0
    if seg_coverage < 0.3:
        logger.debug(
            "Segment overlap coverage %.0f%% too low, falling back to DTW",
            seg_coverage * 100,
        )
        if not lrc_syllables or not whisper_syllables:
            if not lrc_phonemes or not whisper_phonemes:
                logger.warning("No phoneme/syllable data; skipping mapping")
                return lines, [], {}
            path = _build_phoneme_dtw_path(
                lrc_phonemes,
                whisper_phonemes,
                epitran_lang,
            )
            lrc_assignments = _build_word_assignments_from_phoneme_path(
                path, lrc_phonemes, whisper_phonemes
            )
        else:
            lrc_assignments = _build_block_segmented_syllable_assignments(
                lrc_words,
                all_words,
                lrc_syllables,
                whisper_syllables,
                epitran_lang,
            )

    # Build mapped lines with whisper timings
    corrections: List[str] = []
    mapped_lines, mapped_count, total_similarity, mapped_lines_set = (
        _map_lrc_words_to_whisper(
            lines,
            lrc_words,
            all_words,
            lrc_assignments,
            epitran_lang,
            transcription,
        )
    )

    mapped_lines = _shift_repeated_lines_to_next_whisper(mapped_lines, all_words)
    mapped_lines = _enforce_monotonic_line_starts_whisper(mapped_lines, all_words)
    mapped_lines = _resolve_line_overlaps(mapped_lines)
    mapped_lines = _interpolate_unmatched_lines(mapped_lines, mapped_lines_set)

    # Re-apply onset refinement to lines with no Whisper word matches.
    # These lines have correct segment-anchored timing but evenly-spaced
    # words; onsets give better word boundaries even without text matching.
    mapped_lines = _refine_unmatched_lines_with_onsets(
        mapped_lines,
        mapped_lines_set,
        vocals_path,
    )

    # Onset refinement may move lines to positions that violate monotonicity,
    # so re-enforce ordering constraints.
    mapped_lines = _enforce_monotonic_line_starts_whisper(mapped_lines, all_words)
    mapped_lines = _resolve_line_overlaps(mapped_lines)

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
        "dtw_used": 1.0,
        "dtw_mode": 1.0,
        "whisper_model": used_model,
    }

    if mapped_count:
        corrections.append(f"DTW-phonetic mapped {mapped_count} word(s) to Whisper")
    return mapped_lines, corrections, metrics


def correct_timing_with_whisper(  # noqa: C901
    lines: List[models.Line],
    vocals_path: str,
    language: Optional[str] = None,
    model_size: str = "base",
    aggressive: bool = False,
    temperature: float = 0.0,
    trust_lrc_threshold: float = 1.0,
    correct_lrc_threshold: float = 1.5,
    force_dtw: bool = False,
    audio_features: Optional[timing_models.AudioFeatures] = None,
    lenient_vocal_activity_threshold: float = 0.3,
    lenient_activity_bonus: float = 0.4,
    low_word_confidence_threshold: float = 0.5,
) -> Tuple[List[models.Line], List[str], Dict[str, float]]:
    """Correct lyrics timing using Whisper transcription (adaptive approach).

    Strategy:
    1. Transcribe vocals with Whisper
    2. Assess LRC timing quality (what % of lines are within tolerance of Whisper)
    3. If quality > 70%: LRC is good, only fix individual bad lines
    4. If quality 40-70%: Use hybrid approach (fix bad sections, keep good ones)
    5. If quality < 40%: LRC is broken, use DTW for global alignment

    Args:
        lines: Lyrics lines with potentially wrong timing
        vocals_path: Path to vocals audio
        language: Language code (auto-detected if None)
        model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
        aggressive: Use aggressive Whisper settings
        temperature: Temperature for Whisper transcription
        trust_lrc_threshold: If timing error < this, trust LRC (default: 1.0s)
        correct_lrc_threshold: If timing error > this, use Whisper (default: 1.5s)
        force_dtw: Force DTW alignment regardless of quality
        audio_features: Optional pre-extracted audio features
        lenient_vocal_activity_threshold: Threshold for vocal activity
        lenient_activity_bonus: Bonus for phonetic cost under leniency
        low_word_confidence_threshold: Threshold for whisper word confidence

    Returns:
        Tuple of (corrected lines, list of corrections, metrics)
    """
    # Transcribe vocals (returns segments, all_words, and language)
    transcription, all_words, detected_lang, _model = transcribe_vocals(
        vocals_path, language, model_size, aggressive, temperature
    )
    if not audio_features:
        audio_features = extract_audio_features(vocals_path)

    line_texts = [line.text for line in lines if line.text.strip()]
    transcription, all_words, trimmed_end = _trim_whisper_transcription_by_lyrics(
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
        all_words, filled_segments = _fill_vocal_activity_gaps(
            all_words,
            audio_features,
            lenient_vocal_activity_threshold,
            segments=transcription,
        )
        if filled_segments is not None:
            transcription = filled_segments

    # Map to epitran language code for phonetic matching
    epitran_lang = phonetic_utils._whisper_lang_to_epitran(detected_lang)
    logger.debug(
        f"Using epitran language: {epitran_lang} (from Whisper: {detected_lang})"
    )

    # Pre-compute IPA for Whisper words
    logger.debug(f"Pre-computing IPA for {len(all_words)} Whisper words...")
    for w in all_words:
        phonetic_utils._get_ipa(w.text, epitran_lang)

    # Assess LRC quality
    quality, assessments = _assess_lrc_quality(
        lines, all_words, epitran_lang, tolerance=1.5
    )
    logger.info(f"LRC timing quality: {quality:.0%} of lines within 1.5s of Whisper")

    metrics: Dict[str, float] = {}
    if not force_dtw and quality >= 0.7:
        # LRC is mostly good - only fix individual bad lines using hybrid approach
        logger.info("LRC timing is good, using targeted corrections only")
        aligned_lines, alignments = align_hybrid_lrc_whisper(
            lines,
            transcription,
            all_words,
            language=epitran_lang,
            trust_threshold=trust_lrc_threshold,
            correct_threshold=correct_lrc_threshold,
        )
    elif not force_dtw and quality >= 0.4:
        # Mixed quality - use hybrid approach
        logger.info("LRC timing is mixed, using hybrid Whisper alignment")
        aligned_lines, alignments = align_hybrid_lrc_whisper(
            lines,
            transcription,
            all_words,
            language=epitran_lang,
            trust_threshold=trust_lrc_threshold,
            correct_threshold=correct_lrc_threshold,
        )
    else:
        # LRC is broken - use DTW for global alignment
        logger.info("LRC timing is poor, using DTW global alignment")
        aligned_lines, alignments, metrics, lrc_words, alignments_map = (
            _align_dtw_whisper_with_data(
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
            dtw_lines, dtw_fixes = _retime_lines_from_dtw_alignments(
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

    # Post-process: tighten/merge to Whisper segment boundaries for broken LRC
    if quality < 0.4 or force_dtw:
        aligned_lines, merged_first = _merge_first_two_lines_if_segment_matches(
            aligned_lines, transcription, epitran_lang
        )
        if merged_first:
            alignments.append("Merged first two lines via Whisper segment")
        aligned_lines, pair_retimed = _retime_adjacent_lines_to_whisper_window(
            aligned_lines, transcription, epitran_lang
        )
        if pair_retimed:
            alignments.append(
                f"Retimed {pair_retimed} adjacent line pair(s) to Whisper window"
            )
        aligned_lines, pair_windowed = _retime_adjacent_lines_to_segment_window(
            aligned_lines, transcription, epitran_lang
        )
        if pair_windowed:
            alignments.append(
                f"Retimed {pair_windowed} adjacent line pair(s) to Whisper segment window"
            )
        aligned_lines, pulled_next = _pull_next_line_into_segment_window(
            aligned_lines, transcription, epitran_lang
        )
        if pulled_next:
            alignments.append(
                f"Pulled {pulled_next} line(s) into adjacent segment window"
            )
        aligned_lines, pulled_near_end = _pull_lines_near_segment_end(
            aligned_lines, transcription, epitran_lang
        )
        if pulled_near_end:
            alignments.append(f"Pulled {pulled_near_end} line(s) near segment ends")
        aligned_lines, pulled_same = _pull_next_line_into_same_segment(
            aligned_lines, transcription
        )
        if pulled_same:
            alignments.append(f"Pulled {pulled_same} line(s) into same segment")
        # Re-apply adjacent retiming to keep pairs together after pulls.
        aligned_lines, pair_retimed_after = _retime_adjacent_lines_to_whisper_window(
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
        aligned_lines, merged = _merge_lines_to_whisper_segments(
            aligned_lines, transcription, epitran_lang
        )
        if merged:
            alignments.append(f"Merged {merged} line pair(s) via Whisper segments")
        aligned_lines, tightened = _tighten_lines_to_whisper_segments(
            aligned_lines, transcription, epitran_lang
        )
        if tightened:
            alignments.append(f"Tightened {tightened} line(s) to Whisper segments")
        aligned_lines, pulled = _pull_lines_to_best_segments(
            aligned_lines, transcription, epitran_lang
        )
        if pulled:
            alignments.append(f"Pulled {pulled} line(s) to Whisper segments")

    # Post-process: reject corrections that break line ordering
    aligned_lines, alignments = _fix_ordering_violations(
        lines, aligned_lines, alignments
    )
    aligned_lines = _normalize_line_word_timings(aligned_lines)
    aligned_lines = _enforce_monotonic_line_starts(aligned_lines)
    aligned_lines = _enforce_non_overlapping_lines(aligned_lines)
    if force_dtw:
        aligned_lines, pulled_near_end = _pull_lines_near_segment_end(
            aligned_lines, transcription, epitran_lang
        )
        if pulled_near_end:
            alignments.append(
                f"Pulled {pulled_near_end} line(s) near segment ends (post-order)"
            )
        aligned_lines, merged_short = _merge_short_following_line_into_segment(
            aligned_lines, transcription
        )
        if merged_short:
            alignments.append(
                f"Merged {merged_short} short line(s) into prior segments"
            )
        aligned_lines, clamped_repeat = _clamp_repeated_line_duration(aligned_lines)
        if clamped_repeat:
            alignments.append(f"Clamped {clamped_repeat} repeated line(s) duration")

    aligned_lines, deduped = _drop_duplicate_lines(
        aligned_lines, transcription, epitran_lang
    )
    if deduped:
        alignments.append(f"Dropped {deduped} duplicate line(s)")
    before_drop = len(aligned_lines)
    aligned_lines = [line for line in aligned_lines if line.words]
    if len(aligned_lines) != before_drop:
        alignments.append("Dropped empty lines after Whisper merges")
    aligned_lines, timing_deduped = _drop_duplicate_lines_by_timing(aligned_lines)
    if timing_deduped:
        alignments.append(
            f"Dropped {timing_deduped} duplicate line(s) by timing overlap"
        )

    if audio_features is not None:
        aligned_lines, continuous_fixes = _pull_lines_forward_for_continuous_vocals(
            aligned_lines, audio_features
        )
        if continuous_fixes:
            alignments.append(
                f"Pulled {continuous_fixes} line(s) forward for continuous vocals"
            )

    # Final safety: enforce monotonicity after all post-processing.
    aligned_lines = _enforce_monotonic_line_starts(aligned_lines)
    aligned_lines = _enforce_non_overlapping_lines(aligned_lines)

    if alignments:
        logger.info(f"Whisper hybrid alignment: {len(alignments)} lines corrected")

    return aligned_lines, alignments, metrics
