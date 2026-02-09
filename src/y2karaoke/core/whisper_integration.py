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
from . import whisper_cache
from . import whisper_dtw
from . import whisper_alignment
from . import whisper_phonetic_dtw
from . import whisper_utils
from . import whisper_blocks
from . import whisper_mapping

_dedupe_whisper_words = whisper_mapping._dedupe_whisper_words

_dedupe_whisper_segments = whisper_mapping._dedupe_whisper_segments
_build_word_to_segment_index = whisper_mapping._build_word_to_segment_index
_find_segment_for_time = whisper_mapping._find_segment_for_time
_word_match_score = whisper_mapping._word_match_score
_find_nearest_word_in_segment = whisper_mapping._find_nearest_word_in_segment
_normalize_line_text = whisper_mapping._normalize_line_text
_trim_whisper_transcription_by_lyrics = (
    whisper_mapping._trim_whisper_transcription_by_lyrics
)
_choose_segment_for_line = whisper_mapping._choose_segment_for_line
_segment_word_indices = whisper_mapping._segment_word_indices
_collect_unused_words_near_line = whisper_mapping._collect_unused_words_near_line
_collect_unused_words_in_window = whisper_mapping._collect_unused_words_in_window
_register_word_match = whisper_mapping._register_word_match
_select_best_candidate = whisper_mapping._select_best_candidate
_filter_and_order_candidates = whisper_mapping._filter_and_order_candidates
_prepare_line_context = whisper_mapping._prepare_line_context
_match_assigned_words = whisper_mapping._match_assigned_words
_fill_unmatched_gaps = whisper_mapping._fill_unmatched_gaps
_compute_gap_window = whisper_mapping._compute_gap_window
_assemble_mapped_line = whisper_mapping._assemble_mapped_line
_map_lrc_words_to_whisper = whisper_mapping._map_lrc_words_to_whisper
_build_word_assignments_from_phoneme_path = (
    whisper_mapping._build_word_assignments_from_phoneme_path
)
_shift_repeated_lines_to_next_whisper = (
    whisper_mapping._shift_repeated_lines_to_next_whisper
)
_enforce_monotonic_line_starts_whisper = (
    whisper_mapping._enforce_monotonic_line_starts_whisper
)
_resolve_line_overlaps = whisper_mapping._resolve_line_overlaps
_redistribute_word_timings_to_line = whisper_utils._redistribute_word_timings_to_line
_clamp_word_gaps = whisper_utils._clamp_word_gaps
_cap_word_durations = whisper_utils._cap_word_durations

_align_dtw_whisper_with_data = whisper_dtw._align_dtw_whisper_with_data
align_dtw_whisper = whisper_dtw.align_dtw_whisper
_compute_dtw_alignment_metrics = whisper_dtw._compute_dtw_alignment_metrics
_retime_lines_from_dtw_alignments = whisper_dtw._retime_lines_from_dtw_alignments

_extract_alignments_from_path = whisper_dtw._extract_alignments_from_path_base
_apply_dtw_alignments = whisper_dtw._apply_dtw_alignments_base
_extract_lrc_words_base = whisper_dtw._extract_lrc_words_base
_compute_phonetic_costs_base = whisper_dtw._compute_phonetic_costs_base
_extract_alignments_from_path_base = whisper_dtw._extract_alignments_from_path_base
_apply_dtw_alignments_base = whisper_dtw._apply_dtw_alignments_base
align_dtw_whisper_base = whisper_dtw.align_dtw_whisper_base

# Re-export functions for compatibility with other modules
_get_whisper_cache_path = whisper_cache._get_whisper_cache_path
_find_best_cached_whisper_model = whisper_cache._find_best_cached_whisper_model
_load_whisper_cache = whisper_cache._load_whisper_cache
_save_whisper_cache = whisper_cache._save_whisper_cache
_model_index = whisper_cache._model_index
_MODEL_ORDER = whisper_cache._MODEL_ORDER

_find_best_whisper_match = whisper_phonetic_dtw._find_best_whisper_match
align_lyrics_to_transcription = whisper_phonetic_dtw.align_lyrics_to_transcription
align_words_to_whisper = whisper_phonetic_dtw.align_words_to_whisper
_assess_lrc_quality = whisper_phonetic_dtw._assess_lrc_quality
_extract_lrc_words = whisper_phonetic_dtw._extract_lrc_words
_compute_phonetic_costs = whisper_phonetic_dtw._compute_phonetic_costs
_compute_phonetic_costs_unbounded = (
    whisper_phonetic_dtw._compute_phonetic_costs_unbounded
)
_extract_best_alignment_map = whisper_phonetic_dtw._extract_best_alignment_map
_extract_lrc_words_all = whisper_phonetic_dtw._extract_lrc_words_all
_build_dtw_path = whisper_phonetic_dtw._build_dtw_path
_build_phoneme_dtw_path = whisper_phonetic_dtw._build_phoneme_dtw_path
_build_syllable_tokens_from_phonemes = (
    whisper_phonetic_dtw._build_syllable_tokens_from_phonemes
)
_make_syllable_from_tokens = whisper_phonetic_dtw._make_syllable_from_tokens
_build_syllable_dtw_path = whisper_phonetic_dtw._build_syllable_dtw_path
_build_phoneme_tokens_from_lrc_words = (
    whisper_phonetic_dtw._build_phoneme_tokens_from_lrc_words
)
_build_phoneme_tokens_from_whisper_words = (
    whisper_phonetic_dtw._build_phoneme_tokens_from_whisper_words
)

_assign_lrc_lines_to_blocks = whisper_blocks._assign_lrc_lines_to_blocks
_text_overlap_score = whisper_blocks._text_overlap_score
_build_segment_word_info = whisper_blocks._build_segment_word_info
_assign_lrc_lines_to_segments = whisper_blocks._assign_lrc_lines_to_segments
_distribute_words_within_segments = whisper_blocks._distribute_words_within_segments
_build_segment_text_overlap_assignments = (
    whisper_blocks._build_segment_text_overlap_assignments
)
_build_block_word_bags = whisper_blocks._build_block_word_bags
_syl_to_block = whisper_blocks._syl_to_block
_group_syllables_by_block = whisper_blocks._group_syllables_by_block
_run_per_block_dtw = whisper_blocks._run_per_block_dtw
_build_block_segmented_syllable_assignments = (
    whisper_blocks._build_block_segmented_syllable_assignments
)

_normalize_word = whisper_utils._normalize_word
_normalize_words_expanded = whisper_utils._normalize_words_expanded
_segment_start = whisper_utils._segment_start
_segment_end = whisper_utils._segment_end
_get_segment_text = whisper_utils._get_segment_text
_compute_speech_blocks = whisper_utils._compute_speech_blocks
_word_idx_to_block = whisper_utils._word_idx_to_block
_block_time_range = whisper_utils._block_time_range
_SPEECH_BLOCK_GAP = whisper_utils._SPEECH_BLOCK_GAP
_build_word_assignments_from_syllable_path = (
    whisper_utils._build_word_assignments_from_syllable_path
)

align_hybrid_lrc_whisper = whisper_alignment.align_hybrid_lrc_whisper
_enforce_monotonic_line_starts = whisper_alignment._enforce_monotonic_line_starts
_scale_line_to_duration = whisper_alignment._scale_line_to_duration
_enforce_non_overlapping_lines = whisper_alignment._enforce_non_overlapping_lines
_merge_lines_to_whisper_segments = whisper_alignment._merge_lines_to_whisper_segments
_retime_adjacent_lines_to_whisper_window = (
    whisper_alignment._retime_adjacent_lines_to_whisper_window
)
_retime_adjacent_lines_to_segment_window = (
    whisper_alignment._retime_adjacent_lines_to_segment_window
)
_pull_next_line_into_segment_window = (
    whisper_alignment._pull_next_line_into_segment_window
)
_pull_next_line_into_same_segment = whisper_alignment._pull_next_line_into_same_segment
_merge_short_following_line_into_segment = (
    whisper_alignment._merge_short_following_line_into_segment
)
_pull_lines_near_segment_end = whisper_alignment._pull_lines_near_segment_end
_clamp_repeated_line_duration = whisper_alignment._clamp_repeated_line_duration
_merge_first_two_lines_if_segment_matches = (
    whisper_alignment._merge_first_two_lines_if_segment_matches
)
_tighten_lines_to_whisper_segments = (
    whisper_alignment._tighten_lines_to_whisper_segments
)
_pull_lines_to_best_segments = whisper_alignment._pull_lines_to_best_segments
_drop_duplicate_lines = whisper_alignment._drop_duplicate_lines
_drop_duplicate_lines_by_timing = whisper_alignment._drop_duplicate_lines_by_timing
_normalize_line_word_timings = whisper_alignment._normalize_line_word_timings
_find_best_whisper_segment = whisper_alignment._find_best_whisper_segment
_apply_offset_to_line = whisper_alignment._apply_offset_to_line
_calculate_drift_correction = whisper_alignment._calculate_drift_correction
_interpolate_unmatched_lines = whisper_alignment._interpolate_unmatched_lines
_refine_unmatched_lines_with_onsets = (
    whisper_alignment._refine_unmatched_lines_with_onsets
)
_fix_ordering_violations = whisper_alignment._fix_ordering_violations
_pull_lines_forward_for_continuous_vocals = (
    whisper_alignment._pull_lines_forward_for_continuous_vocals
)
_fill_vocal_activity_gaps = whisper_alignment._fill_vocal_activity_gaps

_whisper_lang_to_epitran = phonetic_utils._whisper_lang_to_epitran
_get_ipa = phonetic_utils._get_ipa
_phonetic_similarity = phonetic_utils._phonetic_similarity

logger = get_logger(__name__)

__all__ = [
    "transcribe_vocals",
    "align_lyrics_to_transcription",
    "align_words_to_whisper",
    "align_dtw_whisper",
    "correct_timing_with_whisper",
    "align_lrc_text_to_whisper_timings",
    "align_hybrid_lrc_whisper",
    "_get_whisper_cache_path",
    "_find_best_cached_whisper_model",
    "_load_whisper_cache",
    "_save_whisper_cache",
    "_model_index",
    "_MODEL_ORDER",
    "_find_best_whisper_match",
    "_extract_lrc_words",
    "_extract_lrc_words_base",
    "_compute_phonetic_costs",
    "_compute_phonetic_costs_base",
    "_extract_alignments_from_path",
    "_extract_alignments_from_path_base",
    "_apply_dtw_alignments",
    "_apply_dtw_alignments_base",
    "align_dtw_whisper_base",
    "_align_dtw_whisper_with_data",
    "_compute_dtw_alignment_metrics",
    "_retime_lines_from_dtw_alignments",
    "_merge_lines_to_whisper_segments",
    "_retime_adjacent_lines_to_whisper_window",
    "_retime_adjacent_lines_to_segment_window",
    "_pull_next_line_into_segment_window",
    "_pull_next_line_into_same_segment",
    "_merge_short_following_line_into_segment",
    "_pull_lines_near_segment_end",
    "_clamp_repeated_line_duration",
    "_tighten_lines_to_whisper_segments",
    "_apply_offset_to_line",
    "_calculate_drift_correction",
    "_fix_ordering_violations",
    "_find_best_whisper_segment",
    "_assess_lrc_quality",
    "_pull_lines_to_best_segments",
    "_fill_vocal_activity_gaps",
    "_pull_lines_forward_for_continuous_vocals",
    "_merge_first_two_lines_if_segment_matches",
    "_whisper_lang_to_epitran",
    "_get_ipa",
    "_phonetic_similarity",
]


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
            whisper_cache._save_whisper_cache(
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
    lrc_assignments = whisper_blocks._build_segment_text_overlap_assignments(
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
            lrc_assignments = (
                whisper_blocks._build_block_segmented_syllable_assignments(
                    lrc_words,
                    all_words,
                    lrc_syllables,
                    whisper_syllables,
                    epitran_lang,
                )
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
    aligned_lines, alignments = whisper_alignment._fix_ordering_violations(
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
