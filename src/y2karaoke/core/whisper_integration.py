"""Whisper-based transcription and alignment for lyrics."""

import re
from typing import List, Optional, Tuple, Dict, Any, Set, Sequence, Iterable
from collections import defaultdict
from difflib import SequenceMatcher

import numpy as np

from ..utils.logging import get_logger
from .models import Line, Word
from .timing_models import AudioFeatures, TranscriptionWord, TranscriptionSegment
from .phonetic_utils import (
    _get_ipa,
    _get_ipa_segs,
    _phonetic_similarity,
    _text_similarity_basic,
    _text_similarity,
    _whisper_lang_to_epitran,
    _get_panphon_distance,
    _is_vowel,
)
from ..utils.lex_lookup_installer import ensure_local_lex_lookup
from .audio_analysis import (
    _check_vocal_activity_in_range,
    _compute_silence_overlap,
    _is_time_in_silence,
    extract_audio_features,
)
from .whisper_cache import (
    _get_whisper_cache_path,
    _find_best_cached_whisper_model,
    _load_whisper_cache,
    _save_whisper_cache,
    _model_index,
    _MODEL_ORDER,
)
from .whisper_dtw import (
    _LineMappingContext,
    _extract_lrc_words_base,
    _compute_phonetic_costs_base,
    _extract_alignments_from_path_base,
    _apply_dtw_alignments_base,
    align_dtw_whisper_base,
)
from .whisper_alignment import (
    _enforce_monotonic_line_starts,
    _scale_line_to_duration,
    _enforce_non_overlapping_lines,
    _merge_lines_to_whisper_segments,
    _retime_adjacent_lines_to_whisper_window,
    _retime_adjacent_lines_to_segment_window,
    _pull_next_line_into_segment_window,
    _pull_next_line_into_same_segment,
    _merge_short_following_line_into_segment,
    _pull_lines_near_segment_end,
    _clamp_repeated_line_duration,
    _merge_first_two_lines_if_segment_matches,
    _tighten_lines_to_whisper_segments,
    _pull_lines_to_best_segments,
    _drop_duplicate_lines,
    _drop_duplicate_lines_by_timing,
    _normalize_line_word_timings,
    _find_best_whisper_segment,
    _apply_offset_to_line,
    _calculate_drift_correction,
    _interpolate_unmatched_lines,
    _refine_unmatched_lines_with_onsets,
    align_hybrid_lrc_whisper,
    _fix_ordering_violations,
    _pull_lines_forward_for_continuous_vocals,
    _fill_vocal_activity_gaps,
)

_extract_alignments_from_path = _extract_alignments_from_path_base
_apply_dtw_alignments = _apply_dtw_alignments_base

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
]


def transcribe_vocals(
    vocals_path: str,
    language: Optional[str] = None,
    model_size: str = "base",
    aggressive: bool = False,
    temperature: float = 0.0,
) -> Tuple[List[TranscriptionSegment], List[TranscriptionWord], str, str]:
    """Transcribe vocals using Whisper.

    Results are cached to disk alongside the vocals file to avoid
    expensive re-transcription on subsequent runs.

    Args:
        vocals_path: Path to vocals audio file
        language: Language code (e.g., 'fr', 'en'). Auto-detected if None.
        model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
        temperature: Temperature for transcription (default: 0.0)

    Returns:
        Tuple of (list of TranscriptionSegment, list of all TranscriptionWord,
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

        # Convert to list of TranscriptionSegment with words
        result = []
        all_words = []
        for seg in segments:
            seg_words = []
            if seg.words:
                for w in seg.words:
                    tw = TranscriptionWord(
                        start=w.start,
                        end=w.end,
                        text=w.word.strip(),
                        probability=w.probability,
                    )
                    seg_words.append(tw)
                    all_words.append(tw)
            result.append(
                TranscriptionSegment(
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


def _build_phoneme_tokens_from_lrc_words(
    lrc_words: List[Dict], language: str
) -> List[Dict]:
    """Build phoneme-level tokens from LRC words for DTW."""
    tokens: List[Dict] = []
    for idx, word in enumerate(lrc_words):
        text = word.get("text", "")
        word_start = word.get("start", 0.0)
        word_end = word.get("end", 0.0)
        duration = max(word_end - word_start, 0.01)
        ipa = _get_ipa(text, language) or text
        segs = _get_ipa_segs(ipa) or [ipa]
        portion = duration / len(segs)
        for seg_idx, seg in enumerate(segs):
            start = word_start + portion * seg_idx
            end = start + portion
            if seg_idx == len(segs) - 1:
                end = word_end if word_end >= word_start else start + portion
            tokens.append(
                {
                    "word_idx": idx,
                    "parent_idx": idx,
                    "ipa": seg,
                    "start": start,
                    "end": end,
                }
            )
    return tokens


def _build_phoneme_tokens_from_whisper_words(
    whisper_words: List[TranscriptionWord], language: str
) -> List[Dict]:
    """Build phoneme-level tokens from Whisper words for DTW."""
    tokens: List[Dict] = []
    for idx, word in enumerate(whisper_words):
        text = word.text
        word_start = word.start
        word_end = word.end
        duration = max(word_end - word_start, 0.01)
        ipa = _get_ipa(text, language) or text
        segs = _get_ipa_segs(ipa) or [ipa]
        portion = duration / len(segs)
        for seg_idx, seg in enumerate(segs):
            start = word_start + portion * seg_idx
            end = start + portion
            if seg_idx == len(segs) - 1:
                end = word_end if word_end >= word_start else start + portion
            tokens.append(
                {
                    "word_idx": idx,
                    "parent_idx": idx,
                    "ipa": seg,
                    "start": start,
                    "end": end,
                }
            )
    return tokens


def _phoneme_similarity_from_ipa(
    ipa1: str, ipa2: str, language: str = "fra-Latn"
) -> float:
    """Compute phonetic similarity between two IPA segments."""
    if not ipa1 or not ipa2:
        return 0.0
    dst = _get_panphon_distance()
    if dst is None:
        return 1.0 if ipa1 == ipa2 else 0.0
    segs1 = _get_ipa_segs(ipa1)
    segs2 = _get_ipa_segs(ipa2)
    if not segs1 or not segs2:
        return 1.0 if ipa1 == ipa2 else 0.0
    fed = dst.feature_edit_distance(ipa1, ipa2)
    max_segs = max(len(segs1), len(segs2))
    if max_segs == 0:
        return 0.0
    normalized_distance = fed / max_segs
    return max(0.0, 1.0 - normalized_distance)


def align_lyrics_to_transcription(
    lines: List[Line],
    transcription: List[TranscriptionSegment],
    min_similarity: float = 0.4,
    max_time_shift: float = 10.0,
    language: str = "fra-Latn",
) -> Tuple[List[Line], List[str]]:
    """Align lyrics lines to Whisper transcription using fuzzy matching.

    For each lyrics line, finds the best matching transcription segment
    and adjusts timing accordingly. Only applies corrections that are
    within a reasonable time range to avoid catastrophic misalignment.

    Args:
        lines: Lyrics lines with potentially wrong timing
        transcription: Whisper transcription segments with correct timing
        min_similarity: Minimum text similarity to consider a match
        max_time_shift: Maximum time shift to apply (seconds)

    Returns:
        Tuple of (aligned lines, list of alignment descriptions)
    """
    from .models import Line, Word

    if not lines or not transcription:
        return lines, []

    aligned_lines: List[Line] = []
    alignments: List[str] = []

    # Track used segments to avoid double-matching
    used_segments: set = set()

    for i, line in enumerate(lines):
        if not line.words:
            aligned_lines.append(line)
            continue

        line_text = " ".join(w.text for w in line.words)
        line_start = line.start_time

        # Find best matching transcription segment within reasonable time range
        best_match_idx = None
        best_score = -float("inf")
        best_segment = None
        best_similarity = 0.0

        for j, seg in enumerate(transcription):
            if j in used_segments:
                continue

            # Only consider segments within max_time_shift of current position
            time_diff = abs(seg.start - line_start)
            if time_diff > max_time_shift:
                continue

            similarity = _text_similarity(
                line_text, seg.text, use_phonetic=True, language=language
            )

            if similarity < min_similarity:
                continue

            # Score: prioritize similarity, with small time bonus
            time_bonus = max(0, (max_time_shift - time_diff) / max_time_shift) * 0.1
            score = similarity + time_bonus

            if score > best_score:
                best_score = score
                best_similarity = similarity
                best_match_idx = j
                best_segment = seg

        if best_segment is not None and best_similarity >= min_similarity:
            used_segments.add(best_match_idx)

            # Calculate timing adjustment
            offset = best_segment.start - line_start
            new_duration = best_segment.end - best_segment.start

            # Only adjust if offset is significant but not too large
            if 0.3 < abs(offset) <= max_time_shift:
                # Redistribute words across the new duration
                word_count = len(line.words)
                word_spacing = new_duration / word_count if word_count > 0 else 0

                new_words = []
                for k, word in enumerate(line.words):
                    new_start = best_segment.start + k * word_spacing
                    new_end = new_start + (word_spacing * 0.9)
                    new_words.append(
                        Word(
                            text=word.text,
                            start_time=new_start,
                            end_time=new_end,
                            singer=word.singer,
                        )
                    )

                aligned_line = Line(words=new_words, singer=line.singer)
                aligned_lines.append(aligned_line)

                alignments.append(
                    f"Line {i+1} aligned to transcription: {offset:+.1f}s "
                    f'(similarity: {best_similarity:.0%}) "{line_text[:30]}..."'
                )
                continue

        # No good match found or no adjustment needed
        aligned_lines.append(line)

    return aligned_lines, alignments


def _find_best_whisper_match(
    lrc_text: str,
    lrc_start: float,
    sorted_whisper: List[TranscriptionWord],
    used_indices: set,
    min_similarity: float,
    max_time_shift: float,
    language: str,
    min_index: int = 0,
    whisper_percentiles: Optional[List[float]] = None,
    expected_percentile: float = 0.0,
    order_weight: float = 0.35,
    time_weight: float = 0.25,
) -> Tuple[Optional[TranscriptionWord], Optional[int], float]:
    """Find the best matching Whisper word for an LRC word.

    Returns:
        Tuple of (best_match, best_match_index, best_similarity)

    Args:
        min_index: Whisper index to start searching from so matches stay in order.
        whisper_percentiles: Optional percentile list used for ordering constraints.
        expected_percentile: Expected percentile of the current LRC word to guide ordering.
        order_weight: Weight for penalizing distant percentile matches.
        time_weight: Weight for penalizing long time offsets.
    """
    best_match = None
    best_match_idx = None
    best_similarity = 0.0
    best_score = float("-inf")

    for i in range(min_index, len(sorted_whisper)):
        ww = sorted_whisper[i]
        if i in used_indices:
            continue

        time_diff = abs(ww.start - lrc_start)
        if time_diff > max_time_shift:
            if ww.start > lrc_start + max_time_shift:
                break
            continue

        basic_sim = _text_similarity_basic(lrc_text, ww.text, language)
        if basic_sim < 0.25:
            continue

        similarity = _phonetic_similarity(lrc_text, ww.text, language)

        if similarity < min_similarity:
            continue

        position_penalty = 0.0
        if whisper_percentiles and i < len(whisper_percentiles):
            position_penalty = abs(expected_percentile - whisper_percentiles[i])

        time_penalty = min(time_diff / max_time_shift, 1.0)
        score = (
            similarity - position_penalty * order_weight - time_penalty * time_weight
        )

        if score > best_score:
            best_score = score
            best_match = ww
            best_match_idx = i
            best_similarity = similarity

    return best_match, best_match_idx, best_similarity


def align_words_to_whisper(
    lines: List[Line],
    whisper_words: List[TranscriptionWord],
    min_similarity: float = 0.5,
    max_time_shift: float = 5.0,
    language: str = "fra-Latn",
) -> Tuple[List[Line], List[str]]:
    """Align individual LRC words to Whisper word timestamps using phonetic matching.

    Args:
        lines: Lyrics lines with word-level timing
        whisper_words: All Whisper transcription words with timestamps
        min_similarity: Minimum phonetic similarity to consider a match
        max_time_shift: Maximum time difference to search for matches
        language: Epitran language code for phonetic matching

    Returns:
        Tuple of (aligned lines, list of correction descriptions)
    """
    from .models import Line, Word

    if not lines or not whisper_words:
        return lines, []

    # Pre-compute IPA for all Whisper words
    logger.debug(f"Pre-computing IPA for {len(whisper_words)} Whisper words...")
    for ww in whisper_words:
        _get_ipa(ww.text, language)

    sorted_whisper = sorted(whisper_words, key=lambda w: w.start)
    total_whisper = len(sorted_whisper)
    whisper_percentiles = [i / max(1, total_whisper - 1) for i in range(total_whisper)]
    used_whisper_indices: set = set()
    aligned_lines: List[Line] = []
    corrections: List[str] = []
    min_whisper_index = 0
    total_lrc_words = sum(
        1 for line in lines for word in line.words if len(word.text.strip()) >= 2
    )
    lrc_word_index = 0

    for line in lines:
        if not line.words:
            aligned_lines.append(line)
            continue

        new_words: List[Word] = []
        line_corrections = 0

        for word in line.words:
            lrc_text = word.text.strip()
            if len(lrc_text) < 2:
                new_words.append(word)
                continue

            expected_percentile = (
                lrc_word_index / max(1, total_lrc_words - 1)
                if total_lrc_words > 1
                else 0.0
            )
            best_match, best_idx, _ = _find_best_whisper_match(
                lrc_text,
                word.start_time,
                sorted_whisper,
                used_whisper_indices,
                min_similarity,
                max_time_shift,
                language,
                min_index=min_whisper_index,
                whisper_percentiles=whisper_percentiles,
                expected_percentile=expected_percentile,
            )

            if best_match is not None and best_idx is not None:
                used_whisper_indices.add(best_idx)
                time_shift = best_match.start - word.start_time
                min_whisper_index = max(min_whisper_index, best_idx + 1)
                if abs(time_shift) > 0.15:
                    new_words.append(
                        Word(
                            text=word.text,
                            start_time=best_match.start,
                            end_time=best_match.end,
                            singer=word.singer,
                        )
                    )
                    line_corrections += 1
                else:
                    new_words.append(word)
            else:
                new_words.append(word)

            lrc_word_index += 1

        aligned_lines.append(Line(words=new_words, singer=line.singer))
        if line_corrections > 0:
            line_text = " ".join(w.text for w in line.words)[:40]
            corrections.append(
                f'Line aligned {line_corrections} word(s): "{line_text}..."'
            )

    return aligned_lines, corrections


def _assess_lrc_quality(
    lines: List[Line],
    whisper_words: List[TranscriptionWord],
    language: str = "fra-Latn",
    tolerance: float = 1.5,
) -> Tuple[float, List[Tuple[int, float, float]]]:
    """Assess LRC timing quality by comparing against Whisper.

    Returns:
        Tuple of (quality_score 0-1, list of (line_idx, lrc_time, best_whisper_time))
    """
    if not lines or not whisper_words:
        return 1.0, []

    assessments = []
    good_count = 0

    for line_idx, line in enumerate(lines):
        if not line.words:
            continue

        # Get first significant word in line
        first_word = None
        for w in line.words:
            if len(w.text.strip()) >= 2:
                first_word = w
                break
        if not first_word:
            continue

        lrc_time = first_word.start_time
        lrc_text = first_word.text

        # Find best matching Whisper word
        best_whisper_time = None
        best_similarity = 0.0

        for ww in whisper_words:
            # Only consider words within 20s
            if abs(ww.start - lrc_time) > 20:
                continue

            sim = _phonetic_similarity(lrc_text, ww.text, language)
            if sim > best_similarity:
                best_similarity = sim
                best_whisper_time = ww.start

        if best_whisper_time is not None and best_similarity >= 0.5:
            time_diff = abs(best_whisper_time - lrc_time)
            assessments.append((line_idx, lrc_time, best_whisper_time))
            if time_diff <= tolerance:
                good_count += 1

    quality = good_count / len(assessments) if assessments else 1.0
    return quality, assessments


def _extract_lrc_words(lines: List[Line]) -> List[Dict]:
    """Extract all LRC words with their line/word indices."""
    lrc_words = []
    for line_idx, line in enumerate(lines):
        for word_idx, word in enumerate(line.words):
            if len(word.text.strip()) >= 2:
                lrc_words.append(
                    {
                        "line_idx": line_idx,
                        "word_idx": word_idx,
                        "text": word.text,
                        "start": word.start_time,
                        "end": word.end_time,
                        "word": word,
                    }
                )
    return lrc_words


def _compute_phonetic_costs(
    lrc_words: List[Dict],
    whisper_words: List[TranscriptionWord],
    language: str,
    min_similarity: float,
) -> Dict[Tuple[int, int], float]:
    """Compute sparse phonetic cost matrix for DTW."""

    phonetic_costs = defaultdict(lambda: 1.0)

    for i, lw in enumerate(lrc_words):
        lrc_time = lw["start"]
        for j, ww in enumerate(whisper_words):
            if ww.text == "[VOCAL]":
                # Give a reasonable cost that is still better than no match
                phonetic_costs[(i, j)] = 0.5
                continue
            time_diff = abs(ww.start - lrc_time)
            if time_diff > 20:
                continue
            sim = _phonetic_similarity(lw["text"], ww.text, language)
            if sim >= min_similarity:
                phonetic_costs[(i, j)] = 1.0 - sim

    return phonetic_costs


def _compute_phonetic_costs_unbounded(
    lrc_words: List[Dict],
    whisper_words: List[TranscriptionWord],
    language: str,
    min_similarity: float,
) -> Dict[Tuple[int, int], float]:
    """Compute phonetic costs without time-window constraints."""

    phonetic_costs = defaultdict(lambda: 1.0)

    for i, lw in enumerate(lrc_words):
        for j, ww in enumerate(whisper_words):
            if ww.text == "[VOCAL]":
                phonetic_costs[(i, j)] = 0.5
                continue
            sim = _phonetic_similarity(lw["text"], ww.text, language)
            if sim >= min_similarity:
                phonetic_costs[(i, j)] = 1.0 - sim

    return phonetic_costs


def _extract_best_alignment_map(
    path: List[Tuple[int, int]],
    lrc_words: List[Dict],
    whisper_words: List[TranscriptionWord],
    language: str,
) -> Dict[int, Tuple[int, float]]:
    """Extract best whisper index per LRC word index from a DTW path."""
    alignments: Dict[int, Tuple[int, float]] = {}
    for lrc_idx, whisper_idx in path:
        lw = lrc_words[lrc_idx]
        ww = whisper_words[whisper_idx]
        sim = _phonetic_similarity(lw["text"], ww.text, language)
        if lrc_idx not in alignments or sim > alignments[lrc_idx][1]:
            alignments[lrc_idx] = (whisper_idx, sim)
    return alignments


def _extract_lrc_words_all(lines: List[Line]) -> List[Dict]:
    """Extract all LRC words (including short tokens) with line/word indices."""
    lrc_words = []
    for line_idx, line in enumerate(lines):
        for word_idx, word in enumerate(line.words):
            if not word.text.strip():
                continue
            lrc_words.append(
                {
                    "line_idx": line_idx,
                    "word_idx": word_idx,
                    "text": word.text,
                    "start": word.start_time,
                    "end": word.end_time,
                    "word": word,
                }
            )
    return lrc_words


def _build_dtw_path(
    lrc_words: List[Dict],
    all_words: List[TranscriptionWord],
    phonetic_costs: Dict[Tuple[int, int], float],
    language: str,
) -> List[Tuple[int, int]]:
    """Build a DTW path between LRC words and Whisper words."""
    try:
        from fastdtw import fastdtw  # type: ignore

        lrc_seq = np.arange(len(lrc_words)).reshape(-1, 1)
        whisper_seq = np.arange(len(all_words)).reshape(-1, 1)

        def dtw_dist(a, b):
            i = int(a[0])
            j = int(b[0])
            return phonetic_costs[(i, j)]

        _distance, path = fastdtw(lrc_seq, whisper_seq, dist=dtw_dist)
        return path
    except ImportError:
        logger.warning("fastdtw not available, falling back to greedy alignment")
        path = []
        whisper_idx = 0
        for lrc_idx in range(len(lrc_words)):
            best_idx = whisper_idx
            best_sim = -1.0
            for j in range(whisper_idx, min(whisper_idx + 6, len(all_words))):
                sim = _phonetic_similarity(
                    lrc_words[lrc_idx]["text"], all_words[j].text, language
                )
                if sim > best_sim:
                    best_sim = sim
                    best_idx = j
            path.append((lrc_idx, best_idx))
            whisper_idx = best_idx
        return path


def _build_phoneme_dtw_path(
    lrc_phonemes: List[Dict],
    whisper_phonemes: List[Dict],
    language: str,
) -> List[Tuple[int, int]]:
    """Build DTW path between phoneme tokens."""
    cost_cache: Dict[Tuple[int, int], float] = {}
    n_lrc = max(len(lrc_phonemes), 1)
    n_whisper = max(len(whisper_phonemes), 1)

    def phoneme_cost(i: int, j: int) -> float:
        key = (i, j)
        if key in cost_cache:
            return cost_cache[key]
        ipa1 = lrc_phonemes[i]["ipa"]
        ipa2 = whisper_phonemes[j]["ipa"]
        sim = _phoneme_similarity_from_ipa(ipa1, ipa2, language)
        phon_cost = 1.0 - sim
        pos_penalty = abs(i / n_lrc - j / n_whisper)
        cost_cache[key] = 0.85 * phon_cost + 0.15 * pos_penalty
        return cost_cache[key]

    try:
        from fastdtw import fastdtw  # type: ignore

        lrc_seq = np.arange(len(lrc_phonemes)).reshape(-1, 1)
        whisper_seq = np.arange(len(whisper_phonemes)).reshape(-1, 1)

        def dtw_dist(a, b):
            i = int(a[0])
            j = int(b[0])
            return phoneme_cost(i, j)

        _distance, path = fastdtw(lrc_seq, whisper_seq, dist=dtw_dist)
        return path
    except ImportError:
        logger.warning(
            "fastdtw not available, falling back to phoneme greedy alignment"
        )
        path = []
        whisper_idx = 0
        for lrc_idx in range(len(lrc_phonemes)):
            best_idx = whisper_idx
            best_cost = float("inf")
            for j in range(whisper_idx, min(whisper_idx + 12, len(whisper_phonemes))):
                cost = phoneme_cost(lrc_idx, j)
                if cost < best_cost:
                    best_cost = cost
                    best_idx = j
        path.append((lrc_idx, best_idx))
        whisper_idx = best_idx
    return path


def _build_syllable_tokens_from_phonemes(phoneme_tokens: List[Dict]) -> List[Dict]:
    """Group phoneme tokens into syllable-level units."""
    syllables: List[Dict] = []
    current: List[Dict] = []
    for token in phoneme_tokens:
        current.append(token)
        if _is_vowel(token["ipa"]):
            syllables.append(_make_syllable_from_tokens(current))
            current = []
        elif (
            current
            and token["parent_idx"] != current[-1]["parent_idx"]
            and all(_is_vowel(t["ipa"]) for t in current)
        ):
            syllables.append(_make_syllable_from_tokens(current))
            current = []
    if current:
        syllables.append(_make_syllable_from_tokens(current))
    return syllables


def _make_syllable_from_tokens(tokens: List[Dict]) -> Dict:
    start = min(t["start"] for t in tokens)
    end = max(t["end"] for t in tokens)
    ipa = "".join(t["ipa"] for t in tokens)
    parent_idxs = {t["parent_idx"] for t in tokens}
    word_idxs = {t["word_idx"] for t in tokens}
    return {
        "ipa": ipa,
        "start": start,
        "end": end,
        "parent_idxs": parent_idxs,
        "word_idxs": word_idxs,
    }


def _build_syllable_dtw_path(
    lrc_syllables: List[Dict],
    whisper_syllables: List[Dict],
    language: str,
) -> List[Tuple[int, int]]:
    """Build DTW path between syllable units.

    Uses phonetic similarity as the primary cost and adds a small positional
    penalty so that repeated phrases are matched to the temporally closest
    occurrence rather than an arbitrary one.
    """
    cost_cache: Dict[Tuple[int, int], float] = {}
    n_lrc = max(len(lrc_syllables), 1)
    n_whisper = max(len(whisper_syllables), 1)

    def syllable_cost(i: int, j: int) -> float:
        key = (i, j)
        if key in cost_cache:
            return cost_cache[key]
        ipa1 = lrc_syllables[i]["ipa"]
        ipa2 = whisper_syllables[j]["ipa"]
        sim = _phoneme_similarity_from_ipa(ipa1, ipa2, language)
        phon_cost = 1.0 - sim
        # Positional penalty: discourage matching syllables at very
        # different relative positions in their respective sequences.
        pos_lrc = i / n_lrc
        pos_whisper = j / n_whisper
        pos_penalty = abs(pos_lrc - pos_whisper)
        cost_cache[key] = 0.85 * phon_cost + 0.15 * pos_penalty
        return cost_cache[key]

    try:
        from fastdtw import fastdtw  # type: ignore

        lrc_seq = np.arange(len(lrc_syllables)).reshape(-1, 1)
        whisper_seq = np.arange(len(whisper_syllables)).reshape(-1, 1)

        def dtw_dist(a, b):
            i = int(a[0])
            j = int(b[0])
            return syllable_cost(i, j)

        _distance, path = fastdtw(lrc_seq, whisper_seq, dist=dtw_dist)
        return path
    except ImportError:
        logger.warning(
            "fastdtw not available, falling back to syllable greedy alignment"
        )
        path = []
        whisper_idx = 0
        for lrc_idx in range(len(lrc_syllables)):
            best_idx = whisper_idx
            best_cost = float("inf")
            for j in range(whisper_idx, min(whisper_idx + 12, len(whisper_syllables))):
                cost = syllable_cost(lrc_idx, j)
                if cost < best_cost:
                    best_cost = cost
                    best_idx = j
            path.append((lrc_idx, best_idx))
            whisper_idx = best_idx
        return path


def _normalize_word(w: str) -> str:
    """Lowercase and strip punctuation for bag-of-words comparison."""
    return w.strip(".,!?;:'\"()- ").lower()


def _normalize_words_expanded(w: str) -> List[str]:
    """Normalize and split hyphenated/compound words for overlap matching."""
    base = w.strip(".,!?;:'\"()- ").lower()
    if not base:
        return []
    parts = [p for p in base.split("-") if p]
    return parts if len(parts) > 1 else [base]


def _assign_lrc_lines_to_blocks(
    lrc_words: List[Dict],
    block_word_bags: List[Set[str]],
) -> List[int]:
    """Assign each LRC line to a speech block using text-overlap scoring.

    For each LRC line, count how many of its content words appear in each
    block's word bag.  Normalise by line length, pick the best block, then
    enforce monotonicity (blocks never go backwards).
    """
    lrc_line_count = max((lw["line_idx"] for lw in lrc_words), default=-1) + 1
    lrc_line_block: List[int] = []
    for li in range(lrc_line_count):
        line_words_text = [
            _normalize_word(lw["text"]) for lw in lrc_words if lw["line_idx"] == li
        ]
        if not line_words_text:
            lrc_line_block.append(lrc_line_block[-1] if lrc_line_block else 0)
            continue
        best_blk = 0
        best_score = -1.0
        for bi, bag in enumerate(block_word_bags):
            hits = sum(1 for w in line_words_text if len(w) > 1 and w in bag)
            score = hits / len(line_words_text) if line_words_text else 0.0
            if score > best_score:
                best_score = score
                best_blk = bi
        lrc_line_block.append(best_blk)

    # Enforce monotonic block assignment (lines only advance to later blocks)
    for i in range(1, lrc_line_count):
        if lrc_line_block[i] < lrc_line_block[i - 1]:
            lrc_line_block[i] = lrc_line_block[i - 1]

    return lrc_line_block


def _text_overlap_score(
    line_words: List[str],
    seg_bag: List[str],
) -> float:
    """Score how many content words in *line_words* appear in *seg_bag*."""
    if not line_words or not seg_bag:
        return 0.0
    seg_expanded: Set[str] = set()
    for w in seg_bag:
        seg_expanded.update(_normalize_words_expanded(w))
    hits = 0
    total = 0
    for w in line_words:
        parts = _normalize_words_expanded(w)
        for p in parts:
            if len(p) > 1:
                total += 1
                if p in seg_expanded:
                    hits += 1
    return hits / max(total, 1)


def _build_segment_word_info(
    all_words: List[TranscriptionWord],
    segments: List[TranscriptionSegment],
) -> Tuple[List[Tuple[int, int]], List[List[str]]]:
    """Return (word-index ranges, word-bags) for each Whisper segment."""
    seg_word_ranges: List[Tuple[int, int]] = []
    seg_word_bags: List[List[str]] = []
    for seg in segments:
        seg_start = _segment_start(seg)
        seg_end = _segment_end(seg)
        first_idx = -1
        last_idx = -1
        for wi, w in enumerate(all_words):
            if w.start >= seg_start - 0.05 and w.end <= seg_end + 0.05:
                if first_idx < 0:
                    first_idx = wi
                last_idx = wi
        seg_word_ranges.append((first_idx, last_idx))
        if first_idx >= 0:
            seg_word_bags.append(
                [
                    _normalize_word(all_words[i].text)
                    for i in range(first_idx, last_idx + 1)
                ]
            )
        else:
            seg_word_bags.append([])
    return seg_word_ranges, seg_word_bags


def _assign_lrc_lines_to_segments(
    lrc_lines_words: List[List[Tuple[int, str]]],
    seg_word_bags: List[List[str]],
) -> List[int]:
    """Assign each LRC line to a Whisper segment using text overlap."""
    lrc_line_count = len(lrc_lines_words)
    n_segs = len(seg_word_bags)
    line_to_seg: List[int] = [-1] * lrc_line_count
    seg_cursor = 0
    for li in range(lrc_line_count):
        words = [w for _, w in lrc_lines_words[li]]
        if not words:
            line_to_seg[li] = line_to_seg[li - 1] if li > 0 else 0
            continue
        best_seg = seg_cursor
        best_score = -1.0
        search_end = min(seg_cursor + max(10, n_segs // 4), n_segs)
        for si in range(seg_cursor, search_end):
            score = _text_overlap_score(words, seg_word_bags[si])
            if score > best_score:
                best_score = score
                best_seg = si
            if si + 1 < n_segs:
                merged = seg_word_bags[si] + seg_word_bags[si + 1]
                mscore = _text_overlap_score(words, merged)
                if mscore > best_score:
                    best_score = mscore
                    s1 = _text_overlap_score(words, seg_word_bags[si])
                    # Prefer the earlier segment — karaoke lines should
                    # appear when the first word is sung.  Only use the
                    # later segment if the earlier has zero overlap.
                    best_seg = si if s1 > 0 else si + 1
        # Zero-score lines (e.g. "Oooh") have no text match; advance
        # past the cursor so subsequent lines don't cascade early.
        if best_score <= 0 and best_seg <= seg_cursor:
            best_seg = min(seg_cursor + 1, n_segs - 1)
        # When a line maps to the same segment as the previous line,
        # check if the next segment has a comparable score.  If so,
        # advance — repeated/similar lines should use consecutive segs.
        # Require a minimum absolute score (0.5) for the next segment to
        # prevent spurious advancement on low-overlap matches (e.g. a
        # single common word like "où" matching many unrelated segments).
        if (
            li > 0
            and best_seg == line_to_seg[li - 1]
            and best_seg + 1 < n_segs
            and best_score > 0
        ):
            nxt = _text_overlap_score(words, seg_word_bags[best_seg + 1])
            if nxt >= best_score * 0.7 and nxt > 0.5:
                best_seg = best_seg + 1
                best_score = nxt
        line_to_seg[li] = best_seg
        seg_cursor = max(seg_cursor, best_seg)
    return line_to_seg


def _distribute_words_within_segments(
    line_to_seg: List[int],
    lrc_lines_words: List[List[Tuple[int, str]]],
    seg_word_ranges: List[Tuple[int, int]],
) -> Dict[int, List[int]]:
    """Positionally map LRC words to Whisper words within each segment."""
    seg_to_lines: Dict[int, List[int]] = {}
    for li, si in enumerate(line_to_seg):
        if si >= 0:
            seg_to_lines.setdefault(si, []).append(li)

    assignments: Dict[int, List[int]] = {}
    for si, line_indices in seg_to_lines.items():
        first_wi, last_wi = seg_word_ranges[si]
        if first_wi < 0:
            continue
        seg_wc = last_wi - first_wi + 1
        all_lrc: List[int] = []
        for li in line_indices:
            for idx, _ in lrc_lines_words[li]:
                all_lrc.append(idx)
        total = len(all_lrc)
        if total == 0:
            continue
        for j, lrc_idx in enumerate(all_lrc):
            pos = j / max(total, 1)
            offset = min(int(pos * seg_wc), seg_wc - 1)
            assignments[lrc_idx] = [first_wi + offset]
    return assignments


def _build_segment_text_overlap_assignments(
    lrc_words: List[Dict],
    all_words: List[TranscriptionWord],
    segments: List[TranscriptionSegment],
) -> Dict[int, List[int]]:
    """Assign LRC words to Whisper words via segment-level text overlap.

    Uses the Whisper segment structure as temporal anchors instead of
    syllable DTW, which drifts when word counts differ between LRC and
    Whisper.
    """
    if not segments or not lrc_words:
        return {}

    seg_word_ranges, seg_word_bags = _build_segment_word_info(
        all_words,
        segments,
    )

    lrc_line_count = max((lw["line_idx"] for lw in lrc_words), default=-1) + 1
    lrc_lines_words: List[List[Tuple[int, str]]] = [[] for _ in range(lrc_line_count)]
    for idx, lw in enumerate(lrc_words):
        lrc_lines_words[lw["line_idx"]].append((idx, _normalize_word(lw["text"])))

    line_to_seg = _assign_lrc_lines_to_segments(
        lrc_lines_words,
        seg_word_bags,
    )
    assignments = _distribute_words_within_segments(
        line_to_seg,
        lrc_lines_words,
        seg_word_ranges,
    )

    logger.debug(
        "Segment text-overlap: %d/%d LRC words assigned to %d segments",
        len(assignments),
        len(lrc_words),
        len(segments),
    )
    return assignments


def _build_block_word_bags(
    all_words: List[TranscriptionWord],
    speech_blocks: List[Tuple[int, int]],
) -> List[Set[str]]:
    """Build a bag-of-words set for each speech block."""
    bags: List[Set[str]] = []
    for blk_start, blk_end in speech_blocks:
        bag: Set[str] = set()
        for wi in range(blk_start, blk_end + 1):
            nw = _normalize_word(all_words[wi].text)
            if nw:
                bag.add(nw)
        bags.append(bag)
    return bags


def _syl_to_block(
    syl_parent_idxs: Set[int], speech_blocks: List[Tuple[int, int]]
) -> int:
    """Return the speech block for a syllable (using its first parent word)."""
    if not syl_parent_idxs:
        return -1
    first_word = min(syl_parent_idxs)
    return _word_idx_to_block(first_word, speech_blocks)


def _group_syllables_by_block(
    syl_word_idxs: List[Set[int]],
    speech_blocks: List[Tuple[int, int]],
) -> Dict[int, List[int]]:
    """Group syllable indices by their speech block."""
    block_syls: Dict[int, List[int]] = {}
    for si, pidxs in enumerate(syl_word_idxs):
        blk = _syl_to_block(pidxs, speech_blocks)
        block_syls.setdefault(blk, []).append(si)
    return block_syls


def _run_per_block_dtw(
    speech_blocks: List[Tuple[int, int]],
    block_lrc_syls: Dict[int, List[int]],
    block_whisper_syls: Dict[int, List[int]],
    lrc_syllables: List[Dict],
    whisper_syllables: List[Dict],
    all_words: List[TranscriptionWord],
    language: str,
) -> Dict[int, List[int]]:
    """Run DTW within each speech block and merge into global assignments."""
    combined_assignments: Dict[int, Set[int]] = {}

    for blk_idx in range(len(speech_blocks)):
        blk_lrc_syl_idxs = block_lrc_syls.get(blk_idx, [])
        blk_whi_syl_idxs = block_whisper_syls.get(blk_idx, [])
        if not blk_lrc_syl_idxs or not blk_whi_syl_idxs:
            continue

        blk_lrc_syls_list = [lrc_syllables[i] for i in blk_lrc_syl_idxs]
        blk_whi_syls_list = [whisper_syllables[i] for i in blk_whi_syl_idxs]

        path = _build_syllable_dtw_path(blk_lrc_syls_list, blk_whi_syls_list, language)

        for local_li, local_wi in path:
            global_li = blk_lrc_syl_idxs[local_li]
            global_wi = blk_whi_syl_idxs[local_wi]
            lrc_token = lrc_syllables[global_li]
            whisper_token = whisper_syllables[global_wi]
            for word_idx in lrc_token.get("word_idxs", set()):
                combined_assignments.setdefault(word_idx, set()).update(
                    whisper_token.get("parent_idxs", set())
                )

        blk_start, blk_end = speech_blocks[blk_idx]
        logger.debug(
            "  Block %d: %d LRC syls -> %d Whisper syls (words %d-%d, %.1f-%.1fs)",
            blk_idx,
            len(blk_lrc_syls_list),
            len(blk_whi_syls_list),
            blk_start,
            blk_end,
            all_words[blk_start].start,
            all_words[blk_end].end,
        )

    return {
        word_idx: sorted(list(indices))
        for word_idx, indices in combined_assignments.items()
    }


def _build_block_segmented_syllable_assignments(
    lrc_words: List[Dict],
    all_words: List[TranscriptionWord],
    lrc_syllables: List[Dict],
    whisper_syllables: List[Dict],
    language: str,
) -> Dict[int, List[int]]:
    """Run syllable DTW within each speech block and merge assignments.

    Instead of a single global DTW (which can match repeated phrases to the
    wrong temporal region), this partitions Whisper words into contiguous
    speech blocks separated by silence, assigns LRC words to blocks
    proportionally, and runs an independent DTW within each block.
    """
    speech_blocks = _compute_speech_blocks(all_words)
    if len(speech_blocks) <= 1:
        # No silence gaps – fall back to a single global DTW
        path = _build_syllable_dtw_path(lrc_syllables, whisper_syllables, language)
        return _build_word_assignments_from_syllable_path(
            path, lrc_syllables, whisper_syllables
        )

    logger.debug(
        "Block-segmented DTW: %d speech blocks across %d Whisper words",
        len(speech_blocks),
        len(all_words),
    )

    # ---- build syllable-index → word-index mappings ----
    whisper_syl_word_idxs: List[Set[int]] = [
        (
            s["parent_idxs"]
            if isinstance(s.get("parent_idxs"), set)
            else set(s.get("parent_idxs", set()))
        )
        for s in whisper_syllables
    ]
    lrc_syl_word_idxs: List[Set[int]] = [
        (
            s["word_idxs"]
            if isinstance(s.get("word_idxs"), set)
            else set(s.get("word_idxs", set()))
        )
        for s in lrc_syllables
    ]

    # Group Whisper syllables by block
    block_whisper_syls = _group_syllables_by_block(
        whisper_syl_word_idxs,
        speech_blocks,
    )

    # Assign LRC lines → blocks via text overlap, then map to syllables
    total_lrc_words = len(lrc_words)
    block_word_bags = _build_block_word_bags(all_words, speech_blocks)
    lrc_line_block = _assign_lrc_lines_to_blocks(lrc_words, block_word_bags)
    lrc_word_block = [lrc_line_block[lw["line_idx"]] for lw in lrc_words]

    n_blocks = len(speech_blocks)
    logger.debug(
        "Text-overlap block assignment: %s",
        {bi: sum(1 for b in lrc_line_block if b == bi) for bi in range(n_blocks)},
    )

    # Build LRC syllable ranges per block
    lrc_syl_to_block: List[int] = []
    for _sidx, wids in enumerate(lrc_syl_word_idxs):
        first_word = min(wids) if wids else 0
        blk = (
            lrc_word_block[first_word]
            if first_word < total_lrc_words
            else (n_blocks - 1)
        )
        lrc_syl_to_block.append(blk)

    block_lrc_syls: Dict[int, List[int]] = {}
    for si, blk in enumerate(lrc_syl_to_block):
        block_lrc_syls.setdefault(blk, []).append(si)

    return _run_per_block_dtw(
        speech_blocks,
        block_lrc_syls,
        block_whisper_syls,
        lrc_syllables,
        whisper_syllables,
        all_words,
        language,
    )


def _dedupe_whisper_words(words: List[TranscriptionWord]) -> List[TranscriptionWord]:
    """Remove Whisper words that share the same start time and text."""
    deduped: List[TranscriptionWord] = []
    seen: Set[Tuple[int, str]] = set()
    for word in words:
        key = (round(word.start * 1000), word.text.strip().lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(word)
    return deduped


def _segment_start(segment: Any) -> float:
    if hasattr(segment, "start"):
        return segment.start
    return segment.get("start", 0.0)


def _segment_end(segment: Any) -> float:
    if hasattr(segment, "end"):
        return segment.end
    return segment.get("end", 0.0)


def _get_segment_text(segment: Any) -> str:
    if hasattr(segment, "text"):
        return segment.text or ""
    if isinstance(segment, dict):
        return segment.get("text", "") or ""
    return ""


_MIN_DUPLICATE_SEGMENT_DURATION = 0.25


def _dedupe_whisper_segments(
    segments: List[TranscriptionSegment],
) -> List[TranscriptionSegment]:
    if not segments:
        return segments
    cleaned: List[TranscriptionSegment] = []
    for seg in segments:
        duration = _segment_end(seg) - _segment_start(seg)
        norm_text = _normalize_line_text(_get_segment_text(seg))
        if cleaned:
            prev = cleaned[-1]
            prev_norm = _normalize_line_text(_get_segment_text(prev))
            if (
                duration <= _MIN_DUPLICATE_SEGMENT_DURATION
                and norm_text
                and prev_norm == norm_text
            ):
                prev.end = max(prev.end, _segment_end(seg))
                continue
        cleaned.append(seg)
    return cleaned


def _build_word_to_segment_index(
    all_words: List[TranscriptionWord], segments: Sequence[Any]
) -> Dict[int, int]:
    """Map each Whisper word index to its enclosing segment."""
    mapping: Dict[int, int] = {}
    if not segments:
        return mapping
    seg_idx = 0
    for idx, word in enumerate(all_words):
        while seg_idx + 1 < len(segments) and word.start > _segment_end(
            segments[seg_idx]
        ):
            seg_idx += 1
        seg = segments[seg_idx]
        if _segment_start(seg) <= word.start <= _segment_end(seg):
            mapping[idx] = seg_idx
    return mapping


def _find_segment_for_time(
    time: float,
    segments: Sequence[Any],
    start_idx: int = 0,
    excluded_segments: Optional[Set[int]] = None,
) -> Optional[int]:
    """Find the segment index whose start is closest to the given time."""
    if not segments:
        return None
    best_idx = None
    best_diff = float("inf")
    excluded_segments = excluded_segments or set()
    for idx in range(start_idx, len(segments)):
        if idx in excluded_segments:
            continue
        seg = segments[idx]
        diff = abs(_segment_start(seg) - time)
        if diff < best_diff:
            best_diff = diff
            best_idx = idx
        if _segment_start(seg) > time and diff > best_diff:
            break
    return best_idx


_BACKWARD_START_PENALTY = 0.5


def _word_match_score(
    word_start: float,
    target_start: float,
    word_seg: Optional[int],
    target_seg: Optional[int],
    segments: Sequence[Any],
    line_anchor: Optional[float] = None,
    lrc_idx: Optional[int] = None,
    candidate_idx: Optional[int] = None,
    total_lrc_words: Optional[int] = None,
    total_whisper_words: Optional[int] = None,
) -> float:
    """Score a word match by time difference with a small segment penalty."""
    score = abs(word_start - target_start)
    if target_seg is None or word_seg is None:
        if line_anchor is not None and word_start < line_anchor:
            score += (line_anchor - word_start) * _BACKWARD_START_PENALTY
        return score
    if word_seg != target_seg:
        seg_diff = abs(
            _segment_start(segments[word_seg]) - _segment_start(segments[target_seg])
        )
        score += 0.4 + 0.01 * seg_diff
    if line_anchor is not None and word_start < line_anchor:
        score += (line_anchor - word_start) * _BACKWARD_START_PENALTY

    if (
        lrc_idx is not None
        and candidate_idx is not None
        and total_lrc_words
        and total_whisper_words
    ):
        lrc_pct = lrc_idx / max(1, total_lrc_words - 1)
        cand_pct = candidate_idx / max(1, total_whisper_words - 1)
        score += abs(lrc_pct - cand_pct) * _ORDER_POSITION_WEIGHT

    return score


def _find_nearest_word_in_segment(
    candidates: Iterable[Tuple[TranscriptionWord, int]],
    target_start: float,
    segment_idx: Optional[int],
    word_segment_idx: Dict[int, int],
) -> Optional[Tuple[TranscriptionWord, int]]:
    if segment_idx is None:
        return None
    best = None
    best_diff = float("inf")
    for word, idx in candidates:
        if word_segment_idx.get(idx) != segment_idx:
            continue
        diff = abs(word.start - target_start)
        if diff < best_diff:
            best_diff = diff
            best = (word, idx)
    return best


def _normalize_line_text(text: str) -> str:
    return re.sub(r"[^a-z0-9\\s]", "", text.lower())


def _trim_whisper_transcription_by_lyrics(
    segments: List[TranscriptionSegment],
    words: List[TranscriptionWord],
    lyric_texts: List[str],
    min_similarity: float = 0.45,
) -> Tuple[List[TranscriptionSegment], List[TranscriptionWord], Optional[float]]:
    """Keep only the portion of the Whisper transcript that matches the provided lyrics."""
    normalized_lyrics = [
        _normalize_line_text(text) for text in lyric_texts if text.strip()
    ]
    if not normalized_lyrics or not segments:
        return segments, words, None

    last_match_end: Optional[float] = None
    for seg in reversed(segments):
        seg_norm = _normalize_line_text(seg.text)
        if not seg_norm:
            continue
        best_ratio = max(
            SequenceMatcher(None, seg_norm, lyric).ratio()
            for lyric in normalized_lyrics
        )
        if best_ratio >= min_similarity:
            last_match_end = seg.end
            break

    if last_match_end is None:
        return segments, words, None

    trimmed_segments = [seg for seg in segments if seg.end <= last_match_end]
    trimmed_words = [word for word in words if word.start <= last_match_end]
    return trimmed_segments, trimmed_words, last_match_end


def _choose_segment_for_line(
    line: Line,
    segments: Sequence[Any],
    start_idx: int = 0,
    min_start: float = 0.0,
    excluded_segments: Optional[Set[int]] = None,
) -> Optional[int]:
    """Greedily pick the best future segment that matches the line text."""
    if not segments:
        return None
    norm_line = _normalize_line_text(line.text)
    best_idx = None
    best_score = float("-inf")
    excluded_segments = excluded_segments or set()
    for idx in range(start_idx, len(segments)):
        if idx in excluded_segments:
            continue
        segment = segments[idx]
        seg_text = _get_segment_text(segment)
        if not seg_text:
            continue
        norm_seg = _normalize_line_text(seg_text)
        text_score = SequenceMatcher(None, norm_line, norm_seg).ratio()
        if text_score < _SEGMENT_MIN_TEXT_SCORE:
            continue
        seg_start = _segment_start(segment)
        if seg_start < min_start:
            continue
        if seg_start - line.start_time > _MAX_SEGMENT_LOOKAHEAD:
            break
        if seg_start < min_start:
            continue
        time_diff = abs(seg_start - line.start_time)
        time_penalty = min(time_diff, _SEGMENT_MAX_TIME_DIFF) * _SEGMENT_TIME_WEIGHT
        index_penalty = (idx - start_idx) * _SEGMENT_INDEX_WEIGHT
        score = text_score - time_penalty - index_penalty
        if score > best_score:
            best_score = score
            best_idx = idx
        if time_diff > _SEGMENT_MAX_TIME_DIFF and text_score < 0.5:
            break
    return best_idx


def _segment_word_indices(
    all_words: List[TranscriptionWord],
    word_segment_idx: Dict[int, int],
    segment_idx: int,
) -> List[int]:
    """Return word indices belonging to the given Whisper segment."""
    indices = [idx for idx, seg in word_segment_idx.items() if seg == segment_idx]
    return sorted(indices, key=lambda i: all_words[i].start)


def _collect_unused_words_near_line(
    all_words: List[TranscriptionWord],
    line: Line,
    used_indices: Set[int],
    start_idx: int,
    min_start: float = 0.0,
    lookahead: float = 1.0,
) -> List[int]:
    """Return unused Whisper word indices near the given line timing."""
    window_start = max(min_start, line.start_time - 0.5)
    window_end = line.end_time + lookahead
    collected: List[int] = []
    for idx in range(start_idx, len(all_words)):
        if idx in used_indices:
            continue
        word = all_words[idx]
        if word.start < window_start:
            continue
        if word.start > window_end:
            break
        collected.append(idx)
        if len(collected) >= max(1, len(line.words)):
            break
    return collected


def _collect_unused_words_in_window(
    all_words: List[TranscriptionWord],
    used_indices: Set[int],
    start_idx: int,
    window_start: float,
    window_end: float,
) -> List[int]:
    """Return unused Whisper word indices within the specified time window."""
    if window_end <= window_start:
        return []
    collected: List[int] = []
    for idx in range(start_idx, len(all_words)):
        if idx in used_indices:
            continue
        word = all_words[idx]
        if word.start < window_start:
            continue
        if word.start > window_end:
            break
        collected.append(idx)
    return collected


_SPEECH_BLOCK_GAP = 5.0  # seconds – gap between Whisper words that defines a new block


def _compute_speech_blocks(
    all_words: List[TranscriptionWord],
    min_gap: float = _SPEECH_BLOCK_GAP,
) -> List[Tuple[int, int]]:
    """Group Whisper words into speech blocks separated by gaps >= *min_gap*.

    Returns a list of ``(first_word_idx, last_word_idx)`` inclusive tuples.
    """
    if not all_words:
        return []
    blocks: List[Tuple[int, int]] = []
    block_start = 0
    for i in range(1, len(all_words)):
        gap = all_words[i].start - all_words[i - 1].end
        if gap >= min_gap:
            blocks.append((block_start, i - 1))
            block_start = i
    blocks.append((block_start, len(all_words) - 1))
    return blocks


def _word_idx_to_block(word_idx: int, speech_blocks: List[Tuple[int, int]]) -> int:
    """Return the speech-block index that contains *word_idx*, or -1."""
    for i, (start, end) in enumerate(speech_blocks):
        if start <= word_idx <= end:
            return i
    return -1


def _block_time_range(
    block_idx: int,
    speech_blocks: List[Tuple[int, int]],
    all_words: List[TranscriptionWord],
) -> Tuple[float, float]:
    """Return (start_time, end_time) for the given speech block."""
    first, last = speech_blocks[block_idx]
    return all_words[first].start, all_words[last].end


_TIME_DRIFT_THRESHOLD = 0.8
_SEGMENT_TIME_WEIGHT = 0.18
_SEGMENT_INDEX_WEIGHT = 0.01
_SEGMENT_MAX_TIME_DIFF = 5.0
_SEGMENT_MIN_TEXT_SCORE = 0.25
_MAX_SEGMENT_LOOKAHEAD = 4.0
_ORDER_POSITION_WEIGHT = 0.35


def _build_word_assignments_from_syllable_path(
    path: List[Tuple[int, int]],
    lrc_syllables: List[Dict],
    whisper_syllables: List[Dict],
) -> Dict[int, List[int]]:
    assignments: Dict[int, Set[int]] = {}
    for lrc_idx, whisper_idx in path:
        lrc_token = lrc_syllables[lrc_idx]
        whisper_token = whisper_syllables[whisper_idx]
        for word_idx in lrc_token["word_idxs"]:
            assignments.setdefault(word_idx, set()).update(whisper_token["parent_idxs"])
    return {
        word_idx: sorted(list(indices)) for word_idx, indices in assignments.items()
    }


def _register_word_match(
    ctx: _LineMappingContext,
    line_idx: int,
    word: "Word",
    best_word: TranscriptionWord,
    best_idx: int,
    candidates: List[Tuple[TranscriptionWord, int]],
    line_segment: Optional[int],
    line_matches: List[Tuple[int, Tuple[float, float]]],
    line_match_intervals: Dict[int, Tuple[float, float]],
    word_idx: int,
    line_last_idx_ref: List[Optional[int]],
) -> None:
    """Register a word match and update tracking state."""
    start, end = best_word.start, best_word.end
    line_matches.append((word_idx, (start, end)))
    line_match_intervals[word_idx] = (start, end)
    ctx.mapped_count += 1
    ctx.mapped_lines_set.add(line_idx)
    best_sim = max(
        _phonetic_similarity(word.text, ww.text, ctx.language) for ww, _ in candidates
    )
    ctx.total_similarity += best_sim
    ctx.used_word_indices.add(best_idx)
    best_seg = ctx.word_segment_idx.get(best_idx)
    if best_seg is not None:
        ctx.current_segment = max(ctx.current_segment, best_seg)
    elif line_segment is not None:
        ctx.current_segment = max(ctx.current_segment, line_segment)
    # Advance speech block tracker when a match lands in a later block
    if ctx.speech_blocks:
        match_blk = _word_idx_to_block(best_idx, ctx.speech_blocks)
        if match_blk > ctx.current_block:
            ctx.current_block = match_blk
    if line_last_idx_ref[0] is None or best_idx > line_last_idx_ref[0]:
        line_last_idx_ref[0] = best_idx


def _select_best_candidate(
    ctx: _LineMappingContext,
    whisper_candidates: List[Tuple[TranscriptionWord, int]],
    word: "Word",
    line_shift: float,
    line_segment: Optional[int],
    line_anchor_time: float,
    lrc_idx_opt: Optional[int],
) -> Tuple[TranscriptionWord, int]:
    """Score candidates, pick the best, and apply drift fallback."""
    best_word, best_idx = min(
        whisper_candidates,
        key=lambda pair: _word_match_score(
            pair[0].start,
            word.start_time + line_shift,
            ctx.word_segment_idx.get(pair[1]),
            line_segment,
            ctx.segments,
            line_anchor_time,
            lrc_idx=lrc_idx_opt,
            candidate_idx=pair[1],
            total_lrc_words=ctx.total_lrc_words,
            total_whisper_words=ctx.total_whisper_words,
        ),
    )
    drift = abs(best_word.start - word.start_time)
    if drift > _TIME_DRIFT_THRESHOLD:
        fallback = _find_nearest_word_in_segment(
            whisper_candidates,
            word.start_time,
            line_segment,
            ctx.word_segment_idx,
        )
        if fallback and abs(fallback[0].start - word.start_time) < drift:
            best_word, best_idx = fallback
    return best_word, best_idx


def _filter_and_order_candidates(
    ctx: _LineMappingContext,
    candidate_indices: Set[int],
) -> List[Tuple[TranscriptionWord, int]]:
    """Sort candidates by time, filter used, and prefer ordered ones.

    Speech-block awareness: candidates in the current block are preferred.
    A candidate from a later block is only accepted when the current block
    has no viable candidates left.
    """
    # When speech blocks are available, relax the forward-only constraint for
    # candidates in the current block.  The constraint still applies across
    # block boundaries to prevent jumping backwards to a previous block.
    if ctx.speech_blocks:
        cur_blk = ctx.current_block
        cur_blk_start = (
            ctx.speech_blocks[cur_blk][0] if cur_blk < len(ctx.speech_blocks) else 0
        )
        min_idx = min(ctx.next_word_idx_start, cur_blk_start)
    else:
        min_idx = ctx.next_word_idx_start

    sorted_indices = [
        idx
        for idx in sorted(candidate_indices, key=lambda idx: ctx.all_words[idx].start)
        if idx >= min_idx and idx not in ctx.used_word_indices
    ]
    whisper_candidates = [(ctx.all_words[i], i) for i in sorted_indices]
    ordered = [
        pair for pair in whisper_candidates if pair[0].start >= ctx.last_line_start
    ]
    if ordered:
        whisper_candidates = ordered

    # --- speech-block filtering ---
    if ctx.speech_blocks and whisper_candidates:
        in_block = [
            (w, idx)
            for w, idx in whisper_candidates
            if _word_idx_to_block(idx, ctx.speech_blocks) == cur_blk
        ]
        if in_block:
            whisper_candidates = in_block
        else:
            # Current block exhausted – allow next block only (no skipping)
            next_blk = cur_blk + 1
            in_next = [
                (w, idx)
                for w, idx in whisper_candidates
                if _word_idx_to_block(idx, ctx.speech_blocks) == next_blk
            ]
            if in_next:
                whisper_candidates = in_next

    return whisper_candidates


def _prepare_line_context(
    ctx: _LineMappingContext,
    line: "Line",
) -> Tuple[Optional[int], float, float]:
    """Determine segment, anchor time, and shift for a line."""
    line_segment = _choose_segment_for_line(
        line,
        ctx.segments,
        ctx.current_segment,
        min_start=ctx.last_line_start,
        excluded_segments=ctx.used_segments,
    )
    if line_segment is None:
        line_segment = _find_segment_for_time(
            line.start_time,
            ctx.segments,
            ctx.current_segment,
            excluded_segments=ctx.used_segments,
        )
    if (
        line_segment is not None
        and ctx.segments
        and _segment_start(ctx.segments[line_segment]) < ctx.last_line_start
    ):
        line_segment = None
    line_anchor_time = max(line.start_time, ctx.last_line_start, ctx.prev_line_end)
    line_shift = line_anchor_time - line.start_time
    return line_segment, line_anchor_time, line_shift


def _match_assigned_words(
    ctx: _LineMappingContext,
    line_idx: int,
    line: "Line",
    lrc_index_by_loc: Dict[Tuple[int, int], int],
    lrc_assignments: Dict[int, List[int]],
    line_segment: Optional[int],
    line_anchor_time: float,
    line_shift: float,
    line_matches: List[Tuple[int, Tuple[float, float]]],
    line_match_intervals: Dict[int, Tuple[float, float]],
    line_last_idx_ref: List[Optional[int]],
) -> None:
    """Phase 1: match words using DTW assignments."""
    for word_idx, word in enumerate(line.words):
        lrc_idx_opt = lrc_index_by_loc.get((line_idx, word_idx))
        if lrc_idx_opt is None:
            continue
        assigned = lrc_assignments.get(lrc_idx_opt)
        if not assigned:
            continue
        candidate_indices = set(assigned)
        if line_segment is not None:
            candidate_indices.update(
                _segment_word_indices(ctx.all_words, ctx.word_segment_idx, line_segment)
            )
        whisper_candidates = _filter_and_order_candidates(ctx, candidate_indices)
        if not whisper_candidates:
            fallback_indices = _collect_unused_words_near_line(
                ctx.all_words,
                line,
                ctx.used_word_indices,
                ctx.next_word_idx_start,
                ctx.prev_line_end,
            )
            whisper_candidates = _filter_and_order_candidates(
                ctx, set(fallback_indices)
            )
        if not whisper_candidates:
            continue
        best_word, best_idx = _select_best_candidate(
            ctx,
            whisper_candidates,
            word,
            line_shift,
            line_segment,
            line_anchor_time,
            lrc_idx_opt,
        )
        _register_word_match(
            ctx,
            line_idx,
            word,
            best_word,
            best_idx,
            whisper_candidates,
            line_segment,
            line_matches,
            line_match_intervals,
            word_idx,
            line_last_idx_ref,
        )


def _fill_unmatched_gaps(
    ctx: _LineMappingContext,
    line_idx: int,
    line: "Line",
    lrc_index_by_loc: Dict[Tuple[int, int], int],
    lrc_assignments: Dict[int, List[int]],
    line_segment: Optional[int],
    line_anchor_time: float,
    line_shift: float,
    line_matches: List[Tuple[int, Tuple[float, float]]],
    line_match_intervals: Dict[int, Tuple[float, float]],
    line_last_idx_ref: List[Optional[int]],
) -> None:
    """Phase 2: fill unmatched words using time-window search."""
    unmatched = [
        idx for idx in range(len(line.words)) if idx not in line_match_intervals
    ]
    for word_idx in unmatched:
        word = line.words[word_idx]
        gap_start, gap_end = _compute_gap_window(
            line, word_idx, line_match_intervals, ctx.prev_line_end
        )
        if gap_end <= gap_start:
            continue
        lrc_idx_opt = lrc_index_by_loc.get((line_idx, word_idx))
        assigned = (
            lrc_assignments.get(lrc_idx_opt, []) if lrc_idx_opt is not None else []
        )
        candidate_indices = set(assigned)
        if line_segment is not None:
            candidate_indices.update(
                _segment_word_indices(ctx.all_words, ctx.word_segment_idx, line_segment)
            )
        window_start = max(gap_start - 0.25, 0.0)
        window_end = gap_end + 0.25
        # Clip window to current speech block so we don't reach across silence
        if ctx.speech_blocks and ctx.current_block < len(ctx.speech_blocks):
            blk_start_t, blk_end_t = _block_time_range(
                ctx.current_block, ctx.speech_blocks, ctx.all_words
            )
            window_start = max(window_start, blk_start_t - 0.25)
            window_end = min(window_end, blk_end_t + 0.25)
        # Use block-aware min index (same as _filter_and_order_candidates)
        if ctx.speech_blocks and ctx.current_block < len(ctx.speech_blocks):
            gap_min_idx = min(
                ctx.next_word_idx_start,
                ctx.speech_blocks[ctx.current_block][0],
            )
        else:
            gap_min_idx = ctx.next_word_idx_start
        window_candidates = _collect_unused_words_in_window(
            ctx.all_words,
            ctx.used_word_indices,
            gap_min_idx,
            window_start,
            window_end,
        )
        candidate_indices.update(window_candidates)
        sorted_indices = [
            idx
            for idx in sorted(
                candidate_indices,
                key=lambda idx: ctx.all_words[idx].start,
            )
            if idx >= gap_min_idx and idx not in ctx.used_word_indices
        ]
        filtered_indices = [
            idx
            for idx in sorted_indices
            if ctx.all_words[idx].start >= gap_start
            and ctx.all_words[idx].start <= window_end
        ]
        whisper_candidates = [(ctx.all_words[idx], idx) for idx in filtered_indices]
        ordered = [
            pair for pair in whisper_candidates if pair[0].start >= ctx.last_line_start
        ]
        if ordered:
            whisper_candidates = ordered
        # Apply speech-block preference (same logic as _filter_and_order_candidates)
        if ctx.speech_blocks and whisper_candidates:
            cur_blk = ctx.current_block
            in_block = [
                (w, idx)
                for w, idx in whisper_candidates
                if _word_idx_to_block(idx, ctx.speech_blocks) == cur_blk
            ]
            if in_block:
                whisper_candidates = in_block
            else:
                in_next = [
                    (w, idx)
                    for w, idx in whisper_candidates
                    if _word_idx_to_block(idx, ctx.speech_blocks) == cur_blk + 1
                ]
                if in_next:
                    whisper_candidates = in_next
        if not whisper_candidates:
            continue
        best_word, best_idx = _select_best_candidate(
            ctx,
            whisper_candidates,
            word,
            line_shift,
            line_segment,
            line_anchor_time,
            lrc_idx_opt,
        )
        _register_word_match(
            ctx,
            line_idx,
            word,
            best_word,
            best_idx,
            whisper_candidates,
            line_segment,
            line_matches,
            line_match_intervals,
            word_idx,
            line_last_idx_ref,
        )


def _compute_gap_window(
    line: "Line",
    word_idx: int,
    line_match_intervals: Dict[int, Tuple[float, float]],
    prev_line_end: float,
) -> Tuple[float, float]:
    """Compute the time window for an unmatched word based on neighbors."""
    prev_matches = [idx for idx in line_match_intervals if idx < word_idx]
    next_matches = [idx for idx in line_match_intervals if idx > word_idx]
    gap_start = line.start_time
    if prev_matches:
        gap_start = max(gap_start, line_match_intervals[max(prev_matches)][1])
    gap_start = max(gap_start, prev_line_end)
    gap_end = line.end_time
    if next_matches:
        gap_end = min(gap_end, line_match_intervals[min(next_matches)][0])
    return gap_start, gap_end


def _assemble_mapped_line(
    ctx: _LineMappingContext,
    line_idx: int,
    line: "Line",
    line_matches: List[Tuple[int, Tuple[float, float]]],
    line_match_intervals: Dict[int, Tuple[float, float]],
    line_anchor_time: float,
    line_segment: Optional[int],
    line_last_idx_ref: List[Optional[int]],
) -> "Line":
    """Build a mapped Line from match intervals and update tracking state."""
    from .models import Line as LineModel, Word

    mapped_words: List[Word] = []
    for word_idx, word in enumerate(line.words):
        interval = line_match_intervals.get(word_idx)
        if interval:
            start, end = interval
        else:
            start, end = word.start_time, word.end_time
        mapped_words.append(
            Word(
                text=word.text,
                start_time=start,
                end_time=end,
                singer=word.singer,
            )
        )

    mapped_line = LineModel(words=mapped_words, singer=line.singer)
    if line_matches:
        actual_start = min(iv[0] for _, iv in line_matches)
        actual_end = max(iv[1] for _, iv in line_matches)
    else:
        original_duration = line.end_time - line.start_time
        if original_duration <= 0:
            original_duration = len(line.words) * 0.6
        actual_start = line_anchor_time
        actual_end = actual_start + original_duration
    target_duration = max(actual_end - actual_start, 0.0)
    mapped_line = _redistribute_word_timings_to_line(
        mapped_line,
        line_matches,
        target_duration=target_duration,
        min_word_duration=0.05,
        line_start=actual_start,
    )
    logger.debug(
        "Mapped line %d start=%.2f end=%.2f matches=%d",
        line_idx + 1,
        mapped_line.start_time,
        mapped_line.end_time,
        len(line_matches),
    )
    if mapped_line.words:
        ctx.last_line_start = mapped_line.start_time
        ctx.prev_line_end = mapped_line.end_time
    if line_segment is not None:
        ctx.used_segments.add(line_segment)
    if line_last_idx_ref[0] is not None:
        ctx.next_word_idx_start = line_last_idx_ref[0] + 1
    return mapped_line


def _map_lrc_words_to_whisper(
    lines: List[Line],
    lrc_words: List[Dict],
    all_words: List[TranscriptionWord],
    lrc_assignments: Dict[int, List[int]],
    language: str,
    segments: Sequence[Any],
) -> Tuple[List[Line], int, float, set]:
    """Build mapped lines using Whisper timings based on LRC assignments."""
    segments = segments or []
    speech_blocks = _compute_speech_blocks(all_words)
    if speech_blocks:
        logger.debug(
            "Speech blocks: %d (gaps >= %.1fs)",
            len(speech_blocks),
            _SPEECH_BLOCK_GAP,
        )
    ctx = _LineMappingContext(
        all_words=all_words,
        segments=segments,
        word_segment_idx=_build_word_to_segment_index(all_words, segments),
        language=language,
        total_lrc_words=len(lrc_words),
        total_whisper_words=len(all_words),
        speech_blocks=speech_blocks,
    )
    lrc_index_by_loc = {
        (lw["line_idx"], lw["word_idx"]): idx for idx, lw in enumerate(lrc_words)
    }
    mapped_lines: List[Line] = []

    for line_idx, line in enumerate(lines):
        if not line.words:
            mapped_lines.append(line)
            continue

        line_segment, line_anchor_time, line_shift = _prepare_line_context(ctx, line)

        # Override line_segment with the segment determined by the
        # text-overlap assignment.  _prepare_line_context uses the
        # line's original (placeholder) timing which may be far from
        # the correct position; the assignments already encode the
        # right segment.
        assigned_segs: Dict[int, int] = {}
        for word_idx in range(len(line.words)):
            lrc_idx = lrc_index_by_loc.get((line_idx, word_idx))
            if lrc_idx is not None:
                for wi in lrc_assignments.get(lrc_idx, []):
                    si = ctx.word_segment_idx.get(wi)
                    if si is not None:
                        assigned_segs[si] = assigned_segs.get(si, 0) + 1
        if assigned_segs:
            override_seg = max(assigned_segs, key=assigned_segs.get)  # type: ignore[arg-type]
            if line_segment != override_seg:
                line_segment = override_seg
                if ctx.segments and override_seg < len(ctx.segments):
                    seg_start = _segment_start(ctx.segments[override_seg])
                    line_anchor_time = max(seg_start, ctx.prev_line_end)
                    line_shift = line_anchor_time - line.start_time

        line_matches: List[Tuple[int, Tuple[float, float]]] = []
        line_match_intervals: Dict[int, Tuple[float, float]] = {}
        line_last_idx_ref: List[Optional[int]] = [None]

        _match_assigned_words(
            ctx,
            line_idx,
            line,
            lrc_index_by_loc,
            lrc_assignments,
            line_segment,
            line_anchor_time,
            line_shift,
            line_matches,
            line_match_intervals,
            line_last_idx_ref,
        )
        _fill_unmatched_gaps(
            ctx,
            line_idx,
            line,
            lrc_index_by_loc,
            lrc_assignments,
            line_segment,
            line_anchor_time,
            line_shift,
            line_matches,
            line_match_intervals,
            line_last_idx_ref,
        )
        mapped_line = _assemble_mapped_line(
            ctx,
            line_idx,
            line,
            line_matches,
            line_match_intervals,
            line_anchor_time,
            line_segment,
            line_last_idx_ref,
        )
        mapped_lines.append(mapped_line)

    return mapped_lines, ctx.mapped_count, ctx.total_similarity, ctx.mapped_lines_set


def _build_word_assignments_from_phoneme_path(
    path: List[Tuple[int, int]],
    lrc_phonemes: List[Dict],
    whisper_phonemes: List[Dict],
) -> Dict[int, List[int]]:
    """Convert phoneme-level DTW matches back to word-level assignments."""
    assignments: Dict[int, Set[int]] = {}
    for lpc_idx, wpc_idx in path:
        lrc_token = lrc_phonemes[lpc_idx]
        whisper_token = whisper_phonemes[wpc_idx]
        word_idx = lrc_token["word_idx"]
        whisper_word_idx = whisper_token["parent_idx"]
        assignments.setdefault(word_idx, set()).add(whisper_word_idx)
    return {
        word_idx: sorted(list(indices)) for word_idx, indices in assignments.items()
    }


def _shift_repeated_lines_to_next_whisper(
    mapped_lines: List[Line],
    all_words: List[TranscriptionWord],
) -> List[Line]:
    """Ensure repeated lines reserve later Whisper words when they reappear."""
    from .models import Line, Word

    adjusted_lines: List[Line] = []
    last_idx_by_text: Dict[str, int] = {}
    last_end_time: Dict[str, float] = {}

    for line in mapped_lines:
        if not line.words:
            adjusted_lines.append(line)
            continue

        text_norm = line.text.strip().lower() if getattr(line, "text", "") else ""
        prev_idx = last_idx_by_text.get(text_norm)
        prev_end = last_end_time.get(text_norm)
        assigned_end_idx: Optional[int] = None

        if prev_idx is not None and prev_end is not None:
            required_time = max(prev_end + 0.4, line.start_time)
            start_idx = next(
                (
                    idx
                    for idx, ww in enumerate(all_words)
                    if idx > prev_idx and ww.start >= required_time
                ),
                None,
            )
            if start_idx is None:
                start_idx = next(
                    (idx for idx, _ww in enumerate(all_words) if idx > prev_idx),
                    None,
                )
            # If the next Whisper word is far from the previous occurrence,
            # the audio likely has an instrumental break or the lyrics don't
            # match the audio structure.  Skip the shift and let
            # _interpolate_unmatched_lines distribute the line instead.
            if start_idx is not None and (all_words[start_idx].start - prev_end > 10.0):
                start_idx = None
            if start_idx is not None:
                adjusted_words: List[Word] = []
                for word_idx, w in enumerate(line.words):
                    new_idx = min(start_idx + word_idx, len(all_words) - 1)
                    ww = all_words[new_idx]
                    adjusted_words.append(
                        Word(
                            text=w.text,
                            start_time=ww.start,
                            end_time=ww.end,
                            singer=w.singer,
                        )
                    )
                line = Line(words=adjusted_words, singer=line.singer)
                assigned_end_idx = min(
                    start_idx + len(line.words) - 1, len(all_words) - 1
                )

        adjusted_lines.append(line)
        if line.words:
            if assigned_end_idx is None:
                assigned_end_idx = next(
                    (
                        idx
                        for idx, ww in enumerate(all_words)
                        if abs(ww.start - line.words[-1].start_time) < 0.05
                    ),
                    len(all_words) - 1,
                )
            last_idx_by_text[text_norm] = assigned_end_idx
            last_end_time[text_norm] = line.end_time

    return adjusted_lines


def _enforce_monotonic_line_starts_whisper(
    mapped_lines: List[Line],
    all_words: List[TranscriptionWord],
) -> List[Line]:
    """Ensure line starts are monotonic by shifting backwards lines forward."""
    from .models import Line, Word

    prev_start = None
    prev_end = None
    monotonic_lines: List[Line] = []
    for line in mapped_lines:
        if not line.words:
            monotonic_lines.append(line)
            continue

        if prev_start is not None and line.start_time < prev_start:
            required_time = (prev_end or line.start_time) + 0.01
            start_idx = next(
                (idx for idx, ww in enumerate(all_words) if ww.start >= required_time),
                None,
            )
            # Only snap to Whisper words if they're nearby; otherwise
            # just shift the line forward to avoid jumping across large
            # instrumental gaps.
            if start_idx is not None and (
                all_words[start_idx].start - required_time <= 10.0
            ):
                adjusted_words_2: List[Word] = []
                for word_idx, w in enumerate(line.words):
                    new_idx = min(start_idx + word_idx, len(all_words) - 1)
                    ww = all_words[new_idx]
                    adjusted_words_2.append(
                        Word(
                            text=w.text,
                            start_time=ww.start,
                            end_time=ww.end,
                            singer=w.singer,
                        )
                    )
                line = Line(words=adjusted_words_2, singer=line.singer)
            else:
                shift = required_time - line.start_time
                shifted_words: List[Word] = [
                    Word(
                        text=w.text,
                        start_time=w.start_time + shift,
                        end_time=w.end_time + shift,
                        singer=w.singer,
                    )
                    for w in line.words
                ]
                line = Line(words=shifted_words, singer=line.singer)

        monotonic_lines.append(line)
        if line.words:
            prev_start = line.start_time
            prev_end = line.end_time

    return monotonic_lines


def _resolve_line_overlaps(lines: List[Line]) -> List[Line]:
    """Ensure consecutive lines never overlap in time.

    When line *i* ends after line *i+1* starts, the last word of line *i* is
    trimmed so that line *i* ends exactly when line *i+1* begins.
    """
    from .models import Line as LineModel, Word

    resolved: List[Line] = list(lines)
    for i in range(len(resolved) - 1):
        cur = resolved[i]
        nxt = resolved[i + 1]
        if not cur.words or not nxt.words:
            continue
        if cur.end_time > nxt.start_time:
            gap_point = nxt.start_time
            # Trim the current line's last word(s) so it ends at gap_point
            new_words: List[Word] = []
            for w in cur.words:
                if w.start_time >= gap_point:
                    # This word starts at or after the next line – shrink it
                    new_words.append(
                        Word(
                            text=w.text,
                            start_time=gap_point - 0.01,
                            end_time=gap_point,
                            singer=w.singer,
                        )
                    )
                elif w.end_time > gap_point:
                    new_words.append(
                        Word(
                            text=w.text,
                            start_time=w.start_time,
                            end_time=gap_point,
                            singer=w.singer,
                        )
                    )
                else:
                    new_words.append(w)
            resolved[i] = LineModel(words=new_words, singer=cur.singer)
    return resolved


def align_lrc_text_to_whisper_timings(  # noqa: C901
    lines: List[Line],
    vocals_path: str,
    language: Optional[str] = None,
    model_size: str = "base",
    aggressive: bool = False,
    temperature: float = 0.0,
    min_similarity: float = 0.15,
    audio_features: Optional[AudioFeatures] = None,
    lenient_vocal_activity_threshold: float = 0.3,
    lenient_activity_bonus: float = 0.4,
    low_word_confidence_threshold: float = 0.5,
) -> Tuple[List[Line], List[str], Dict[str, float]]:
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

    epitran_lang = _whisper_lang_to_epitran(detected_lang)
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
        _get_ipa(ww.text, epitran_lang)
    for lw in lrc_words:
        _get_ipa(lw["text"], epitran_lang)

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


def _compute_dtw_alignment_metrics(
    lines: List[Line],
    lrc_words: List[Dict],
    alignments_map: Dict[int, Tuple[TranscriptionWord, float]],
) -> Dict[str, float]:
    if not lrc_words:
        return {"matched_ratio": 0.0, "avg_similarity": 0.0, "line_coverage": 0.0}

    total_words = len(lrc_words)
    matched_words = len(alignments_map)
    matched_ratio = matched_words / total_words if total_words else 0.0

    total_similarity = 0.0
    for _, (_ww, sim) in alignments_map.items():
        total_similarity += sim
    avg_similarity = total_similarity / matched_words if matched_words else 0.0

    total_lines = sum(1 for line in lines if line.words)
    matched_lines = {
        lrc_words[lrc_idx]["line_idx"] for lrc_idx in alignments_map.keys()
    }
    line_coverage = len(matched_lines) / total_lines if total_lines > 0 else 0.0

    return {
        "matched_ratio": matched_ratio,
        "word_coverage": matched_ratio,
        "avg_similarity": avg_similarity,
        "line_coverage": line_coverage,
    }


def _retime_lines_from_dtw_alignments(
    lines: List[Line],
    lrc_words: List[Dict],
    alignments_map: Dict[int, Tuple[TranscriptionWord, float]],
    min_word_duration: float = 0.05,
) -> Tuple[List[Line], List[str]]:

    aligned_by_line: Dict[int, List[Tuple[int, TranscriptionWord]]] = {}
    for lrc_idx, (ww, _sim) in alignments_map.items():
        lw = lrc_words[lrc_idx]
        aligned_by_line.setdefault(lw["line_idx"], []).append((lw["word_idx"], ww))

    retimed_lines: List[Line] = []
    corrections: List[str] = []

    for line_idx, line in enumerate(lines):
        if not line.words:
            retimed_lines.append(line)
            continue

        matches = aligned_by_line.get(line_idx, [])
        if not matches:
            retimed_lines.append(line)
            continue

        matches.sort(key=lambda item: item[0])
        target_duration = max(line.end_time - line.start_time, min_word_duration)
        tuple_matches = [(word_idx, (ww.start, ww.end)) for word_idx, ww in matches]
        retimed_line = _redistribute_word_timings_to_line(
            line,
            tuple_matches,
            target_duration=target_duration,
            min_word_duration=min_word_duration,
        )
        retimed_lines.append(retimed_line)
        corrections.append(f"DTW retimed line {line_idx} from matched words")

    return retimed_lines, corrections


def _redistribute_word_timings_to_line(
    line: Line,
    matches: List[Tuple[int, Tuple[float, float]]],
    target_duration: float,
    min_word_duration: float,
    line_start: Optional[float] = None,
    max_gap: float = 0.4,
) -> Line:
    """Redistribute word timings based on Whisper durations within the line."""
    from .models import Line, Word

    if not line.words:
        return line

    min_word_duration = max(min_word_duration, 0.0)
    match_map = {
        word_idx: (start, end)
        for word_idx, (start, end) in matches
        if start is not None and end is not None
    }

    max_line_duration = min(max(4.0, len(line.words) * 0.6), 8.0)
    target_duration = min(target_duration, max_line_duration)
    min_weight = max(min_word_duration, 0.01)
    weights: List[float] = []
    for word_idx in range(len(line.words)):
        interval = match_map.get(word_idx)
        if interval:
            start, end = interval
            weight = max(end - start, min_weight)
        else:
            weight = min_weight
        weights.append(weight)

    total_weight = sum(weights) or 1.0
    durations = [
        (weights[i] / total_weight) * target_duration for i in range(len(line.words))
    ]
    duration_sum = sum(durations)
    if durations:
        durations[-1] += target_duration - duration_sum

    max_word_duration = min(3.0, target_duration * 0.5)
    durations = _cap_word_durations(durations, target_duration, max_word_duration)

    current = line_start if line_start is not None else line.start_time
    new_words: List[Word] = []
    for idx, word in enumerate(line.words):
        duration = durations[idx]
        start_time = current
        end_time = start_time + duration
        if idx == len(line.words) - 1:
            end_time = line.start_time + target_duration
            if end_time <= start_time:
                end_time = start_time + max(min_word_duration, 0.01)
        new_words.append(
            Word(
                text=word.text,
                start_time=start_time,
                end_time=end_time,
                singer=word.singer,
            )
        )
        current = end_time

    adjusted_words = _clamp_word_gaps(new_words, max_gap)
    clamped_line = Line(words=adjusted_words, singer=line.singer)
    scaled_line = _scale_line_to_duration(
        clamped_line,
        target_duration=target_duration,
    )
    target_start = line_start if line_start is not None else scaled_line.start_time
    offset = target_start - scaled_line.start_time
    adjusted_scaled_words = [
        Word(
            text=w.text,
            start_time=w.start_time + offset,
            end_time=w.end_time + offset,
            singer=w.singer,
        )
        for w in scaled_line.words
    ]
    return Line(words=adjusted_scaled_words, singer=line.singer)


def _clamp_word_gaps(words: List[Word], max_gap: float) -> List[Word]:
    if not words or max_gap is None:
        return words
    adjusted: List[Word] = []
    total_shift = 0.0
    adjusted.append(
        Word(
            text=words[0].text,
            start_time=words[0].start_time,
            end_time=words[0].end_time,
            singer=words[0].singer,
        )
    )
    for current in words[1:]:
        prev = adjusted[-1]
        shifted_start = current.start_time - total_shift
        gap = shifted_start - prev.end_time
        shift = 0.0
        if gap > max_gap:
            shift = gap - max_gap
        total_shift += shift
        duration = current.end_time - current.start_time
        new_start = shifted_start - shift
        new_end = new_start + duration
        adjusted.append(
            Word(
                text=current.text,
                start_time=new_start,
                end_time=new_end,
                singer=current.singer,
            )
        )
    return adjusted


def _cap_word_durations(
    durations: List[float], total_duration: float, max_word_duration: float
) -> List[float]:
    if not durations:
        return durations
    capped = []
    remainder = total_duration
    for duration in durations:
        limit = min(max_word_duration, remainder - 0.01 * len(durations))
        capped_value = min(duration, limit)
        capped.append(capped_value)
        remainder -= capped_value
    if remainder > 0 and capped:
        capped[-1] += remainder
    return capped


def _align_dtw_whisper_with_data(
    lines: List[Line],
    whisper_words: List[TranscriptionWord],
    language: str = "fra-Latn",
    min_similarity: float = 0.4,
    silence_regions: Optional[List[Tuple[float, float]]] = None,
    audio_features: Optional[AudioFeatures] = None,
    lenient_vocal_activity_threshold: float = 0.3,
    lenient_activity_bonus: float = 0.4,
    low_word_confidence_threshold: float = 0.5,
) -> Tuple[
    List[Line],
    List[str],
    Dict[str, float],
    List[Dict],
    Dict[int, Tuple[TranscriptionWord, float]],
]:
    """Align LRC to Whisper using DTW and return alignment data for confidence gating."""
    if not lines or not whisper_words:
        return (
            lines,
            [],
            {"matched_ratio": 0.0, "avg_similarity": 0.0, "line_coverage": 0.0},
            [],
            {},
        )

    if audio_features:
        whisper_words, _ = _fill_vocal_activity_gaps(
            whisper_words, audio_features, lenient_vocal_activity_threshold
        )

    lrc_words = _extract_lrc_words(lines)
    if not lrc_words:
        return (
            lines,
            [],
            {"matched_ratio": 0.0, "avg_similarity": 0.0, "line_coverage": 0.0},
            [],
            {},
        )

    # Pre-compute IPA
    logger.debug(f"DTW: Pre-computing IPA for {len(whisper_words)} Whisper words...")
    for ww in whisper_words:
        _get_ipa(ww.text, language)
    for lw in lrc_words:
        _get_ipa(lw["text"], language)

    logger.debug(
        f"DTW: Building cost matrix ({len(lrc_words)} x {len(whisper_words)})..."
    )
    phonetic_costs = _compute_phonetic_costs(
        lrc_words, whisper_words, language, min_similarity
    )

    # Run DTW
    logger.debug("DTW: Running alignment...")
    use_silence = silence_regions or []
    try:
        from fastdtw import fastdtw  # type: ignore

        lrc_times = np.array([lw["start"] for lw in lrc_words])
        whisper_times = np.array([ww.start for ww in whisper_words])

        lrc_seq = np.column_stack([np.arange(len(lrc_words)), lrc_times])
        whisper_seq = np.column_stack([np.arange(len(whisper_words)), whisper_times])

        def dtw_dist(a, b):
            i, lrc_t = int(a[0]), a[1]
            j, whisper_t = int(b[0]), b[1]
            phon_cost = phonetic_costs[(i, j)]

            # Leniency mechanism: if Whisper word has low confidence but there is vocal activity,
            # be more lenient about phonetic mismatch.
            if (
                audio_features
                and whisper_words[j].probability < low_word_confidence_threshold
            ):
                # Check activity around the whisper word
                w_start = whisper_words[j].start
                w_end = whisper_words[j].end
                vocal_activity = _check_vocal_activity_in_range(
                    w_start, w_end, audio_features
                )
                if vocal_activity > lenient_vocal_activity_threshold:
                    phon_cost = max(0.0, phon_cost - lenient_activity_bonus)

            time_diff = abs(whisper_t - lrc_t)
            time_penalty = min(time_diff / 20.0, 1.0)
            gap_start = min(lrc_t, whisper_t)
            gap_end = max(lrc_t, whisper_t)
            silence_overlap = _compute_silence_overlap(gap_start, gap_end, use_silence)
            silence_penalty = min(silence_overlap / 2.0, 1.0)
            if _is_time_in_silence(whisper_t, use_silence):
                silence_penalty = max(silence_penalty, 0.8)
            activity_penalty = 0.0
            if audio_features and gap_end - gap_start > 0.5:
                activity = _check_vocal_activity_in_range(
                    gap_start, gap_end, audio_features
                )
                non_silent = max(gap_end - gap_start - silence_overlap, 0.0)
                if activity > 0.5 and non_silent > 0.5:
                    activity_penalty = min(activity, 1.0)
            return (
                0.5 * phon_cost
                + 0.2 * time_penalty
                + 0.2 * silence_penalty
                + 0.1 * activity_penalty
            )

        _distance, path = fastdtw(lrc_seq, whisper_seq, dist=dtw_dist)

    except ImportError:
        logger.warning("fastdtw not available, falling back to greedy alignment")
        return (
            lines,
            [],
            {"matched_ratio": 0.0, "avg_similarity": 0.0, "line_coverage": 0.0},
            lrc_words,
            {},
        )

    alignments_map = _extract_alignments_from_path_base(
        path, lrc_words, whisper_words, language, min_similarity
    )

    aligned_lines, corrections = _apply_dtw_alignments_base(
        lines, lrc_words, alignments_map
    )
    metrics = _compute_dtw_alignment_metrics(lines, lrc_words, alignments_map)

    logger.info(f"DTW alignment complete: {len(corrections)} lines modified")
    return aligned_lines, corrections, metrics, lrc_words, alignments_map


def align_dtw_whisper(
    lines: List[Line],
    whisper_words: List[TranscriptionWord],
    language: str = "fra-Latn",
    min_similarity: float = 0.4,
) -> Tuple[List[Line], List[str], Dict[str, float]]:
    """Align LRC to Whisper using Dynamic Time Warping."""
    lrc_words = _extract_lrc_words_base(lines)
    if not lrc_words:
        return (
            lines,
            [],
            {"matched_ratio": 0.0, "avg_similarity": 0.0, "line_coverage": 0.0},
        )

    # Pre-compute IPA
    for ww in whisper_words:
        _get_ipa(ww.text, language)
    for lw in lrc_words:
        _get_ipa(lw["text"], language)

    phonetic_costs = _compute_phonetic_costs_base(
        lrc_words, whisper_words, language, min_similarity
    )

    # Simple greedy alignment if fastdtw missing
    try:
        from fastdtw import fastdtw  # type: ignore

        lrc_times = np.array([lw["start"] for lw in lrc_words])
        whisper_times = np.array([ww.start for ww in whisper_words])
        lrc_seq = np.column_stack([np.arange(len(lrc_words)), lrc_times])
        whisper_seq = np.column_stack([np.arange(len(whisper_words)), whisper_times])

        def dtw_dist(a, b):
            i, lrc_t = int(a[0]), a[1]
            j, whisper_t = int(b[0]), b[1]
            phon_cost = phonetic_costs[(i, j)]
            time_diff = abs(whisper_t - lrc_t)
            time_penalty = min(time_diff / 20.0, 1.0)
            return 0.7 * phon_cost + 0.3 * time_penalty

        distance, path = fastdtw(lrc_seq, whisper_seq, dist=dtw_dist)
    except ImportError:
        logger.warning("fastdtw not available, falling back to greedy alignment")
        return (
            lines,
            [],
            {"matched_ratio": 0.0, "avg_similarity": 0.0, "line_coverage": 0.0},
        )

    alignments_map = _extract_alignments_from_path_base(
        path, lrc_words, whisper_words, language, min_similarity
    )

    aligned_lines, corrections = _apply_dtw_alignments_base(
        lines, lrc_words, alignments_map
    )
    metrics = _compute_dtw_alignment_metrics(lines, lrc_words, alignments_map)

    return aligned_lines, corrections, metrics


def correct_timing_with_whisper(  # noqa: C901
    lines: List[Line],
    vocals_path: str,
    language: Optional[str] = None,
    model_size: str = "base",
    aggressive: bool = False,
    temperature: float = 0.0,
    trust_lrc_threshold: float = 1.0,
    correct_lrc_threshold: float = 1.5,
    force_dtw: bool = False,
    audio_features: Optional[AudioFeatures] = None,
    lenient_vocal_activity_threshold: float = 0.3,
    lenient_activity_bonus: float = 0.4,
    low_word_confidence_threshold: float = 0.5,
) -> Tuple[List[Line], List[str], Dict[str, float]]:
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
    epitran_lang = _whisper_lang_to_epitran(detected_lang)
    logger.debug(
        f"Using epitran language: {epitran_lang} (from Whisper: {detected_lang})"
    )

    # Pre-compute IPA for Whisper words
    logger.debug(f"Pre-computing IPA for {len(all_words)} Whisper words...")
    for w in all_words:
        _get_ipa(w.text, epitran_lang)

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
