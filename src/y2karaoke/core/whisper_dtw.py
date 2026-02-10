"""Dynamic Time Warping (DTW) alignment for Whisper and LRC."""

import logging
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

import numpy as np

from .models import Line, Word
from .timing_models import TranscriptionWord
from . import timing_models
from . import phonetic_utils
from .phonetic_utils import (
    _get_ipa,
    _phonetic_similarity,
)
from .whisper_dtw_tokens import _LineMappingContext  # noqa: F401

logger = logging.getLogger(__name__)


def _extract_lrc_words_base(lines: List[Line]) -> List[Dict]:
    """Extract all LRC words with their line indices (base version)."""
    lrc_words = []
    for line_idx, line in enumerate(lines):
        for word_idx, word in enumerate(line.words):
            text = word.text.strip()
            if text:
                lrc_words.append(
                    {
                        "text": text,
                        "start": word.start_time,
                        "line_idx": line_idx,
                        "word_idx": word_idx,
                        "word": word,
                    }
                )
    return lrc_words


def _compute_phonetic_costs_base(
    lrc_words: List[Dict],
    whisper_words: List[TranscriptionWord],
    language: str,
    min_similarity: float,
) -> Dict[Tuple[int, int], float]:
    """Compute sparse phonetic cost matrix for DTW (base version)."""
    phonetic_costs = defaultdict(lambda: 1.0)  # Default high cost

    for i, lw in enumerate(lrc_words):
        lrc_time = lw["start"]
        for j, ww in enumerate(whisper_words):
            # Only consider words within 20s of each other
            time_diff = abs(ww.start - lrc_time)
            if time_diff > 20:
                continue

            sim = _phonetic_similarity(lw["text"], ww.text, language)
            if sim >= min_similarity:
                phonetic_costs[(i, j)] = 1.0 - sim

    return phonetic_costs


def _extract_alignments_from_path_base(
    path: List[Tuple[int, int]],
    lrc_words: List[Dict],
    whisper_words: List[TranscriptionWord],
    language: str,
    min_similarity: float,
) -> Dict[int, Tuple[TranscriptionWord, float]]:
    """Extract validated alignments from DTW path (base version)."""
    alignments_map = {}  # lrc_word_idx -> whisper_word

    for lrc_idx, whisper_idx in path:
        if lrc_idx not in alignments_map:
            # Only take first match for each LRC word
            ww = whisper_words[whisper_idx]
            lw = lrc_words[lrc_idx]
            # Verify it's a reasonable match
            sim = _phonetic_similarity(lw["text"], ww.text, language)
            if sim >= min_similarity:
                alignments_map[lrc_idx] = (ww, sim)

    return alignments_map


def _apply_dtw_alignments_base(
    lines: List[Line],
    lrc_words: List[Dict],
    alignments_map: Dict[int, Tuple[TranscriptionWord, float]],
) -> Tuple[List[Line], List[str]]:
    """Apply DTW alignments to create corrected lines (base version)."""
    corrections = []
    aligned_lines = []

    for line_idx, line in enumerate(lines):
        new_words = []
        line_corrections = 0

        for word_idx, word in enumerate(line.words):
            # Find this word in lrc_words
            lrc_word_idx = None
            for i, lw in enumerate(lrc_words):
                if lw["line_idx"] == line_idx and lw["word_idx"] == word_idx:
                    lrc_word_idx = i
                    break

            if lrc_word_idx is not None and lrc_word_idx in alignments_map:
                ww, sim = alignments_map[lrc_word_idx]
                time_shift = ww.start - word.start_time

                # Only correct if shift is significant (> 1s)
                if abs(time_shift) > 1.0:
                    new_words.append(
                        Word(
                            text=word.text,
                            start_time=ww.start,
                            end_time=ww.end,
                            singer=word.singer,
                        )
                    )
                    line_corrections += 1
                else:
                    new_words.append(word)
            else:
                new_words.append(word)

        aligned_lines.append(Line(words=new_words, singer=line.singer))

        if line_corrections > 0:
            line_text = " ".join(w.text for w in line.words)[:40]
            corrections.append(
                f'DTW aligned {line_corrections} word(s) in line {line_idx}: "{line_text}..."'
            )

    return aligned_lines, corrections


def align_dtw_whisper_base(
    lines: List[Line],
    whisper_words: List[TranscriptionWord],
    language: str = "fra-Latn",
    min_similarity: float = 0.4,
) -> Tuple[List[Line], List[str]]:
    """Align LRC to Whisper using Dynamic Time Warping (base version)."""
    if not lines or not whisper_words:
        return lines, []

    lrc_words = _extract_lrc_words_base(lines)
    if not lrc_words:
        return lines, []

    # Pre-compute IPA
    logger.debug(f"DTW: Pre-computing IPA for {len(whisper_words)} Whisper words...")
    for ww in whisper_words:
        _get_ipa(ww.text, language)
    for lw in lrc_words:
        _get_ipa(lw["text"], language)

    logger.debug(
        f"DTW: Building cost matrix ({len(lrc_words)} x {len(whisper_words)})..."
    )
    phonetic_costs = _compute_phonetic_costs_base(
        lrc_words, whisper_words, language, min_similarity
    )

    # Run DTW
    logger.debug("DTW: Running alignment...")
    try:
        from fastdtw import fastdtw

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
        return lines, []

    alignments_map = _extract_alignments_from_path_base(
        path, lrc_words, whisper_words, language, min_similarity
    )

    aligned_lines, corrections = _apply_dtw_alignments_base(
        lines, lrc_words, alignments_map
    )

    logger.info(f"DTW alignment complete: {len(corrections)} lines modified")
    return aligned_lines, corrections


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
    from .whisper_utils import _redistribute_word_timings_to_line

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


def _align_dtw_whisper_with_data(
    lines: List[Line],
    whisper_words: List[TranscriptionWord],
    language: str = "fra-Latn",
    min_similarity: float = 0.4,
    silence_regions: Optional[List[Tuple[float, float]]] = None,
    audio_features: Optional[timing_models.AudioFeatures] = None,
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
    from .whisper_integration import _fill_vocal_activity_gaps
    from .audio_analysis import (
        _check_vocal_activity_in_range,
        _compute_silence_overlap,
        _is_time_in_silence,
    )
    from . import whisper_phonetic_dtw

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

    lrc_words = whisper_phonetic_dtw._extract_lrc_words(lines)
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
        phonetic_utils._get_ipa(ww.text, language)
    for lw in lrc_words:
        phonetic_utils._get_ipa(lw["text"], language)

    logger.debug(
        f"DTW: Building cost matrix ({len(lrc_words)} x {len(whisper_words)})..."
    )
    phonetic_costs = whisper_phonetic_dtw._compute_phonetic_costs(
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
        phonetic_utils._get_ipa(ww.text, language)
    for lw in lrc_words:
        phonetic_utils._get_ipa(lw["text"], language)

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

        _distance, path = fastdtw(lrc_seq, whisper_seq, dist=dtw_dist)
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
