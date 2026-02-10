"""Phonetic and DTW-based alignment logic for Whisper integration."""

from typing import List, Optional, Tuple, Dict

from ....utils.logging import get_logger
from ... import models
from ..alignment import timing_models
from ... import phonetic_utils
from . import whisper_phonetic_paths as _paths
from . import whisper_phonetic_tokens as _tokens

logger = get_logger(__name__)


_build_phoneme_tokens_from_lrc_words = _tokens._build_phoneme_tokens_from_lrc_words
_build_phoneme_tokens_from_whisper_words = (
    _tokens._build_phoneme_tokens_from_whisper_words
)
_phoneme_similarity_from_ipa = _tokens._phoneme_similarity_from_ipa


def align_lyrics_to_transcription(
    lines: List[models.Line],
    transcription: List[timing_models.TranscriptionSegment],
    min_similarity: float = 0.4,
    max_time_shift: float = 10.0,
    language: str = "fra-Latn",
) -> Tuple[List[models.Line], List[str]]:
    """Align lyrics lines to Whisper transcription using fuzzy matching."""
    if not lines or not transcription:
        return lines, []

    aligned_lines: List[models.Line] = []
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

            similarity = phonetic_utils._text_similarity(
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
                        models.Word(
                            text=word.text,
                            start_time=new_start,
                            end_time=new_end,
                            singer=word.singer,
                        )
                    )

                aligned_line = models.Line(words=new_words, singer=line.singer)
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
    sorted_whisper: List[timing_models.TranscriptionWord],
    used_indices: set,
    min_similarity: float,
    max_time_shift: float,
    language: str,
    min_index: int = 0,
    whisper_percentiles: Optional[List[float]] = None,
    expected_percentile: float = 0.0,
    order_weight: float = 0.35,
    time_weight: float = 0.25,
) -> Tuple[Optional[timing_models.TranscriptionWord], Optional[int], float]:
    """Find the best matching Whisper word for an LRC word."""
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

        basic_sim = phonetic_utils._text_similarity_basic(lrc_text, ww.text, language)
        if basic_sim < 0.25:
            continue

        similarity = phonetic_utils._phonetic_similarity(lrc_text, ww.text, language)

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
    lines: List[models.Line],
    whisper_words: List[timing_models.TranscriptionWord],
    min_similarity: float = 0.5,
    max_time_shift: float = 5.0,
    language: str = "fra-Latn",
) -> Tuple[List[models.Line], List[str]]:
    """Align individual LRC words to Whisper word timestamps using phonetic matching."""
    if not lines or not whisper_words:
        return lines, []

    # Pre-compute IPA for all Whisper words
    logger.debug(f"Pre-computing IPA for {len(whisper_words)} Whisper words...")
    for ww in whisper_words:
        phonetic_utils._get_ipa(ww.text, language)

    sorted_whisper = sorted(whisper_words, key=lambda w: w.start)
    total_whisper = len(sorted_whisper)
    whisper_percentiles = [i / max(1, total_whisper - 1) for i in range(total_whisper)]
    used_whisper_indices: set = set()
    aligned_lines: List[models.Line] = []
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

        new_words: List[models.Word] = []
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
                        models.Word(
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

        aligned_lines.append(models.Line(words=new_words, singer=line.singer))
        if line_corrections > 0:
            line_text = " ".join(w.text for w in line.words)[:40]
            corrections.append(
                f'Line aligned {line_corrections} word(s): "{line_text}..."'
            )

    return aligned_lines, corrections


def _assess_lrc_quality(
    lines: List[models.Line],
    whisper_words: List[timing_models.TranscriptionWord],
    language: str = "fra-Latn",
    tolerance: float = 1.5,
) -> Tuple[float, List[Tuple[int, float, float]]]:
    """Assess LRC timing quality by comparing against Whisper."""
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

            sim = phonetic_utils._phonetic_similarity(lrc_text, ww.text, language)
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


def _extract_lrc_words(lines: List[models.Line]) -> List[Dict]:
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


_compute_phonetic_costs = _paths._compute_phonetic_costs
_compute_phonetic_costs_unbounded = _paths._compute_phonetic_costs_unbounded
_extract_best_alignment_map = _paths._extract_best_alignment_map


def _extract_lrc_words_all(lines: List[models.Line]) -> List[Dict]:
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


_build_dtw_path = _paths._build_dtw_path


def _build_phoneme_dtw_path(
    lrc_phonemes: List[Dict],
    whisper_phonemes: List[Dict],
    language: str,
) -> List[Tuple[int, int]]:
    """Build DTW path between phoneme tokens."""
    return _paths._build_phoneme_dtw_path(
        lrc_phonemes,
        whisper_phonemes,
        language,
        _phoneme_similarity_from_ipa,
    )


_build_syllable_tokens_from_phonemes = _tokens._build_syllable_tokens_from_phonemes
_make_syllable_from_tokens = _tokens._make_syllable_from_tokens


def _build_syllable_dtw_path(
    lrc_syllables: List[Dict],
    whisper_syllables: List[Dict],
    language: str,
) -> List[Tuple[int, int]]:
    """Build DTW path between syllable units."""
    return _paths._build_syllable_dtw_path(
        lrc_syllables,
        whisper_syllables,
        language,
        _phoneme_similarity_from_ipa,
    )
