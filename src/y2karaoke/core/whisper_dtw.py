"""Dynamic Time Warping (DTW) alignment for Whisper and LRC."""

import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Set, Sequence
from collections import defaultdict

import numpy as np

from .models import Line, Word
from .timing_models import TranscriptionWord
from .phonetic_utils import (
    _get_ipa,
    _phonetic_similarity,
)

logger = logging.getLogger(__name__)


@dataclass
class _LineMappingContext:
    """Shared mutable state for _map_lrc_words_to_whisper and its helpers."""

    all_words: List[TranscriptionWord]
    segments: Sequence[Any]
    word_segment_idx: Dict[int, int]
    language: str
    total_lrc_words: int
    total_whisper_words: int
    mapped_count: int = 0
    total_similarity: float = 0.0
    mapped_lines_set: Set[int] = field(default_factory=set)
    used_word_indices: Set[int] = field(default_factory=set)
    used_segments: Set[int] = field(default_factory=set)
    speech_blocks: List[Tuple[int, int]] = field(default_factory=list)
    next_word_idx_start: int = 0
    current_segment: int = 0
    current_block: int = 0
    last_line_start: float = float("-inf")
    prev_line_end: float = float("-inf")


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
