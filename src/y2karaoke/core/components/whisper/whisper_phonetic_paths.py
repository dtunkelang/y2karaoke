"""DTW and cost helpers for Whisper phonetic alignment."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from ....utils.logging import get_logger
from ... import phonetic_utils
from ..alignment import timing_models

logger = get_logger(__name__)


def _compute_phonetic_costs(
    lrc_words: List[Dict],
    whisper_words: List[timing_models.TranscriptionWord],
    language: str,
    min_similarity: float,
) -> Dict[Tuple[int, int], float]:
    """Compute sparse phonetic cost matrix for DTW."""
    phonetic_costs = defaultdict(lambda: 1.0)
    for i, lw in enumerate(lrc_words):
        lrc_time = lw["start"]
        for j, ww in enumerate(whisper_words):
            if ww.text == "[VOCAL]":
                phonetic_costs[(i, j)] = 0.5
                continue
            time_diff = abs(ww.start - lrc_time)
            if time_diff > 20:
                continue
            sim = phonetic_utils._phonetic_similarity(lw["text"], ww.text, language)
            if sim >= min_similarity:
                phonetic_costs[(i, j)] = 1.0 - sim
    return phonetic_costs


def _compute_phonetic_costs_unbounded(
    lrc_words: List[Dict],
    whisper_words: List[timing_models.TranscriptionWord],
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
            sim = phonetic_utils._phonetic_similarity(lw["text"], ww.text, language)
            if sim >= min_similarity:
                phonetic_costs[(i, j)] = 1.0 - sim
    return phonetic_costs


def _extract_best_alignment_map(
    path: List[Tuple[int, int]],
    lrc_words: List[Dict],
    whisper_words: List[timing_models.TranscriptionWord],
    language: str,
) -> Dict[int, Tuple[int, float]]:
    """Extract best whisper index per LRC word index from a DTW path."""
    alignments: Dict[int, Tuple[int, float]] = {}
    for lrc_idx, whisper_idx in path:
        lw = lrc_words[lrc_idx]
        ww = whisper_words[whisper_idx]
        sim = phonetic_utils._phonetic_similarity(lw["text"], ww.text, language)
        if lrc_idx not in alignments or sim > alignments[lrc_idx][1]:
            alignments[lrc_idx] = (whisper_idx, sim)
    return alignments


def _build_dtw_path(
    lrc_words: List[Dict],
    all_words: List[timing_models.TranscriptionWord],
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
                sim = phonetic_utils._phonetic_similarity(
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
    phoneme_similarity_fn,
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
        sim = phoneme_similarity_fn(ipa1, ipa2, language)
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


def _build_syllable_dtw_path(
    lrc_syllables: List[Dict],
    whisper_syllables: List[Dict],
    language: str,
    phoneme_similarity_fn,
) -> List[Tuple[int, int]]:
    """Build DTW path between syllable units."""
    cost_cache: Dict[Tuple[int, int], float] = {}
    n_lrc = max(len(lrc_syllables), 1)
    n_whisper = max(len(whisper_syllables), 1)

    def syllable_cost(i: int, j: int) -> float:
        key = (i, j)
        if key in cost_cache:
            return cost_cache[key]
        ipa1 = lrc_syllables[i]["ipa"]
        ipa2 = whisper_syllables[j]["ipa"]
        sim = phoneme_similarity_fn(ipa1, ipa2, language)
        phon_cost = 1.0 - sim
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
