"""Tokenization and DTW path helpers for Whisper/LRC alignment."""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Set, Tuple

import numpy as np

from ..alignment.timing_models import TranscriptionWord
from ...phonetic_utils import (
    _get_ipa,
    _get_ipa_segs,
    _get_panphon_distance,
    _is_vowel,
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
        sim = _phoneme_similarity_from_ipa(ipa1, ipa2, language)
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
            for j in range(
                whisper_idx,
                min(whisper_idx + 12, len(whisper_syllables)),
            ):
                cost = syllable_cost(lrc_idx, j)
                if cost < best_cost:
                    best_cost = cost
                    best_idx = j
            path.append((lrc_idx, best_idx))
            whisper_idx = best_idx
        return path
