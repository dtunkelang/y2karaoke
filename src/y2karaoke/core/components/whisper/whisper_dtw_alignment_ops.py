"""Reusable DTW alignment operations for Whisper/LRC mapping."""

from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

from ...models import Line, Word
from ..alignment.timing_models import TranscriptionWord


def empty_dtw_metrics() -> Dict[str, float]:
    return {
        "matched_ratio": 0.0,
        "word_coverage": 0.0,
        "avg_similarity": 0.0,
        "line_coverage": 0.0,
        "phonetic_similarity_coverage": 0.0,
        "high_similarity_ratio": 0.0,
        "exact_match_ratio": 0.0,
        "unmatched_ratio": 1.0,
    }


def extract_lrc_words_base(lines: List[Line]) -> List[Dict]:
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


def compute_phonetic_costs_base(
    lrc_words: List[Dict],
    whisper_words: List[TranscriptionWord],
    language: str,
    min_similarity: float,
    phonetic_similarity_fn: Callable[[str, str, str], float],
) -> Dict[Tuple[int, int], float]:
    """Compute sparse phonetic cost matrix for DTW (base version)."""
    phonetic_costs = defaultdict(lambda: 1.0)  # Default high cost
    similarity_cache: Dict[Tuple[str, str], float] = {}

    def cached_similarity(left: str, right: str) -> float:
        key = (left, right)
        if key in similarity_cache:
            return similarity_cache[key]
        sim = phonetic_similarity_fn(left, right, language)
        similarity_cache[key] = sim
        return sim

    for i, lw in enumerate(lrc_words):
        lrc_time = lw["start"]
        for j, ww in enumerate(whisper_words):
            time_diff = abs(ww.start - lrc_time)
            if time_diff > 20:
                continue

            sim = cached_similarity(lw["text"], ww.text)
            if sim >= min_similarity:
                phonetic_costs[(i, j)] = 1.0 - sim

    return phonetic_costs


def extract_alignments_from_path_base(
    path: List[Tuple[int, int]],
    lrc_words: List[Dict],
    whisper_words: List[TranscriptionWord],
    language: str,
    min_similarity: float,
    phonetic_similarity_fn: Callable[[str, str, str], float],
    precomputed_similarity: Optional[Dict[Tuple[int, int], float]] = None,
) -> Dict[int, Tuple[TranscriptionWord, float]]:
    """Extract validated alignments from DTW path (base version)."""
    alignments_map = {}

    def _norm_token(text: str) -> str:
        return "".join(ch for ch in text.lower() if ch.isalnum())

    for lrc_idx, whisper_idx in path:
        if lrc_idx in alignments_map:
            continue
        ww = whisper_words[whisper_idx]
        lw = lrc_words[lrc_idx]
        lrc_norm = _norm_token(lw["text"])
        whisper_norm = _norm_token(ww.text)
        exact_text_match = (
            bool(lrc_norm)
            and bool(whisper_norm)
            and len(lrc_norm) >= 2
            and len(whisper_norm) >= 2
            and lrc_norm == whisper_norm
        )
        sim = None
        if precomputed_similarity is not None:
            sim = precomputed_similarity.get((lrc_idx, whisper_idx))
        if sim is None:
            sim = (
                1.0
                if exact_text_match
                else phonetic_similarity_fn(lw["text"], ww.text, language)
            )
        if sim >= min_similarity or exact_text_match:
            alignments_map[lrc_idx] = (ww, sim)

    return alignments_map


def apply_dtw_alignments_base(
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
            lrc_word_idx = None
            for i, lw in enumerate(lrc_words):
                if lw["line_idx"] == line_idx and lw["word_idx"] == word_idx:
                    lrc_word_idx = i
                    break

            if lrc_word_idx is not None and lrc_word_idx in alignments_map:
                ww, sim = alignments_map[lrc_word_idx]
                time_shift = ww.start - word.start_time

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


def compute_dtw_alignment_metrics(
    lines: List[Line],
    lrc_words: List[Dict],
    alignments_map: Dict[int, Tuple[TranscriptionWord, float]],
) -> Dict[str, float]:
    if not lrc_words:
        return empty_dtw_metrics()

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

    high_similarity_matches = 0
    exact_matches = 0
    for lrc_idx, (ww, sim) in alignments_map.items():
        if sim >= 0.75:
            high_similarity_matches += 1
        lrc_norm = "".join(
            ch for ch in lrc_words[lrc_idx]["text"].lower() if ch.isalpha()
        )
        whisper_norm = "".join(ch for ch in ww.text.lower() if ch.isalpha())
        if lrc_norm and whisper_norm and lrc_norm == whisper_norm:
            exact_matches += 1

    high_similarity_ratio = (
        high_similarity_matches / matched_words if matched_words else 0.0
    )
    exact_match_ratio = exact_matches / matched_words if matched_words else 0.0
    phonetic_similarity_coverage = matched_ratio * avg_similarity

    return {
        "matched_ratio": matched_ratio,
        "word_coverage": matched_ratio,
        "avg_similarity": avg_similarity,
        "line_coverage": line_coverage,
        "phonetic_similarity_coverage": phonetic_similarity_coverage,
        "high_similarity_ratio": high_similarity_ratio,
        "exact_match_ratio": exact_match_ratio,
        "unmatched_ratio": 1.0 - matched_ratio,
    }


def retime_lines_from_dtw_alignments(
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
