"""Dynamic Time Warping (DTW) alignment for Whisper and LRC."""

import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Set, Sequence
from collections import defaultdict

import numpy as np

from .models import Line, Word
from .timing_models import TranscriptionWord, TranscriptionSegment
from .phonetic_utils import (
    _get_ipa,
    _get_ipa_segs,
    _phonetic_similarity,
    _get_panphon_distance,
    _is_vowel,
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
            return key in cost_cache
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


def _segment_start(segment: Any) -> float:
    if hasattr(segment, "start"):
        return float(segment.start)
    if isinstance(segment, dict) and "start" in segment:
        return float(segment["start"])
    return 0.0


def _segment_end(segment: Any) -> float:
    if hasattr(segment, "end"):
        return float(segment.end)
    if isinstance(segment, dict) and "end" in segment:
        return float(segment["end"])
    return 0.0


def _get_segment_text(segment: Any) -> str:
    if hasattr(segment, "text"):
        return str(segment.text)
    if isinstance(segment, dict) and "text" in segment:
        return str(segment["text"])
    return ""


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
