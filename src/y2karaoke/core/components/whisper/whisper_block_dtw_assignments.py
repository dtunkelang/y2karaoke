"""Block-scoped syllable DTW assignment helpers."""

import json
from typing import Dict, List, Set, Tuple

from ....utils.logging import get_logger
from ..alignment import timing_models
from . import whisper_phonetic_dtw, whisper_utils
from .whisper_assignment_trace import _json_safe_value
from .whisper_mapping_runtime_config import (
    WhisperMappingTraceConfig,
    load_whisper_mapping_trace_config,
)

logger = get_logger(__name__)


def _assign_lrc_lines_to_blocks(
    lrc_words: List[Dict],
    block_word_bags: List[Set[str]],
) -> List[int]:
    lrc_line_count = max((lw["line_idx"] for lw in lrc_words), default=-1) + 1
    lrc_line_block: List[int] = []
    for li in range(lrc_line_count):
        line_words_text = [
            whisper_utils._normalize_word(lw["text"])
            for lw in lrc_words
            if lw["line_idx"] == li
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

    for i in range(1, lrc_line_count):
        if lrc_line_block[i] < lrc_line_block[i - 1]:
            lrc_line_block[i] = lrc_line_block[i - 1]

    return lrc_line_block


def _build_block_word_bags(
    all_words: List[timing_models.TranscriptionWord],
    speech_blocks: List[Tuple[int, int]],
) -> List[Set[str]]:
    bags: List[Set[str]] = []
    for blk_start, blk_end in speech_blocks:
        bag: Set[str] = set()
        for wi in range(blk_start, blk_end + 1):
            nw = whisper_utils._normalize_word(all_words[wi].text)
            if nw:
                bag.add(nw)
        bags.append(bag)
    return bags


def _syl_to_block(
    syl_parent_idxs: Set[int], speech_blocks: List[Tuple[int, int]]
) -> int:
    if not syl_parent_idxs:
        return -1
    first_word = min(syl_parent_idxs)
    return whisper_utils._word_idx_to_block(first_word, speech_blocks)


def _group_syllables_by_block(
    syl_word_idxs: List[Set[int]],
    speech_blocks: List[Tuple[int, int]],
) -> Dict[int, List[int]]:
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
    all_words: List[timing_models.TranscriptionWord],
    language: str,
    trace_rows: List[Dict] | None = None,
    word_idx_to_line_idx: Dict[int, int] | None = None,
    trace_line_range: Tuple[int, int] | None = None,
) -> Dict[int, List[int]]:
    combined_assignments: Dict[int, Set[int]] = {}

    for blk_idx in range(len(speech_blocks)):
        blk_lrc_syl_idxs = block_lrc_syls.get(blk_idx, [])
        blk_whi_syl_idxs = block_whisper_syls.get(blk_idx, [])
        if not blk_lrc_syl_idxs or not blk_whi_syl_idxs:
            continue

        blk_lrc_syls_list = [lrc_syllables[i] for i in blk_lrc_syl_idxs]
        blk_whi_syls_list = [whisper_syllables[i] for i in blk_whi_syl_idxs]

        path = whisper_phonetic_dtw._build_syllable_dtw_path(
            blk_lrc_syls_list, blk_whi_syls_list, language
        )

        for local_li, local_wi in path:
            global_li = blk_lrc_syl_idxs[local_li]
            global_wi = blk_whi_syl_idxs[local_wi]
            lrc_token = lrc_syllables[global_li]
            whisper_token = whisper_syllables[global_wi]
            if trace_rows is not None and word_idx_to_line_idx is not None:
                line_indices = sorted(
                    {
                        word_idx_to_line_idx[word_idx]
                        for word_idx in lrc_token.get("word_idxs", set())
                        if word_idx in word_idx_to_line_idx
                    }
                )
                if line_indices and (
                    trace_line_range is None
                    or any(
                        trace_line_range[0] <= line_idx + 1 <= trace_line_range[1]
                        for line_idx in line_indices
                    )
                ):
                    trace_rows.append(
                        {
                            "block_index": blk_idx,
                            "lrc_syllable_index": global_li,
                            "lrc_word_idxs": sorted(lrc_token.get("word_idxs", set())),
                            "line_indices": [line_idx + 1 for line_idx in line_indices],
                            "whisper_syllable_index": global_wi,
                            "whisper_parent_idxs": sorted(
                                whisper_token.get("parent_idxs", set())
                            ),
                            "whisper_parent_texts": sorted(
                                whisper_token.get("parent_texts", set())
                            ),
                            "whisper_start": round(whisper_token.get("start", 0.0), 3),
                            "whisper_end": round(whisper_token.get("end", 0.0), 3),
                        }
                    )
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
    all_words: List[timing_models.TranscriptionWord],
    lrc_syllables: List[Dict],
    whisper_syllables: List[Dict],
    language: str,
    *,
    trace_config: WhisperMappingTraceConfig | None = None,
) -> Dict[int, List[int]]:
    resolved_trace_config = trace_config or load_whisper_mapping_trace_config()
    trace_path = resolved_trace_config.syllable_assignments_path
    trace_line_range = resolved_trace_config.line_range
    speech_blocks = whisper_utils._compute_speech_blocks(all_words)
    word_idx_to_line_idx = {idx: lw["line_idx"] for idx, lw in enumerate(lrc_words)}
    if len(speech_blocks) <= 1:
        path = whisper_phonetic_dtw._build_syllable_dtw_path(
            lrc_syllables, whisper_syllables, language
        )
        assignments = whisper_utils._build_word_assignments_from_syllable_path(
            path, lrc_syllables, whisper_syllables
        )
        if trace_path:
            rows = []
            for lrc_idx, whisper_idx in path:
                lrc_token = lrc_syllables[lrc_idx]
                line_indices = sorted(
                    {
                        word_idx_to_line_idx[word_idx]
                        for word_idx in lrc_token.get("word_idxs", set())
                        if word_idx in word_idx_to_line_idx
                    }
                )
                if line_indices and (
                    trace_line_range is None
                    or any(
                        trace_line_range[0] <= line_idx + 1 <= trace_line_range[1]
                        for line_idx in line_indices
                    )
                ):
                    whisper_token = whisper_syllables[whisper_idx]
                    rows.append(
                        {
                            "block_index": 0,
                            "lrc_syllable_index": lrc_idx,
                            "lrc_word_idxs": sorted(lrc_token.get("word_idxs", set())),
                            "line_indices": [line_idx + 1 for line_idx in line_indices],
                            "whisper_syllable_index": whisper_idx,
                            "whisper_parent_idxs": sorted(
                                whisper_token.get("parent_idxs", set())
                            ),
                            "whisper_parent_texts": sorted(
                                whisper_token.get("parent_texts", set())
                            ),
                            "whisper_start": round(whisper_token.get("start", 0.0), 3),
                            "whisper_end": round(whisper_token.get("end", 0.0), 3),
                        }
                    )
            with open(trace_path, "w", encoding="utf-8") as fh:
                json.dump(_json_safe_value({"rows": rows}), fh, indent=2)
        return assignments

    logger.debug(
        "Block-segmented DTW: %d speech blocks across %d Whisper words",
        len(speech_blocks),
        len(all_words),
    )

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

    block_whisper_syls = _group_syllables_by_block(
        whisper_syl_word_idxs,
        speech_blocks,
    )

    total_lrc_words = len(lrc_words)
    block_word_bags = _build_block_word_bags(all_words, speech_blocks)
    lrc_line_block = _assign_lrc_lines_to_blocks(lrc_words, block_word_bags)
    lrc_word_block = [lrc_line_block[lw["line_idx"]] for lw in lrc_words]

    n_blocks = len(speech_blocks)
    logger.debug(
        "Text-overlap block assignment: %s",
        {bi: sum(1 for b in lrc_line_block if b == bi) for bi in range(n_blocks)},
    )

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

    trace_rows: List[Dict] | None = [] if trace_path else None
    assignments = _run_per_block_dtw(
        speech_blocks,
        block_lrc_syls,
        block_whisper_syls,
        lrc_syllables,
        whisper_syllables,
        all_words,
        language,
        trace_rows=trace_rows,
        word_idx_to_line_idx=word_idx_to_line_idx,
        trace_line_range=trace_line_range,
    )
    if trace_path and trace_rows is not None:
        with open(trace_path, "w", encoding="utf-8") as fh:
            json.dump(_json_safe_value({"rows": trace_rows}), fh, indent=2)
    return assignments
