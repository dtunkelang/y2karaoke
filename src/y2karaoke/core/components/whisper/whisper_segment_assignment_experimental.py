"""Parallel experimental segment assignment entry point.

This module exists to isolate larger redesigns of stalled-run candidate
generation, ownership, and distribution from the default live path.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple, cast

from ..alignment import timing_models
from . import whisper_blocks
from . import whisper_utils

_EXPERIMENTAL_ASSIGNER_MODE = "parallel_experimental"
_MIN_PLACEHOLDER_POSITIVE_LINE_RATIO = 0.15
_FUNCTION_WORDS = {
    "",
    "a",
    "ai",
    "au",
    "ce",
    "ces",
    "d",
    "dans",
    "de",
    "des",
    "du",
    "elle",
    "en",
    "est",
    "et",
    "il",
    "j",
    "je",
    "l",
    "la",
    "le",
    "les",
    "m",
    "ma",
    "me",
    "mes",
    "mon",
    "n",
    "ne",
    "ou",
    "pour",
    "qu",
    "que",
    "qui",
    "s",
    "se",
    "sur",
    "te",
    "toi",
    "ton",
    "tu",
    "un",
    "une",
    "y",
}
_NO_CONTENT_MIN_WORDS = 6
_NO_CONTENT_MAX_FORWARD_JUMP = 2


def _token_is_content(token: str) -> bool:
    return len(token) > 2 and token not in _FUNCTION_WORDS


def _row_scores(row: dict[str, object]) -> list[dict[str, object]]:
    raw_scores = row.get("scores")
    if not isinstance(raw_scores, list):
        return []
    return cast(list[dict[str, object]], raw_scores)


def _score_row_float(row: dict[str, object], key: str) -> float:
    value = row.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _should_keep_experimental_assignments(
    trace_rows: List[dict[str, object]] | None,
    line_count: int,
) -> bool:
    if not trace_rows or line_count <= 0:
        return False
    placeholder_positive_lines = 0
    for row in trace_rows:
        if any(
            _score_row_float(score_row, "placeholder_ratio") >= 0.5
            and _score_row_float(score_row, "score") > 0.0
            for score_row in _row_scores(row)
        ):
            placeholder_positive_lines += 1
    return (
        placeholder_positive_lines / max(line_count, 1)
        >= _MIN_PLACEHOLDER_POSITIVE_LINE_RATIO
    )


def _score_segment_for_line_experimental(  # noqa: C901
    line_words: List[str],
    seg_bag: List[str],
) -> dict[str, float | int]:
    if not line_words or not seg_bag:
        return {
            "score": 0.0,
            "raw_score": 0.0,
            "content_hits": 0,
            "function_hits": 0,
            "placeholder_ratio": 0.0,
        }

    seg_expanded: set[str] = set()
    for word in seg_bag:
        seg_expanded.update(whisper_utils._normalize_words_expanded(word))

    content_hits = 0
    function_hits = 0
    weighted_hits = 0.0
    weighted_total = 0.0
    raw_hits = 0
    raw_total = 0
    for word in line_words:
        for part in whisper_utils._normalize_words_expanded(word):
            if len(part) <= 1:
                continue
            raw_total += 1
            if part in seg_expanded:
                raw_hits += 1
            if _token_is_content(part):
                weighted_total += 1.0
                if part in seg_expanded:
                    content_hits += 1
                    weighted_hits += 1.0
            else:
                weighted_total += 0.2
                if part in seg_expanded:
                    function_hits += 1
                    weighted_hits += 0.2

    placeholder_ratio = whisper_blocks._segment_placeholder_ratio(seg_bag)
    raw_score = raw_hits / max(raw_total, 1)
    if placeholder_ratio < 0.5:
        score = raw_score
    else:
        score = weighted_hits / max(weighted_total, 1e-6)
        if placeholder_ratio >= 0.8 and content_hits == 0:
            score *= 0.2
        elif placeholder_ratio >= 0.5 and content_hits == 0:
            score *= 0.5
    return {
        "score": score,
        "raw_score": raw_score,
        "content_hits": content_hits,
        "function_hits": function_hits,
        "placeholder_ratio": placeholder_ratio,
    }


def _assign_lrc_lines_to_segments_experimental(  # noqa: C901
    lrc_lines_words: List[List[Tuple[int, str]]],
    seg_word_bags: List[List[str]],
    seg_durations: List[float],
    config: Any,
    trace_rows: List[dict[str, object]] | None = None,
) -> List[int]:
    lrc_line_count = len(lrc_lines_words)
    n_segs = len(seg_word_bags)
    line_to_seg: List[int] = [-1] * lrc_line_count
    seg_cursor = 0
    low_score_stall_run_length = 0

    for li in range(lrc_line_count):
        words = [w for _, w in lrc_lines_words[li]]
        if not words:
            line_to_seg[li] = line_to_seg[li - 1] if li > 0 else 0
            continue

        search_start, search_end = whisper_blocks._segment_search_window(
            seg_cursor=seg_cursor,
            n_segs=n_segs,
            config=config,
            low_score_stall_run_length=low_score_stall_run_length,
        )
        scored_candidates: list[dict[str, float | int]] = []
        best_seg = seg_cursor
        best_score = -1.0
        best_content_hits = -1
        for si in range(search_start, search_end):
            stats = _score_segment_for_line_experimental(words, seg_word_bags[si])
            stats["segment_index"] = si
            scored_candidates.append(stats)
            score = float(stats["score"])
            content_hits = int(stats["content_hits"])
            if score > best_score or (
                score == best_score and content_hits > best_content_hits
            ):
                best_score = score
                best_seg = si
                best_content_hits = content_hits
            if si + 1 >= n_segs:
                continue
            candidate = whisper_blocks._merged_segment_candidate(
                words=words,
                line_index=li,
                segment_index=si,
                current_best_score=best_score,
                seg_word_bags=seg_word_bags,
                seg_durations=seg_durations,
                config=config,
            )
            if candidate is None:
                continue
            merged_score, merged_seg = candidate
            merged_stats = _score_segment_for_line_experimental(
                words,
                seg_word_bags[si] + seg_word_bags[si + 1],
            )
            best_score, best_seg = merged_score, merged_seg
            best_content_hits = max(
                best_content_hits, int(merged_stats["content_hits"])
            )

        content_word_count = sum(1 for word in words if _token_is_content(word))
        if content_word_count >= 2 and best_content_hits <= 0:
            positive = [row for row in scored_candidates if float(row["score"]) > 0.0]
            if positive:
                local_positive = [
                    row
                    for row in positive
                    if int(row["segment_index"])
                    <= seg_cursor + _NO_CONTENT_MAX_FORWARD_JUMP
                ]
                if local_positive:
                    chosen = max(local_positive, key=lambda row: float(row["score"]))
                    best_seg = int(chosen["segment_index"])
                    best_score = float(chosen["score"])
                elif len(words) >= _NO_CONTENT_MIN_WORDS:
                    best_seg = min(
                        seg_cursor + _NO_CONTENT_MAX_FORWARD_JUMP, n_segs - 1
                    )
                    best_score = 0.0
            elif len(words) >= _NO_CONTENT_MIN_WORDS:
                best_seg = min(seg_cursor + 1, n_segs - 1)
                best_score = 0.0

        if best_score <= 0 and best_seg <= seg_cursor:
            best_seg = min(seg_cursor + 1, n_segs - 1)

        if trace_rows is not None:
            trace_rows.append(
                {
                    "line_index": li + 1,
                    "words": words,
                    "seg_cursor_before": seg_cursor,
                    "search_start": search_start,
                    "search_end": search_end,
                    "final_segment": best_seg,
                    "final_score": round(best_score, 4),
                    "scores": [
                        {
                            "segment_index": int(row["segment_index"]),
                            "score": round(float(row["score"]), 4),
                            "content_hits": int(row["content_hits"]),
                            "function_hits": int(row["function_hits"]),
                            "placeholder_ratio": round(
                                float(row["placeholder_ratio"]), 4
                            ),
                            "bag_preview": seg_word_bags[int(row["segment_index"])][
                                :16
                            ],
                        }
                        for row in scored_candidates
                    ],
                }
            )

        line_to_seg[li] = best_seg
        if (
            best_seg == seg_cursor
            and best_score <= config.terminal_stall_max_current_score
        ):
            low_score_stall_run_length += 1
        else:
            low_score_stall_run_length = 0
        seg_cursor = max(seg_cursor, best_seg)
    return line_to_seg


def _segment_assign_pipeline_mode() -> str:
    return (
        os.getenv("Y2K_WHISPER_SEGMENT_ASSIGN_PIPELINE", "default").strip() or "default"
    )


def build_segment_text_overlap_assignments(
    lrc_words: List[Dict],
    all_words: List[timing_models.TranscriptionWord],
    segments: List[timing_models.TranscriptionSegment],
) -> Dict[int, List[int]]:
    """Dispatch between the stable and experimental segment assigners."""
    mode = _segment_assign_pipeline_mode()
    if mode == _EXPERIMENTAL_ASSIGNER_MODE:
        return _build_segment_text_overlap_assignments_experimental(
            lrc_words=lrc_words,
            all_words=all_words,
            segments=segments,
        )
    return whisper_blocks._build_segment_text_overlap_assignments(
        lrc_words=lrc_words,
        all_words=all_words,
        segments=segments,
    )


def _build_segment_text_overlap_assignments_experimental(
    *,
    lrc_words: List[Dict],
    all_words: List[timing_models.TranscriptionWord],
    segments: List[timing_models.TranscriptionSegment],
) -> Dict[int, List[int]]:
    """Experimental assigner with content-word-aware segment selection."""
    if not segments or not lrc_words:
        return {}

    config = whisper_blocks._segment_assignment_config_from_env()
    trace_path = os.environ.get("Y2K_TRACE_SEGMENT_SELECTION_JSON", "").strip()
    trace_line_range = whisper_blocks._parse_trace_line_range_env()
    seg_word_ranges, seg_word_bags, seg_durations = (
        whisper_blocks._build_segment_word_info(
            all_words,
            segments,
        )
    )
    first_enclosing_ranges = whisper_blocks._build_first_enclosing_segment_word_ranges(
        all_words,
        segments,
    )

    lrc_line_count = max((lw["line_idx"] for lw in lrc_words), default=-1) + 1
    lrc_lines_words: List[List[Tuple[int, str]]] = [[] for _ in range(lrc_line_count)]
    for idx, lw in enumerate(lrc_words):
        lrc_lines_words[lw["line_idx"]].append(
            (idx, whisper_utils._normalize_word(lw["text"]))
        )

    trace_rows: List[dict[str, object]] = []
    line_to_seg = _assign_lrc_lines_to_segments_experimental(
        lrc_lines_words=lrc_lines_words,
        seg_word_bags=seg_word_bags,
        seg_durations=seg_durations,
        config=config,
        trace_rows=trace_rows,
    )
    if not _should_keep_experimental_assignments(trace_rows, len(lrc_lines_words)):
        return whisper_blocks._build_segment_text_overlap_assignments(
            lrc_words=lrc_words,
            all_words=all_words,
            segments=segments,
        )
    assignments = whisper_blocks._distribute_words_within_segments(
        line_to_seg,
        lrc_lines_words,
        seg_word_ranges,
    )
    assignments.seg_word_ranges = seg_word_ranges
    assignments.first_enclosing_ranges = first_enclosing_ranges
    if trace_path:
        filtered_rows = trace_rows
        if trace_line_range is not None:
            filtered_rows = [
                row
                for row in trace_rows
                if isinstance(row.get("line_index"), int)
                and trace_line_range[0]
                <= cast(int, row.get("line_index"))
                <= trace_line_range[1]
            ]
        with open(trace_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "line_to_seg": {
                        str(i + 1): seg for i, seg in enumerate(line_to_seg)
                    },
                    "segment_spans": {
                        str(i): [
                            round(whisper_utils._segment_start(seg), 3),
                            round(whisper_utils._segment_end(seg), 3),
                        ]
                        for i, seg in enumerate(segments)
                    },
                    "rows": filtered_rows,
                },
                fh,
                indent=2,
            )
    return assignments
