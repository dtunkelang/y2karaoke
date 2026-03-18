"""Segment-level Whisper/LRC assignment heuristics."""

import json
from collections import Counter
from typing import Any, Dict, List, Set, Tuple

from ....utils.logging import get_logger
from ..alignment import timing_models
from . import whisper_utils
from .whisper_assignment_trace import _json_safe_value
from .whisper_mapping_runtime_config import (
    SegmentAssignmentRuntimeConfig,
    WhisperMappingTraceConfig,
    load_segment_assignment_runtime_config,
    load_whisper_mapping_trace_config,
)

logger = get_logger(__name__)

_MAX_SEGMENT_WORDS_PER_LRC_WORD = 8.0
_PLACEHOLDER_SEGMENT_TOKENS = {"[vocal]"}
_SegmentAssignmentConfig = SegmentAssignmentRuntimeConfig


class _SegmentAssignments(dict[int, list[int]]):
    """Dict-like assignments container with optional segment-selection metadata."""

    def __init__(
        self,
        *args,
        selected_segment_by_lrc_idx: Dict[int, int] | None = None,
        seg_word_ranges: List[Tuple[int, int]] | None = None,
        first_enclosing_ranges: List[Tuple[int, int]] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.selected_segment_by_lrc_idx = selected_segment_by_lrc_idx or {}
        self.seg_word_ranges = seg_word_ranges or []
        self.first_enclosing_ranges = first_enclosing_ranges or []


def _parse_trace_line_range_env() -> Tuple[int, int] | None:
    return load_whisper_mapping_trace_config().line_range


def _segment_assignment_config_from_env() -> _SegmentAssignmentConfig:
    return load_segment_assignment_runtime_config()


def _text_overlap_score(
    line_words: List[str],
    seg_bag: List[str],
) -> float:
    if not line_words or not seg_bag:
        return 0.0
    seg_expanded: Set[str] = set()
    for w in seg_bag:
        seg_expanded.update(whisper_utils._normalize_words_expanded(w))
    hits = 0
    total = 0
    for w in line_words:
        parts = whisper_utils._normalize_words_expanded(w)
        for p in parts:
            if len(p) > 1:
                total += 1
                if p in seg_expanded:
                    hits += 1
    return hits / max(total, 1)


def _segment_placeholder_ratio(seg_bag: List[str]) -> float:
    if not seg_bag:
        return 0.0
    placeholder_count = sum(
        1 for token in seg_bag if token in _PLACEHOLDER_SEGMENT_TOKENS
    )
    return placeholder_count / len(seg_bag)


def _build_segment_word_info(
    all_words: List[timing_models.TranscriptionWord],
    segments: List[timing_models.TranscriptionSegment],
) -> Tuple[List[Tuple[int, int]], List[List[str]], List[float]]:
    seg_word_ranges: List[Tuple[int, int]] = []
    seg_word_bags: List[List[str]] = []
    seg_durations: List[float] = []
    for seg in segments:
        seg_start = whisper_utils._segment_start(seg)
        seg_end = whisper_utils._segment_end(seg)
        seg_durations.append(max(0.0, seg_end - seg_start))
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
                    whisper_utils._normalize_word(all_words[i].text)
                    for i in range(first_idx, last_idx + 1)
                ]
            )
        else:
            seg_word_bags.append([])
    return seg_word_ranges, seg_word_bags, seg_durations


def _build_first_enclosing_segment_word_ranges(
    all_words: List[timing_models.TranscriptionWord],
    segments: List[timing_models.TranscriptionSegment],
) -> List[Tuple[int, int]]:
    ranges: List[Tuple[int, int]] = [(-1, -1) for _ in segments]
    if not segments:
        return ranges
    seg_idx = 0
    for word_idx, word in enumerate(all_words):
        while seg_idx + 1 < len(segments) and word.start > whisper_utils._segment_end(
            segments[seg_idx]
        ):
            seg_idx += 1
        seg = segments[seg_idx]
        if not (
            whisper_utils._segment_start(seg)
            <= word.start
            <= whisper_utils._segment_end(seg)
        ):
            continue
        first_idx, last_idx = ranges[seg_idx]
        if first_idx < 0:
            ranges[seg_idx] = (word_idx, word_idx)
        else:
            ranges[seg_idx] = (first_idx, word_idx)
    return ranges


def _assign_lrc_lines_to_segments(
    lrc_lines_words: List[List[Tuple[int, str]]],
    seg_word_bags: List[List[str]],
    seg_durations: List[float],
    config: _SegmentAssignmentConfig,
    *,
    trace_config: WhisperMappingTraceConfig | None = None,
) -> List[int]:
    resolved_trace_config = trace_config or load_whisper_mapping_trace_config()
    trace_path = resolved_trace_config.segment_selection_path
    trace_line_range = resolved_trace_config.line_range
    trace_rows: List[Dict[str, Any]] | None = [] if trace_path else None
    lrc_line_count = len(lrc_lines_words)
    n_segs = len(seg_word_bags)
    line_to_seg: List[int] = [-1] * lrc_line_count
    seg_cursor = 0
    low_score_stall_run_length = 0
    for li in range(lrc_line_count):
        words = [w for _, w in lrc_lines_words[li]]
        content_words = [w for w in words if len(w) > 2]
        repeated_phrase_like = any(
            count >= 2 for _, count in Counter(content_words).items()
        )
        if not words:
            line_to_seg[li] = line_to_seg[li - 1] if li > 0 else 0
            continue
        search_start, search_end = _segment_search_window(
            seg_cursor=seg_cursor,
            n_segs=n_segs,
            config=config,
            low_score_stall_run_length=low_score_stall_run_length,
        )
        line_trace = _init_segment_selection_trace(
            trace_rows=trace_rows,
            trace_line_range=trace_line_range,
            line_index=li + 1,
            words=words,
            seg_cursor=seg_cursor,
            search_start=search_start,
            search_end=search_end,
        )
        best_seg, best_score = _score_segments_for_line(
            words=words,
            line_index=li,
            search_start=search_start,
            seg_cursor=seg_cursor,
            search_end=search_end,
            n_segs=n_segs,
            seg_word_bags=seg_word_bags,
            seg_durations=seg_durations,
            config=config,
            line_trace=line_trace,
        )
        best_seg, best_score = _select_segment_for_line_mode(
            words=words,
            repeated_phrase_like=repeated_phrase_like,
            best_seg=best_seg,
            best_score=best_score,
            seg_cursor=seg_cursor,
            seg_word_bags=seg_word_bags,
            n_segs=n_segs,
            config=config,
            line_trace=line_trace,
        )
        if best_score <= 0 and best_seg <= seg_cursor:
            if line_trace is not None:
                line_trace["zero_score_advance"] = {
                    "from_segment": best_seg,
                    "to_segment": min(seg_cursor + 1, n_segs - 1),
                }
            best_seg = min(seg_cursor + 1, n_segs - 1)
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
                if line_trace is not None:
                    line_trace["advanced_to_next_segment"] = {
                        "next_segment": best_seg,
                        "next_score": round(nxt, 4),
                    }
        line_to_seg[li] = best_seg
        if (
            best_seg == seg_cursor
            and best_score <= config.terminal_stall_max_current_score
        ):
            low_score_stall_run_length += 1
        else:
            low_score_stall_run_length = 0
        seg_cursor = max(seg_cursor, best_seg)
        _finalize_segment_selection_trace(
            line_trace=line_trace,
            trace_rows=trace_rows,
            best_seg=best_seg,
            best_score=best_score,
            seg_cursor=seg_cursor,
        )
    if trace_path and trace_rows is not None:
        with open(trace_path, "w", encoding="utf-8") as fh:
            json.dump(_json_safe_value({"rows": trace_rows}), fh, indent=2)
    return line_to_seg


def _init_segment_selection_trace(
    *,
    trace_rows: List[Dict[str, Any]] | None,
    trace_line_range: Tuple[int, int] | None,
    line_index: int,
    words: List[str],
    seg_cursor: int,
    search_start: int,
    search_end: int,
) -> Dict[str, Any] | None:
    if trace_rows is None:
        return None
    if trace_line_range is not None and not (
        trace_line_range[0] <= line_index <= trace_line_range[1]
    ):
        return None
    return {
        "line_index": line_index,
        "words": words,
        "seg_cursor_before": seg_cursor,
        "search_start": search_start,
        "search_end": search_end,
        "scores": [],
    }


def _segment_search_window(
    *,
    seg_cursor: int,
    n_segs: int,
    config: _SegmentAssignmentConfig,
    low_score_stall_run_length: int,
) -> Tuple[int, int]:
    search_start = seg_cursor
    if (
        config.selection_mode == "experimental_stall_widened_search"
        and low_score_stall_run_length >= config.stalled_search_min_run
        and seg_cursor > 0
    ):
        search_start = max(0, seg_cursor - config.stalled_search_lookback_segs)
    search_end = min(seg_cursor + max(10, n_segs // 4), n_segs)
    return search_start, search_end


def _score_segments_for_line(
    *,
    words: List[str],
    line_index: int,
    search_start: int,
    seg_cursor: int,
    search_end: int,
    n_segs: int,
    seg_word_bags: List[List[str]],
    seg_durations: List[float],
    config: _SegmentAssignmentConfig,
    line_trace: Dict[str, Any] | None,
) -> Tuple[int, float]:
    best_seg = seg_cursor
    best_score = -1.0
    for si in range(search_start, search_end):
        score = _text_overlap_score(words, seg_word_bags[si])
        if line_trace is not None:
            scores = line_trace["scores"]
            assert isinstance(scores, list)
            scores.append(
                {
                    "segment_index": si,
                    "score": round(score, 4),
                    "placeholder_ratio": round(
                        _segment_placeholder_ratio(seg_word_bags[si]), 4
                    ),
                    "bag_size": len(seg_word_bags[si]),
                    "bag_preview": seg_word_bags[si][:16],
                }
            )
        if score > best_score:
            best_score = score
            best_seg = si
        if si + 1 >= n_segs:
            continue
        candidate = _merged_segment_candidate(
            words=words,
            line_index=line_index,
            segment_index=si,
            current_best_score=best_score,
            seg_word_bags=seg_word_bags,
            seg_durations=seg_durations,
            config=config,
        )
        if candidate is None:
            continue
        if line_trace is not None:
            merged_candidates = line_trace.setdefault("merged_candidates", [])
            assert isinstance(merged_candidates, list)
            merged_candidates.append(
                {
                    "segment_index": si,
                    "score": round(candidate[0], 4),
                    "placeholder_ratio": round(
                        _segment_placeholder_ratio(
                            seg_word_bags[si] + seg_word_bags[si + 1]
                        ),
                        4,
                    ),
                    "bag_size": len(seg_word_bags[si] + seg_word_bags[si + 1]),
                    "chosen_segment": candidate[1],
                    "left_bag_preview": seg_word_bags[si][:8],
                    "right_bag_preview": seg_word_bags[si + 1][:8],
                }
            )
        best_score, best_seg = candidate
    return best_seg, best_score


def _finalize_segment_selection_trace(
    *,
    line_trace: Dict[str, Any] | None,
    trace_rows: List[Dict[str, Any]] | None,
    best_seg: int,
    best_score: float,
    seg_cursor: int,
) -> None:
    if line_trace is None:
        return
    line_trace["final_segment"] = best_seg
    line_trace["final_score"] = round(best_score, 4)
    line_trace["seg_cursor_after"] = seg_cursor
    assert trace_rows is not None
    trace_rows.append(line_trace)


def _merged_segment_candidate(
    *,
    words: List[str],
    line_index: int,
    segment_index: int,
    current_best_score: float,
    seg_word_bags: List[List[str]],
    seg_durations: List[float],
    config: _SegmentAssignmentConfig,
) -> Tuple[float, int] | None:
    merged = seg_word_bags[segment_index] + seg_word_bags[segment_index + 1]
    mscore = _text_overlap_score(words, merged)
    if mscore <= current_best_score:
        return None
    s1 = _text_overlap_score(words, seg_word_bags[segment_index])
    s2 = _text_overlap_score(words, seg_word_bags[segment_index + 1])
    best_seg = segment_index if s1 > 0 else segment_index + 1
    if _should_prefer_later_segment(
        words=words,
        s1=s1,
        s2=s2,
        mscore=mscore,
        seg_word_bags=seg_word_bags,
        seg_durations=seg_durations,
        segment_index=segment_index,
        config=config,
    ):
        _trace_later_merge_tiebreak(
            line_index=line_index,
            words=words,
            segment_index=segment_index,
            s1=s1,
            s2=s2,
            mscore=mscore,
            seg_word_bags=seg_word_bags,
            seg_durations=seg_durations,
            config=config,
        )
        best_seg = segment_index + 1
    return mscore, best_seg


def _should_prefer_later_segment(
    *,
    words: List[str],
    s1: float,
    s2: float,
    mscore: float,
    seg_word_bags: List[List[str]],
    seg_durations: List[float],
    segment_index: int,
    config: _SegmentAssignmentConfig,
) -> bool:
    if not config.prefer_later_on_strong_merge:
        return False
    return (
        len(words) >= 3
        and s1 > 0
        and s2 > 0
        and mscore >= 0.6
        and s2 >= max(0.35, s1 * 0.8)
        and (mscore - s1) >= 0.2
        and len(seg_word_bags[segment_index + 1]) >= max(4, int(len(words) * 0.7))
        and seg_durations[segment_index + 1] >= max(0.9, 0.18 * len(words))
        and len(seg_word_bags[segment_index])
        <= max(3, int(len(seg_word_bags[segment_index + 1]) * 0.6))
        and seg_durations[segment_index]
        <= max(2.2, seg_durations[segment_index + 1] * 0.7)
    )


def _trace_later_merge_tiebreak(
    *,
    line_index: int,
    words: List[str],
    segment_index: int,
    s1: float,
    s2: float,
    mscore: float,
    seg_word_bags: List[List[str]],
    seg_durations: List[float],
    config: _SegmentAssignmentConfig,
) -> None:
    if not config.later_trace:
        return
    logger.info(
        (
            "later-merge-tiebreak line=%d words=%s si=%d "
            "s1=%.3f s2=%.3f merged=%.3f seg1_wc=%d "
            "seg2_wc=%d seg1_dur=%.2f seg2_dur=%.2f"
        ),
        line_index,
        " ".join(words),
        segment_index,
        s1,
        s2,
        mscore,
        len(seg_word_bags[segment_index]),
        len(seg_word_bags[segment_index + 1]),
        seg_durations[segment_index],
        seg_durations[segment_index + 1],
    )


def _rescue_zero_score_repeated_line_assignment(
    *,
    words: List[str],
    repeated_phrase_like: bool,
    best_seg: int,
    best_score: float,
    seg_cursor: int,
    seg_word_bags: List[List[str]],
    n_segs: int,
    config: _SegmentAssignmentConfig,
) -> Tuple[int, float]:
    if (
        not config.zero_score_lookback_enabled
        or not repeated_phrase_like
        or len(words) < 4
        or best_score > 0
        or seg_cursor <= 0
    ):
        return best_seg, best_score

    lookback = config.zero_score_lookback_segs
    lb_start = max(0, seg_cursor - lookback)
    lb_best_seg = best_seg
    lb_best_score = best_score
    for si in range(lb_start, seg_cursor + 1):
        s = _text_overlap_score(words, seg_word_bags[si])
        if s > lb_best_score:
            lb_best_score = s
            lb_best_seg = si
        if si + 1 < n_segs:
            ms = _text_overlap_score(words, seg_word_bags[si] + seg_word_bags[si + 1])
            if ms > lb_best_score:
                lb_best_score = ms
                lb_best_seg = si + 1
    if lb_best_score > 0:
        return lb_best_seg, lb_best_score
    return best_seg, best_score


def _rescue_terminal_stall_line_assignment(
    *,
    words: List[str],
    best_seg: int,
    best_score: float,
    seg_cursor: int,
    seg_word_bags: List[List[str]],
    n_segs: int,
    config: _SegmentAssignmentConfig,
) -> Tuple[int, float]:
    if (
        len(words) < 4
        or n_segs < 2
        or seg_cursor != n_segs - 1
        or best_seg != seg_cursor
        or best_score > config.terminal_stall_max_current_score
    ):
        return best_seg, best_score

    lb_start = max(0, seg_cursor - config.terminal_stall_lookback_segs)
    rescue_seg = best_seg
    rescue_score = best_score
    for si in range(lb_start, seg_cursor):
        score = _text_overlap_score(words, seg_word_bags[si])
        if score > rescue_score:
            rescue_seg = si
            rescue_score = score
        if si + 1 < n_segs:
            merged_score = _text_overlap_score(
                words, seg_word_bags[si] + seg_word_bags[si + 1]
            )
            if merged_score > rescue_score:
                rescue_seg = si + 1
                rescue_score = merged_score
    if rescue_score >= max(
        config.terminal_stall_min_rescue_score,
        best_score + config.terminal_stall_min_score_gain,
    ):
        return rescue_seg, rescue_score
    return best_seg, best_score


def _select_segment_for_line_mode(
    *,
    words: List[str],
    repeated_phrase_like: bool,
    best_seg: int,
    best_score: float,
    seg_cursor: int,
    seg_word_bags: List[List[str]],
    n_segs: int,
    config: _SegmentAssignmentConfig,
    line_trace: Dict[str, Any] | None,
) -> Tuple[int, float]:
    mode = config.selection_mode
    if line_trace is not None:
        line_trace["selection_mode"] = mode
    best_seg, best_score = _rescue_zero_score_repeated_line_assignment(
        words=words,
        repeated_phrase_like=repeated_phrase_like,
        best_seg=best_seg,
        best_score=best_score,
        seg_cursor=seg_cursor,
        seg_word_bags=seg_word_bags,
        n_segs=n_segs,
        config=config,
    )
    if mode in {
        "experimental_terminal_stall_lookback",
        "experimental_terminal_stall_tie_break",
    }:
        lookback_scores = []
        lb_start = max(0, seg_cursor - config.terminal_stall_lookback_segs)
        for si in range(lb_start, seg_cursor):
            score = _text_overlap_score(words, seg_word_bags[si])
            merged_score = None
            if si + 1 < n_segs:
                merged_score = _text_overlap_score(
                    words, seg_word_bags[si] + seg_word_bags[si + 1]
                )
            lookback_scores.append(
                {
                    "segment_index": si,
                    "score": round(score, 4),
                    "merged_score": (
                        round(merged_score, 4) if merged_score is not None else None
                    ),
                    "placeholder_ratio": round(
                        _segment_placeholder_ratio(seg_word_bags[si]), 4
                    ),
                    "merged_placeholder_ratio": (
                        round(
                            _segment_placeholder_ratio(
                                seg_word_bags[si] + seg_word_bags[si + 1]
                            ),
                            4,
                        )
                        if merged_score is not None
                        else None
                    ),
                    "bag_size": len(seg_word_bags[si]),
                    "bag_preview": seg_word_bags[si][:8],
                }
            )
        if line_trace is not None:
            trace_key = (
                "experimental_terminal_stall_lookback_scores"
                if mode == "experimental_terminal_stall_lookback"
                else "experimental_terminal_stall_tie_scores"
            )
            line_trace[trace_key] = lookback_scores
    if mode == "experimental_terminal_stall_lookback":
        rescued_seg, rescued_score = _rescue_terminal_stall_line_assignment(
            words=words,
            best_seg=best_seg,
            best_score=best_score,
            seg_cursor=seg_cursor,
            seg_word_bags=seg_word_bags,
            n_segs=n_segs,
            config=config,
        )
        if line_trace is not None and (
            rescued_seg != best_seg or rescued_score != best_score
        ):
            line_trace["experimental_terminal_stall_rescue"] = {
                "from_segment": best_seg,
                "to_segment": rescued_seg,
                "from_score": round(best_score, 4),
                "to_score": round(rescued_score, 4),
            }
        return rescued_seg, rescued_score
    return best_seg, best_score


def _distribute_words_within_segments(
    line_to_seg: List[int],
    lrc_lines_words: List[List[Tuple[int, str]]],
    seg_word_ranges: List[Tuple[int, int]],
    trace_rows: List[Dict] | None = None,
) -> _SegmentAssignments:
    seg_to_lines: Dict[int, List[int]] = {}
    for li, si in enumerate(line_to_seg):
        if si >= 0:
            seg_to_lines.setdefault(si, []).append(li)

    selected_segment_by_lrc_idx: Dict[int, int] = {}
    assignments: Dict[int, List[int]] = {}
    for si, line_indices in seg_to_lines.items():
        first_wi, last_wi = seg_word_ranges[si]
        if first_wi < 0:
            continue
        seg_wc = last_wi - first_wi + 1
        all_lrc = _collect_segment_lrc_word_indices(
            line_indices=line_indices,
            lrc_lines_words=lrc_lines_words,
        )
        total = len(all_lrc)
        if total == 0:
            continue
        if seg_wc / total > _MAX_SEGMENT_WORDS_PER_LRC_WORD:
            _append_segment_distribution_trace(
                trace_rows=trace_rows,
                line_indices=line_indices,
                segment_index=si,
                segment_word_range=(first_wi, last_wi),
                segment_word_count=seg_wc,
                lrc_word_indices=all_lrc,
                skipped=True,
                assignments=None,
            )
            continue
        for j, lrc_idx in enumerate(all_lrc):
            pos = j / max(total, 1)
            offset = min(int(pos * seg_wc), seg_wc - 1)
            assignments[lrc_idx] = [first_wi + offset]
            selected_segment_by_lrc_idx[lrc_idx] = si
        _append_segment_distribution_trace(
            trace_rows=trace_rows,
            line_indices=line_indices,
            segment_index=si,
            segment_word_range=(first_wi, last_wi),
            segment_word_count=seg_wc,
            lrc_word_indices=all_lrc,
            skipped=False,
            assignments=assignments,
        )
    return _SegmentAssignments(
        assignments,
        selected_segment_by_lrc_idx=selected_segment_by_lrc_idx,
    )


def _collect_segment_lrc_word_indices(
    *,
    line_indices: List[int],
    lrc_lines_words: List[List[Tuple[int, str]]],
) -> List[int]:
    all_lrc: List[int] = []
    for li in line_indices:
        for idx, _ in lrc_lines_words[li]:
            all_lrc.append(idx)
    return all_lrc


def _append_segment_distribution_trace(
    *,
    trace_rows: List[Dict] | None,
    line_indices: List[int],
    segment_index: int,
    segment_word_range: Tuple[int, int],
    segment_word_count: int,
    lrc_word_indices: List[int],
    skipped: bool,
    assignments: Dict[int, List[int]] | None,
) -> None:
    if trace_rows is None:
        return
    row: Dict[str, Any] = {
        "line_indices": [li + 1 for li in line_indices],
        "segment_index": segment_index,
        "segment_word_range": [segment_word_range[0], segment_word_range[1]],
        "segment_word_count": segment_word_count,
        "lrc_word_indices": lrc_word_indices,
        "skipped": skipped,
    }
    if skipped:
        row["skip_reason"] = "segment_too_wide_for_positional_distribution"
    else:
        assert assignments is not None
        row["distributed_assignments"] = {
            str(lrc_idx): assignments[lrc_idx][0] for lrc_idx in lrc_word_indices
        }
    trace_rows.append(row)


def _build_segment_text_overlap_assignments(
    lrc_words: List[Dict],
    all_words: List[timing_models.TranscriptionWord],
    segments: List[timing_models.TranscriptionSegment],
    *,
    config: _SegmentAssignmentConfig | None = None,
    trace_config: WhisperMappingTraceConfig | None = None,
) -> Dict[int, List[int]]:
    resolved_trace_config = trace_config or load_whisper_mapping_trace_config()
    trace_path = resolved_trace_config.segment_assignments_path
    trace_line_range = resolved_trace_config.line_range
    if not segments or not lrc_words:
        return {}
    resolved_config = config or _segment_assignment_config_from_env()

    seg_word_ranges, seg_word_bags, seg_durations = _build_segment_word_info(
        all_words,
        segments,
    )
    first_enclosing_ranges = _build_first_enclosing_segment_word_ranges(
        all_words,
        segments,
    )

    lrc_line_count = max((lw["line_idx"] for lw in lrc_words), default=-1) + 1
    lrc_lines_words: List[List[Tuple[int, str]]] = [[] for _ in range(lrc_line_count)]
    for idx, lw in enumerate(lrc_words):
        lrc_lines_words[lw["line_idx"]].append(
            (idx, whisper_utils._normalize_word(lw["text"]))
        )

    line_to_seg = _assign_lrc_lines_to_segments(
        lrc_lines_words,
        seg_word_bags,
        seg_durations,
        resolved_config,
        trace_config=resolved_trace_config,
    )
    trace_rows: List[Dict] | None = [] if trace_path else None
    assignments = _distribute_words_within_segments(
        line_to_seg,
        lrc_lines_words,
        seg_word_ranges,
        trace_rows=trace_rows,
    )
    assignments.seg_word_ranges = seg_word_ranges
    assignments.first_enclosing_ranges = first_enclosing_ranges

    if trace_path and trace_rows is not None:
        filtered_rows = trace_rows
        if trace_line_range is not None:
            filtered_rows = [
                row
                for row in trace_rows
                if any(
                    trace_line_range[0] <= line_idx <= trace_line_range[1]
                    for line_idx in row["line_indices"]
                )
            ]
        with open(trace_path, "w", encoding="utf-8") as fh:
            json.dump(
                _json_safe_value(
                    {
                        "line_to_seg": {
                            str(i + 1): seg for i, seg in enumerate(line_to_seg)
                        },
                        "seg_word_ranges": {
                            str(i): [first, last]
                            for i, (first, last) in enumerate(seg_word_ranges)
                        },
                        "first_enclosing_seg_word_ranges": {
                            str(i): [first, last]
                            for i, (first, last) in enumerate(first_enclosing_ranges)
                        },
                        "segment_spans": {
                            str(i): [
                                round(whisper_utils._segment_start(seg), 3),
                                round(whisper_utils._segment_end(seg), 3),
                            ]
                            for i, seg in enumerate(segments)
                        },
                        "rows": filtered_rows,
                    }
                ),
                fh,
                indent=2,
            )

    logger.debug(
        "Segment text-overlap: %d/%d LRC words assigned to %d segments",
        len(assignments),
        len(lrc_words),
        len(segments),
    )
    return assignments
