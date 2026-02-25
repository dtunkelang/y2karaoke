from __future__ import annotations

import math
import os
import re
from typing import List, Optional, Tuple

from ..models import TargetLine
from ..text_utils import normalize_text_basic
from .refinement_dense_run_postpasses import (
    _compress_overlong_sparse_line_timings as _compress_overlong_sparse_line_timings_impl,
    _promote_unresolved_first_repeated_lines as _promote_unresolved_first_repeated_lines_impl,
    _pull_dense_short_runs_toward_previous_anchor as _pull_dense_short_runs_toward_previous_anchor_impl,
    _rebalance_early_lead_shared_visibility_runs as _rebalance_early_lead_shared_visibility_runs_impl,
    _retime_compressed_shared_visibility_blocks as _retime_compressed_shared_visibility_blocks_impl,
    _retime_dense_runs_after_overlong_lead as _retime_dense_runs_after_overlong_lead_impl,
    _retime_late_first_lines_in_shared_visibility_blocks as _retime_late_first_lines_in_shared_visibility_blocks_impl,
    _shrink_overlong_leads_in_dense_shared_visibility_runs as _shrink_overlong_leads_in_dense_shared_visibility_runs_impl,
)
from .refinement_shared_visibility_postpasses import (
    _rebalance_two_followups_after_short_lead as _rebalance_two_followups_after_short_lead_impl,
    _retime_followups_in_short_lead_shared_visibility_runs as _retime_followups_in_short_lead_shared_visibility_runs_impl,
    _retime_large_gaps_with_early_visibility as _retime_large_gaps_with_early_visibility_impl,
)
from .refinement_transition_postpasses import (
    _clamp_line_ends_to_visibility_windows as _clamp_line_ends_to_visibility_windows_impl,
    _pull_lines_earlier_after_visibility_transitions as _pull_lines_earlier_after_visibility_transitions_impl,
    _rebalance_middle_lines_in_four_line_shared_visibility_runs as _rebalance_middle_lines_impl,
    _retime_short_interstitial_lines_between_anchors as _retime_short_interstitial_lines_between_anchors_impl,
)
from .refinement_repetition_postpasses import (
    _pull_late_first_lines_in_alternating_repeated_blocks as _pull_late_first_lines_in_alternating_repeated_blocks_impl,
    _retime_repeated_blocks_with_long_tail_gap as _retime_repeated_blocks_with_long_tail_gap_impl,
)


def _canonical_line_text(ln: TargetLine) -> str:
    """Light normalization for repetition guards in timing postpasses."""
    text = ln.text if ln.text else " ".join(ln.words)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\\s']+", " ", text)
    return re.sub(r"\\s+", " ", text).strip()


def _assign_line_level_word_timings(
    ln: TargetLine,
    line_start: Optional[float],
    line_end: Optional[float],
    line_confidence: float,
) -> None:
    n_words = len(ln.words)
    if n_words == 0:
        ln.word_starts = []
        ln.word_ends = []
        ln.word_confidences = []
        return

    start = line_start if line_start is not None else ln.start
    if start is None:
        start = 0.0

    if line_end is not None and line_end > start + 0.05:
        end = line_end
    elif ln.end is not None and ln.end > start + 0.05:
        end = ln.end
    else:
        end = start + max(1.0, 0.2 * n_words)

    min_word_duration = 0.12
    inter_word_gap = 0.04
    min_line_span = n_words * min_word_duration + max(0, n_words - 1) * inter_word_gap
    span = max(end - start, min_line_span)
    end = start + span

    raw_weights = [max(sum(ch.isalnum() for ch in w), 1) for w in ln.words]
    shaped_weights = [math.sqrt(float(w)) for w in raw_weights]
    weight_sum = sum(shaped_weights)
    if weight_sum <= 0:
        shaped_weights = [1.0] * n_words
        weight_sum = float(n_words)

    available = span - max(0, n_words - 1) * inter_word_gap
    base_floor = min_word_duration * n_words
    extra = max(0.0, available - base_floor)
    durations = [min_word_duration + extra * (w / weight_sum) for w in shaped_weights]

    starts: List[Optional[float]] = []
    ends: List[Optional[float]] = []
    cursor = start
    for i, dur in enumerate(durations):
        starts.append(cursor)
        word_end = cursor + dur
        if i == n_words - 1:
            word_end = end
        ends.append(word_end)
        cursor = word_end + inter_word_gap

    conf = max(0.2, min(0.5, float(line_confidence) * 0.6 if line_confidence else 0.3))
    ln.word_starts = starts
    ln.word_ends = ends
    ln.word_confidences = [conf] * n_words


def _line_start(ln: TargetLine) -> Optional[float]:
    if ln.word_starts and ln.word_starts[0] is not None:
        return float(ln.word_starts[0])
    return None


def _line_end(ln: TargetLine) -> Optional[float]:
    if ln.word_ends and ln.word_ends[-1] is not None:
        return float(ln.word_ends[-1])
    if ln.end is not None:
        return float(ln.end)
    return None


def _compute_line_min_start_time(
    ln: TargetLine,
    *,
    last_assigned_start: Optional[float],
    last_assigned_visibility_end: Optional[float],
) -> Optional[float]:
    min_start: Optional[float] = None
    if ln.visibility_start is not None:
        min_start = float(ln.visibility_start)
        if ln.visibility_end is not None:
            vis_span = max(0.0, float(ln.visibility_end) - float(ln.visibility_start))
            if vis_span >= 8.0:
                min_start -= 2.0
            elif vis_span >= 5.0:
                min_start -= 1.0
        min_start = max(0.0, min_start)

    enforce_global_gate = True
    if (
        ln.visibility_start is not None
        and last_assigned_visibility_end is not None
        and float(ln.visibility_start) < (float(last_assigned_visibility_end) - 0.2)
    ):
        enforce_global_gate = False

    if last_assigned_start is not None and enforce_global_gate:
        gate = float(last_assigned_start + 0.05)
        min_start = gate if min_start is None else max(min_start, gate)
    return min_start


def _retime_late_first_lines_in_shared_visibility_blocks(
    g_jobs: List[Tuple[TargetLine, float, float]],
) -> None:
    _retime_late_first_lines_in_shared_visibility_blocks_impl(
        g_jobs,
        line_start_fn=_line_start,
        line_end_fn=_line_end,
        assign_line_level_word_timings_fn=_assign_line_level_word_timings,
    )


def _retime_compressed_shared_visibility_blocks(  # noqa: C901
    g_jobs: List[Tuple[TargetLine, float, float]],
) -> None:
    _retime_compressed_shared_visibility_blocks_impl(
        g_jobs,
        line_start_fn=_line_start,
        line_end_fn=_line_end,
        assign_line_level_word_timings_fn=_assign_line_level_word_timings,
    )


def _promote_unresolved_first_repeated_lines(  # noqa: C901
    g_jobs: List[Tuple[TargetLine, float, float]],
) -> None:
    """Backfill early repeated-line starts when only later repeats were resolved."""
    _promote_unresolved_first_repeated_lines_impl(
        g_jobs,
        line_start_fn=_line_start,
        line_end_fn=_line_end,
        canonical_line_text_fn=_canonical_line_text,
        assign_line_level_word_timings_fn=_assign_line_level_word_timings,
    )


def _compress_overlong_sparse_line_timings(  # noqa: C901
    g_jobs: List[Tuple[TargetLine, float, float]],
) -> None:
    """Compress lines with sparse, overlong word timings inside shared-visibility blocks."""
    _compress_overlong_sparse_line_timings_impl(
        g_jobs,
        line_start_fn=_line_start,
        assign_line_level_word_timings_fn=_assign_line_level_word_timings,
    )


def _retime_large_gaps_with_early_visibility(  # noqa: C901
    g_jobs: List[Tuple[TargetLine, float, float]],
) -> None:
    _retime_large_gaps_with_early_visibility_impl(
        g_jobs,
        line_start_fn=_line_start,
        line_end_fn=_line_end,
        assign_line_level_word_timings_fn=_assign_line_level_word_timings,
    )


def _retime_followups_in_short_lead_shared_visibility_runs(  # noqa: C901
    g_jobs: List[Tuple[TargetLine, float, float]],
) -> None:
    _retime_followups_in_short_lead_shared_visibility_runs_impl(
        g_jobs,
        line_start_fn=_line_start,
        line_end_fn=_line_end,
        assign_line_level_word_timings_fn=_assign_line_level_word_timings,
    )


def _rebalance_two_followups_after_short_lead(
    g_jobs: List[Tuple[TargetLine, float, float]],
) -> None:
    _rebalance_two_followups_after_short_lead_impl(
        g_jobs,
        line_start_fn=_line_start,
        line_end_fn=_line_end,
        assign_line_level_word_timings_fn=_assign_line_level_word_timings,
    )


def _clamp_line_ends_to_visibility_windows(
    g_jobs: List[Tuple[TargetLine, float, float]],
) -> None:
    _clamp_line_ends_to_visibility_windows_impl(
        g_jobs,
        line_start_fn=_line_start,
        line_end_fn=_line_end,
        assign_line_level_word_timings_fn=_assign_line_level_word_timings,
    )


def _pull_lines_earlier_after_visibility_transitions(
    g_jobs: List[Tuple[TargetLine, float, float]],
) -> None:
    _pull_lines_earlier_after_visibility_transitions_impl(
        g_jobs,
        line_start_fn=_line_start,
        line_end_fn=_line_end,
        assign_line_level_word_timings_fn=_assign_line_level_word_timings,
    )


def _retime_short_interstitial_lines_between_anchors(
    g_jobs: List[Tuple[TargetLine, float, float]],
) -> None:
    _retime_short_interstitial_lines_between_anchors_impl(
        g_jobs,
        line_start_fn=_line_start,
        line_end_fn=_line_end,
        assign_line_level_word_timings_fn=_assign_line_level_word_timings,
    )


def _rebalance_middle_lines_in_four_line_shared_visibility_runs(
    g_jobs: List[Tuple[TargetLine, float, float]],
) -> None:
    _rebalance_middle_lines_impl(
        g_jobs,
        line_start_fn=_line_start,
        line_end_fn=_line_end,
        assign_line_level_word_timings_fn=_assign_line_level_word_timings,
    )


def _rebalance_early_lead_shared_visibility_runs(  # noqa: C901
    g_jobs: List[Tuple[TargetLine, float, float]],
) -> None:
    _rebalance_early_lead_shared_visibility_runs_impl(
        g_jobs,
        line_start_fn=_line_start,
        line_end_fn=_line_end,
        assign_line_level_word_timings_fn=_assign_line_level_word_timings,
    )


def _shrink_overlong_leads_in_dense_shared_visibility_runs(  # noqa: C901
    g_jobs: List[Tuple[TargetLine, float, float]],
) -> None:
    _shrink_overlong_leads_in_dense_shared_visibility_runs_impl(
        g_jobs,
        line_start_fn=_line_start,
        line_end_fn=_line_end,
        canonical_line_text_fn=_canonical_line_text,
        assign_line_level_word_timings_fn=_assign_line_level_word_timings,
    )


def _retime_dense_runs_after_overlong_lead(
    g_jobs: List[Tuple[TargetLine, float, float]],
) -> None:
    _retime_dense_runs_after_overlong_lead_impl(
        g_jobs,
        line_start_fn=_line_start,
        line_end_fn=_line_end,
        canonical_line_text_fn=_canonical_line_text,
        assign_line_level_word_timings_fn=_assign_line_level_word_timings,
    )


def _pull_dense_short_runs_toward_previous_anchor(
    g_jobs: List[Tuple[TargetLine, float, float]],
) -> None:
    _pull_dense_short_runs_toward_previous_anchor_impl(
        g_jobs,
        line_start_fn=_line_start,
        line_end_fn=_line_end,
        canonical_line_text_fn=_canonical_line_text,
        assign_line_level_word_timings_fn=_assign_line_level_word_timings,
    )


def _retime_repeated_blocks_with_long_tail_gap(
    g_jobs: List[Tuple[TargetLine, float, float]],
) -> None:
    _retime_repeated_blocks_with_long_tail_gap_impl(
        g_jobs,
        line_start_fn=_line_start,
        line_end_fn=_line_end,
        canonical_line_text_fn=_canonical_line_text,
        assign_line_level_word_timings_fn=_assign_line_level_word_timings,
    )


def _pull_late_first_lines_in_alternating_repeated_blocks(
    g_jobs: List[Tuple[TargetLine, float, float]],
) -> None:
    _pull_late_first_lines_in_alternating_repeated_blocks_impl(
        g_jobs,
        line_start_fn=_line_start,
        line_end_fn=_line_end,
        canonical_line_text_fn=_canonical_line_text,
        assign_line_level_word_timings_fn=_assign_line_level_word_timings,
    )


def _token_list(ln: TargetLine) -> list[str]:
    return [t for t in _canonical_line_text(ln).split() if t]


def _is_fragment_of_line(a: TargetLine, b: TargetLine) -> bool:
    ta = _token_list(a)
    tb = _token_list(b)
    if len(ta) < 1 or len(tb) < 2:
        return False
    if len(ta) >= len(tb):
        return False
    # contiguous token subphrase match
    for i in range(0, len(tb) - len(ta) + 1):
        if tb[i : i + len(ta)] == ta:
            return True
    # split-word OCR fragments like "con ting" vs "counting"
    if len(ta) == 2 and len(tb) >= 1:
        glued = "".join(ta)
        if any(glued == tok for tok in tb):
            return True
    return False


def _trace_clean_blocks(label: str, target_lines: List[TargetLine]) -> None:
    if os.environ.get("Y2K_VISUAL_BLOCK_TRACE", "0") != "1":
        return
    blocks: list[dict[str, object]] = []
    i = 0
    while i < len(target_lines):
        ln = target_lines[i]
        if ln.visibility_start is None or ln.visibility_end is None:
            i += 1
            continue
        vs0 = float(ln.visibility_start)
        ve0 = float(ln.visibility_end)
        block = [ln]
        j = i + 1
        while j < len(target_lines):
            lnj = target_lines[j]
            if lnj.visibility_start is None or lnj.visibility_end is None:
                break
            vs = float(lnj.visibility_start)
            ve = float(lnj.visibility_end)
            overlap = min(ve0, ve) - max(vs0, vs)
            if abs(vs - vs0) > 0.6 or overlap < 0.5:
                break
            block.append(lnj)
            j += 1
        if 2 <= len(block) <= 6:
            blocks.append(
                {
                    "start_idx": i,
                    "size": len(block),
                    "vs_span": round(
                        max(float(x.visibility_start or 0.0) for x in block)
                        - min(float(x.visibility_start or 0.0) for x in block),
                        2,
                    ),
                    "texts": [" ".join(x.words[:6]) for x in block],
                }
            )
        i = max(i + 1, j)
    import logging

    logging.getLogger(__name__).info("BLOCK_TRACE %s blocks=%s", label, blocks[:20])


def _retime_clean_screen_blocks_by_vertical_order(  # noqa: C901
    g_jobs: List[Tuple[TargetLine, float, float]],
) -> None:
    """Base-case karaoke block retiming: preserve top-to-bottom order in clean blocks.

    Applies only to small contiguous shared-visibility blocks with a clear timing inversion.
    """
    lines = [ln for ln, _, _ in g_jobs]
    n = len(lines)
    if n < 2:
        return

    i = 0
    while i < n:
        ln0 = lines[i]
        s0 = _line_start(ln0)
        if (
            s0 is None
            or ln0.visibility_start is None
            or ln0.visibility_end is None
            or (float(ln0.visibility_end) - float(ln0.visibility_start)) < 1.0
        ):
            i += 1
            continue

        vs0 = float(ln0.visibility_start)
        ve0 = float(ln0.visibility_end)
        block = [lines[i]]
        j = i + 1
        while j < n:
            ln = lines[j]
            sj = _line_start(ln)
            if sj is None or ln.visibility_start is None or ln.visibility_end is None:
                break
            vs = float(ln.visibility_start)
            ve = float(ln.visibility_end)
            overlap = min(ve0, ve) - max(vs0, vs)
            if abs(vs - vs0) > 0.45 or abs(ve - ve0) > 1.75 or overlap < 0.75:
                break
            block.append(ln)
            j += 1

        if 2 <= len(block) <= 5:
            y_vals = [float(ln.y) for ln in block]
            if (max(y_vals) - min(y_vals)) >= 24.0:
                by_y = sorted(
                    block, key=lambda ln: (float(ln.y), _line_start(ln) or 0.0)
                )
                starts_by_y = [_line_start(ln) or 0.0 for ln in by_y]
                has_inversion = any(
                    starts_by_y[k] > starts_by_y[k + 1] + 0.35
                    for k in range(len(starts_by_y) - 1)
                )
                if has_inversion:
                    target_starts = sorted(starts_by_y)
                    prev_end: Optional[float] = None
                    for k, ln in enumerate(by_y):
                        old_start = _line_start(ln)
                        if old_start is None:
                            continue
                        old_end = _line_end(ln)
                        dur = (
                            max(0.25, float(old_end) - float(old_start))
                            if old_end is not None
                            else max(0.8, 0.22 * max(1, len(ln.words)))
                        )
                        new_start = target_starts[k]
                        if prev_end is not None:
                            new_start = max(new_start, prev_end + 0.05)
                        if ln.visibility_start is not None:
                            new_start = max(
                                new_start, float(ln.visibility_start) - 0.15
                            )
                        new_end = new_start + dur
                        if ln.visibility_end is not None:
                            new_end = min(new_end, float(ln.visibility_end) + 0.4)
                        if new_end <= new_start + 0.12:
                            new_end = new_start + 0.12

                        if abs(new_start - float(old_start)) > 0.2:
                            conf = 0.4
                            if ln.word_confidences:
                                vals = [c for c in ln.word_confidences if c is not None]
                                if vals:
                                    conf = float(sum(vals) / len(vals))
                            _assign_line_level_word_timings(
                                ln, new_start, new_end, conf
                            )
                        end_val = _line_end(ln)
                        prev_end = float(end_val) if end_val is not None else new_end

        i = max(j, i + 1)


def _reorder_clean_screen_blocks_target_lines(  # noqa: C901
    target_lines: List[TargetLine],
) -> None:
    """Reorder clean screenful blocks in target-line list by vertical order.

    This is the base-case karaoke rule: for a simple shared-visibility screen block,
    preserve top-to-bottom row order. We only touch small contiguous blocks with a
    strong timing inversion to avoid harming repeated/messy sections.
    """
    if len(target_lines) < 2:
        return

    indexed: list[tuple[int, TargetLine, float, float, float]] = []
    for idx, ln in enumerate(target_lines):
        if ln.visibility_start is None or ln.visibility_end is None:
            continue
        s = _line_start(ln)
        if s is None:
            continue
        vs = float(ln.visibility_start)
        ve = float(ln.visibility_end)
        if ve <= vs:
            continue
        indexed.append((idx, ln, vs, ve, float(s)))
    if len(indexed) < 2:
        return

    blocks: list[list[tuple[int, TargetLine, float, float, float]]] = []
    cur: list[tuple[int, TargetLine, float, float, float]] = []
    for rec in indexed:
        if not cur:
            cur = [rec]
            continue
        prev_idx = cur[-1][0]
        idx, _ln, vs, ve, _s = rec
        # Require contiguity in target-line order to avoid broad reordering.
        if idx != prev_idx + 1:
            blocks.append(cur)
            cur = [rec]
            continue
        block_vs = [r[2] for r in cur]
        block_ve = [r[3] for r in cur]
        c_start = min(block_vs)
        c_end = max(block_ve)
        overlap = min(c_end, ve) - max(c_start, vs)
        if (vs - c_start) <= 1.4 and overlap > 0.2:
            cur.append(rec)
        else:
            blocks.append(cur)
            cur = [rec]
    if cur:
        blocks.append(cur)

    reordered_any = False
    for block in blocks:
        if not (2 <= len(block) <= 5):
            continue
        ys = [float(rec[1].y) for rec in block]
        if (max(ys) - min(ys)) < 20.0:
            continue
        block_vs = [rec[2] for rec in block]
        block_ve = [rec[3] for rec in block]
        if (max(block_vs) - min(block_vs)) > 0.45:
            continue
        if (max(block_ve) - min(block_ve)) > 12.0:
            continue

        by_y = sorted(block, key=lambda r: (float(r[1].y), r[4], r[0]))
        starts_by_y = [r[4] for r in by_y]
        has_inversion = any(
            starts_by_y[k] > starts_by_y[k + 1] + 0.75
            for k in range(len(starts_by_y) - 1)
        )
        if not has_inversion:
            continue

        start_idx = block[0][0]
        end_idx = block[-1][0] + 1
        replacement = [rec[1] for rec in by_y]
        target_lines[start_idx:end_idx] = replacement
        reordered_any = True

    if reordered_any:
        for i, ln in enumerate(target_lines):
            ln.line_index = i + 1
            ln.block_order_hint = i


def _assign_block_sequence_hints_from_visibility(  # noqa: C901
    target_lines: List[TargetLine],
) -> None:
    """Assign explicit block/row ordering hints from shared visibility windows.

    This is a block-first flattening prototype: detect simple screenful blocks using
    visibility windows, order rows by vertical position, then assign a global order
    hint by block chronology (visibility start).
    """
    if len(target_lines) < 2:
        return

    indexed: list[tuple[int, TargetLine]] = list(enumerate(target_lines))
    blocks: list[list[tuple[int, TargetLine]]] = []
    current: list[tuple[int, TargetLine]] = []

    def _line_vs(ln: TargetLine) -> Optional[float]:
        return float(ln.visibility_start) if ln.visibility_start is not None else None

    def _line_ve(ln: TargetLine) -> Optional[float]:
        return float(ln.visibility_end) if ln.visibility_end is not None else None

    for idx, ln in indexed:
        vs = _line_vs(ln)
        ve = _line_ve(ln)
        if vs is None or ve is None or ve <= vs:
            if current:
                blocks.append(current)
                current = []
            blocks.append([(idx, ln)])
            continue

        if not current:
            current = [(idx, ln)]
            continue

        c_vs = [
            float(c_ln.visibility_start or 0.0)
            for _, c_ln in current
            if c_ln.visibility_start is not None
        ]
        c_ve = [
            float(c_ln.visibility_end or 0.0)
            for _, c_ln in current
            if c_ln.visibility_end is not None
        ]
        if not c_vs or not c_ve:
            blocks.append(current)
            current = [(idx, ln)]
            continue
        c_start = min(c_vs)
        c_end = max(c_ve)
        overlap = min(c_end, ve) - max(c_start, vs)
        same_block = abs(vs - c_start) <= 0.55 and overlap >= 0.5 and len(current) < 6
        if same_block:
            current.append((idx, ln))
        else:
            blocks.append(current)
            current = [(idx, ln)]

    if current:
        blocks.append(current)

    block_records: list[tuple[float, int, list[tuple[int, TargetLine]]]] = []
    for block in blocks:
        vs_vals = [
            float(ln.visibility_start)
            for _, ln in block
            if ln.visibility_start is not None
        ]
        block_vs_key = min(vs_vals) if vs_vals else float(block[0][1].start)
        block_records.append((block_vs_key, block[0][0], block))

    # True block-first prototype: sequence blocks by visibility onset (screen chronology),
    # then preserve row order within each block.
    block_records.sort(key=lambda rec: (rec[0], rec[1]))

    sequenced: list[TargetLine] = []
    for _block_vs_key, _first_idx, block in block_records:
        if len(block) == 1:
            sequenced.append(block[0][1])
            continue
        ys = [float(ln.y) for _, ln in block]
        if max(ys) - min(ys) < 20.0:
            sequenced.extend([ln for _, ln in block])
            continue
        ordered_block = sorted(
            block,
            key=lambda p: (
                float(p[1].y),
                (
                    _line_start(p[1])
                    if _line_start(p[1]) is not None
                    else float(p[1].start)
                ),
                p[0],
            ),
        )
        sequenced.extend([ln for _, ln in ordered_block])

    if len(sequenced) != len(target_lines):
        return
    for i, ln in enumerate(sequenced):
        ln.block_order_hint = i


def _demote_fragment_lines_within_clean_blocks(target_lines: List[TargetLine]) -> None:
    """Move short fragment rows after fuller rows within simple screen blocks.

    This preserves lines but improves block-local ordering before output assembly.
    """
    if len(target_lines) < 2:
        return

    i = 0
    while i < len(target_lines):
        ln0 = target_lines[i]
        if ln0.visibility_start is None or ln0.visibility_end is None:
            i += 1
            continue
        vs0 = float(ln0.visibility_start)
        ve0 = float(ln0.visibility_end)
        block = [target_lines[i]]
        j = i + 1
        while j < len(target_lines):
            ln = target_lines[j]
            if ln.visibility_start is None or ln.visibility_end is None:
                break
            vs = float(ln.visibility_start)
            ve = float(ln.visibility_end)
            overlap = min(ve0, ve) - max(vs0, vs)
            if abs(vs - vs0) > 0.45 or abs(ve - ve0) > 3.0 or overlap < 0.75:
                break
            block.append(ln)
            j += 1

        if 2 <= len(block) <= 5:
            by_y = sorted(block, key=lambda ln: float(ln.y))
            fragment_flags: dict[int, bool] = {}
            for ln in by_y:
                fragment_flags[id(ln)] = any(
                    other is not ln and _is_fragment_of_line(ln, other)
                    for other in by_y
                )
            if any(fragment_flags.values()):
                reordered = sorted(
                    block,
                    key=lambda ln: (
                        1 if fragment_flags.get(id(ln), False) else 0,
                        float(ln.y),
                        _line_start(ln) or float(ln.start),
                    ),
                )
                if reordered != block:
                    target_lines[i:j] = reordered
                    for k, ln in enumerate(target_lines):
                        ln.line_index = k + 1
                        ln.block_order_hint = k
        i = max(i + 1, j)


def _merge_prefix_fragment_rows_in_clean_blocks(  # noqa: C901
    target_lines: List[TargetLine],
) -> None:
    """Conservatively merge split lyric rows like 'But baby I've been' + 'I've been'.

    This targets basic karaoke blocks where OCR split one logical row into a prefix row
    and a repeated fragment row on an adjacent row.
    """
    if len(target_lines) < 2:
        return

    def _top_variant_text(ln: TargetLine) -> Optional[str]:
        meta = (
            ln.reconstruction_meta if isinstance(ln.reconstruction_meta, dict) else None
        )
        if not meta:
            return None
        variants = meta.get("top_text_variants")
        if not isinstance(variants, list) or not variants:
            return None
        first = variants[0]
        if isinstance(first, dict):
            txt = first.get("text")
            if isinstance(txt, str) and txt.strip():
                return txt.strip()
        return None

    i = 0
    while i < len(target_lines):
        ln0 = target_lines[i]
        if ln0.visibility_start is None or ln0.visibility_end is None:
            i += 1
            continue
        vs0 = float(ln0.visibility_start)
        ve0 = float(ln0.visibility_end)
        block = [target_lines[i]]
        j = i + 1
        while j < len(target_lines):
            ln = target_lines[j]
            if ln.visibility_start is None or ln.visibility_end is None:
                break
            vs = float(ln.visibility_start)
            ve = float(ln.visibility_end)
            overlap = min(ve0, ve) - max(vs0, vs)
            if abs(vs - vs0) > 0.45 or abs(ve - ve0) > 3.0 or overlap < 0.75:
                break
            block.append(ln)
            j += 1

        if 2 <= len(block) <= 5:
            by_y = sorted(block, key=lambda ln: float(ln.y))
            for upper, lower in zip(by_y, by_y[1:]):
                upper_tokens = _token_list(upper)
                lower_tokens = _token_list(lower)
                if len(lower_tokens) != 2:
                    continue
                if len(upper_tokens) < 3 or len(upper_tokens) > 6:
                    continue
                # Typical pattern: upper ends with the repeated fragment ("i've been"),
                # lower is exactly that fragment; merge into upper and drop lower.
                if upper_tokens[-2:] != lower_tokens:
                    continue
                if (
                    upper.visibility_end is not None
                    and upper.visibility_start is not None
                ):
                    if (
                        float(upper.visibility_end) - float(upper.visibility_start)
                    ) > 1.0:
                        # Only merge very short lead rows (likely split row), not stable rows.
                        continue
                merged_text = " ".join(upper.words + lower.words)
                # Require evidence that the merged text (or close variant) exists elsewhere in the song.
                found_support = False
                merged_norm = normalize_text_basic(merged_text)
                for other in target_lines:
                    if other is upper or other is lower:
                        continue
                    other_norm = normalize_text_basic(" ".join(other.words))
                    if not other_norm:
                        continue
                    if (
                        merged_norm == other_norm
                        or merged_norm in other_norm
                        or other_norm in merged_norm
                    ):
                        if len(other_norm.split()) >= len(merged_norm.split()):
                            found_support = True
                            break
                if not found_support:
                    continue

                # Apply merge: append lower words/timings to upper, then clear lower so output drops it.
                upper.words = upper.words + lower.words
                if upper.word_rois and lower.word_rois:
                    upper.word_rois = list(upper.word_rois) + list(lower.word_rois)
                if upper.word_starts and lower.word_starts:
                    upper.word_starts = list(upper.word_starts) + list(
                        lower.word_starts
                    )
                if upper.word_ends and lower.word_ends:
                    upper.word_ends = list(upper.word_ends) + list(lower.word_ends)
                if upper.word_confidences and lower.word_confidences:
                    upper.word_confidences = list(upper.word_confidences) + list(
                        lower.word_confidences
                    )
                upper.text = " ".join(upper.words)

                lower.words = []
                lower.text = ""
                lower.word_starts = []
                lower.word_ends = []
                lower.word_confidences = []
                lower.word_rois = []
                lower.end = lower.start
                break

        i = max(i + 1, j)
