from __future__ import annotations

import math
import re
from typing import List, Optional, Tuple

from ..models import TargetLine
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
