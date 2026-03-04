"""Dense shared-visibility run retiming post-pass helpers."""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from ..models import TargetLine
from .refinement_dense_visibility_passes import (
    cluster_unresolved_visibility_lines as _cluster_unresolved_visibility_lines_impl,
    retime_compressed_shared_visibility_blocks as _retime_compressed_shared_visibility_blocks_impl,
    retime_late_first_lines_in_shared_visibility_blocks as _retime_late_first_lines_in_shared_visibility_blocks_impl,
)
from .refinement_dense_repeat_postpasses import (
    compress_overlong_sparse_line_timings as _compress_overlong_sparse_line_timings_impl,
)
from .refinement_dense_repeat_postpasses import (
    promote_unresolved_first_repeated_lines as _promote_unresolved_first_repeated_lines_impl,
)
from .refinement_dense_repeat_postpasses import (
    rebalance_early_lead_shared_visibility_runs as _rebalance_early_lead_shared_visibility_runs_impl,
)
from .refinement_dense_tail_postpasses import (
    pull_dense_short_runs_toward_previous_anchor as _pull_dense_short_runs_toward_previous_anchor_impl,
)
from .refinement_dense_tail_postpasses import (
    retime_dense_runs_after_overlong_lead as _retime_dense_runs_after_overlong_lead_impl,
)
from .refinement_dense_tail_postpasses import (
    shrink_overlong_leads_in_dense_shared_visibility_runs as _shrink_overlong_leads_in_dense_shared_visibility_runs_impl,
)


def _cluster_unresolved_visibility_lines(
    lines: List[TargetLine],
) -> List[List[TargetLine]]:
    return _cluster_unresolved_visibility_lines_impl(lines)


def _retime_late_first_lines_in_shared_visibility_blocks(
    g_jobs: List[Tuple[TargetLine, float, float]],
    *,
    line_start_fn: Callable[[TargetLine], Optional[float]],
    line_end_fn: Callable[[TargetLine], Optional[float]],
    assign_line_level_word_timings_fn: Callable[
        [TargetLine, Optional[float], Optional[float], float], None
    ],
) -> None:
    _retime_late_first_lines_in_shared_visibility_blocks_impl(
        g_jobs,
        line_start_fn=line_start_fn,
        line_end_fn=line_end_fn,
        assign_line_level_word_timings_fn=assign_line_level_word_timings_fn,
    )


def _retime_compressed_shared_visibility_blocks(  # noqa: C901
    g_jobs: List[Tuple[TargetLine, float, float]],
    *,
    line_start_fn: Callable[[TargetLine], Optional[float]],
    line_end_fn: Callable[[TargetLine], Optional[float]],
    assign_line_level_word_timings_fn: Callable[
        [TargetLine, Optional[float], Optional[float], float], None
    ],
) -> None:
    _retime_compressed_shared_visibility_blocks_impl(
        g_jobs,
        line_start_fn=line_start_fn,
        line_end_fn=line_end_fn,
        assign_line_level_word_timings_fn=assign_line_level_word_timings_fn,
    )


def _promote_unresolved_first_repeated_lines(  # noqa: C901
    g_jobs: List[Tuple[TargetLine, float, float]],
    *,
    line_start_fn: Callable[[TargetLine], Optional[float]],
    line_end_fn: Callable[[TargetLine], Optional[float]],
    canonical_line_text_fn: Callable[[TargetLine], str],
    assign_line_level_word_timings_fn: Callable[
        [TargetLine, Optional[float], Optional[float], float], None
    ],
) -> None:
    _promote_unresolved_first_repeated_lines_impl(
        g_jobs,
        line_start_fn=line_start_fn,
        line_end_fn=line_end_fn,
        canonical_line_text_fn=canonical_line_text_fn,
        assign_line_level_word_timings_fn=assign_line_level_word_timings_fn,
    )


def _compress_overlong_sparse_line_timings(  # noqa: C901
    g_jobs: List[Tuple[TargetLine, float, float]],
    *,
    line_start_fn: Callable[[TargetLine], Optional[float]],
    assign_line_level_word_timings_fn: Callable[
        [TargetLine, Optional[float], Optional[float], float], None
    ],
) -> None:
    _compress_overlong_sparse_line_timings_impl(
        g_jobs,
        line_start_fn=line_start_fn,
        assign_line_level_word_timings_fn=assign_line_level_word_timings_fn,
    )


def _rebalance_early_lead_shared_visibility_runs(  # noqa: C901
    g_jobs: List[Tuple[TargetLine, float, float]],
    *,
    line_start_fn: Callable[[TargetLine], Optional[float]],
    line_end_fn: Callable[[TargetLine], Optional[float]],
    assign_line_level_word_timings_fn: Callable[
        [TargetLine, Optional[float], Optional[float], float], None
    ],
) -> None:
    _rebalance_early_lead_shared_visibility_runs_impl(
        g_jobs,
        line_start_fn=line_start_fn,
        line_end_fn=line_end_fn,
        assign_line_level_word_timings_fn=assign_line_level_word_timings_fn,
    )


def _shrink_overlong_leads_in_dense_shared_visibility_runs(  # noqa: C901
    g_jobs: List[Tuple[TargetLine, float, float]],
    *,
    line_start_fn: Callable[[TargetLine], Optional[float]],
    line_end_fn: Callable[[TargetLine], Optional[float]],
    canonical_line_text_fn: Callable[[TargetLine], str],
    assign_line_level_word_timings_fn: Callable[
        [TargetLine, Optional[float], Optional[float], float], None
    ],
) -> None:
    _shrink_overlong_leads_in_dense_shared_visibility_runs_impl(
        g_jobs,
        line_start_fn=line_start_fn,
        line_end_fn=line_end_fn,
        canonical_line_text_fn=canonical_line_text_fn,
        assign_line_level_word_timings_fn=assign_line_level_word_timings_fn,
    )


def _retime_dense_runs_after_overlong_lead(
    g_jobs: List[Tuple[TargetLine, float, float]],
    *,
    line_start_fn: Callable[[TargetLine], Optional[float]],
    line_end_fn: Callable[[TargetLine], Optional[float]],
    canonical_line_text_fn: Callable[[TargetLine], str],
    assign_line_level_word_timings_fn: Callable[
        [TargetLine, Optional[float], Optional[float], float], None
    ],
) -> None:
    _retime_dense_runs_after_overlong_lead_impl(
        g_jobs,
        line_start_fn=line_start_fn,
        line_end_fn=line_end_fn,
        canonical_line_text_fn=canonical_line_text_fn,
        assign_line_level_word_timings_fn=assign_line_level_word_timings_fn,
    )


def _pull_dense_short_runs_toward_previous_anchor(
    g_jobs: List[Tuple[TargetLine, float, float]],
    *,
    line_start_fn: Callable[[TargetLine], Optional[float]],
    line_end_fn: Callable[[TargetLine], Optional[float]],
    canonical_line_text_fn: Callable[[TargetLine], str],
    assign_line_level_word_timings_fn: Callable[
        [TargetLine, Optional[float], Optional[float], float], None
    ],
) -> None:
    _pull_dense_short_runs_toward_previous_anchor_impl(
        g_jobs,
        line_start_fn=line_start_fn,
        line_end_fn=line_end_fn,
        canonical_line_text_fn=canonical_line_text_fn,
        assign_line_level_word_timings_fn=assign_line_level_word_timings_fn,
    )
