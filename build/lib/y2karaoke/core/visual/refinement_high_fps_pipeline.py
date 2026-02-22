"""High-FPS refinement orchestration helper."""

from __future__ import annotations

from typing import Any, Callable, List, Tuple

from ..models import TargetLine

Jobs = List[Tuple[TargetLine, float, float]]
Frames = List[Tuple[float, Any, Any]]


def run_high_fps_refinement_pipeline(
    *,
    cap: Any,
    target_lines: List[TargetLine],
    roi_rect: tuple[int, int, int, int],
    build_line_refinement_jobs: Callable[..., Jobs],
    merge_line_refinement_jobs: Callable[..., List[Tuple[float, float, Jobs]]],
    read_window_frames: Callable[..., Frames],
    slice_frames_for_window: Callable[..., Frames],
    refine_line_with_frames: Callable[[TargetLine, Frames], None],
    apply_persistent_block_highlight_order: Callable[[Jobs, Frames, List[float]], None],
    assign_surrogate_timings_for_unresolved_overlap_blocks: Callable[..., None],
    retime_late_first_lines_in_shared_visibility_blocks: Callable[[Jobs], None],
    retime_compressed_shared_visibility_blocks: Callable[[Jobs], None],
    promote_unresolved_first_repeated_lines: Callable[[Jobs], None],
    compress_overlong_sparse_line_timings: Callable[[Jobs], None],
    retime_large_gaps_with_early_visibility: Callable[[Jobs], None],
    retime_followups_in_short_lead_shared_visibility_runs: Callable[[Jobs], None],
    rebalance_two_followups_after_short_lead: Callable[[Jobs], None],
    rebalance_early_lead_shared_visibility_runs: Callable[[Jobs], None],
    shrink_overlong_leads_in_dense_shared_visibility_runs: Callable[[Jobs], None],
    retime_dense_runs_after_overlong_lead: Callable[[Jobs], None],
    pull_dense_short_runs_toward_previous_anchor: Callable[[Jobs], None],
    retime_repeated_blocks_with_long_tail_gap: Callable[[Jobs], None],
    pull_late_first_lines_in_alternating_repeated_blocks: Callable[[Jobs], None],
    clamp_line_ends_to_visibility_windows: Callable[[Jobs], None],
    pull_lines_earlier_after_visibility_transitions: Callable[[Jobs], None],
    retime_short_interstitial_lines_between_anchors: Callable[[Jobs], None],
    rebalance_middle_lines_in_four_line_shared_visibility_runs: Callable[[Jobs], None],
) -> None:
    jobs = build_line_refinement_jobs(target_lines)
    groups = merge_line_refinement_jobs(jobs, max_group_duration_sec=90.0)

    for g_start, g_end, g_jobs in groups:
        group_frames = read_window_frames(
            cap,
            v_start=g_start,
            v_end=g_end,
            roi_rect=roi_rect,
        )
        if not group_frames:
            continue
        group_times = [frame[0] for frame in group_frames]

        for ln, v_start, v_end in g_jobs:
            line_frames = slice_frames_for_window(
                group_frames,
                group_times,
                v_start=v_start,
                v_end=v_end,
            )
            if len(line_frames) < 20:
                continue
            refine_line_with_frames(ln, line_frames)

        apply_persistent_block_highlight_order(g_jobs, group_frames, group_times)
        assign_surrogate_timings_for_unresolved_overlap_blocks(
            g_jobs,
            group_frames=group_frames,
            group_times=group_times,
        )
        retime_late_first_lines_in_shared_visibility_blocks(g_jobs)
        retime_compressed_shared_visibility_blocks(g_jobs)
        promote_unresolved_first_repeated_lines(g_jobs)
        compress_overlong_sparse_line_timings(g_jobs)

    retime_large_gaps_with_early_visibility(jobs)
    retime_followups_in_short_lead_shared_visibility_runs(jobs)
    rebalance_two_followups_after_short_lead(jobs)
    rebalance_early_lead_shared_visibility_runs(jobs)
    shrink_overlong_leads_in_dense_shared_visibility_runs(jobs)
    retime_dense_runs_after_overlong_lead(jobs)
    pull_dense_short_runs_toward_previous_anchor(jobs)
    retime_repeated_blocks_with_long_tail_gap(jobs)
    pull_late_first_lines_in_alternating_repeated_blocks(jobs)
    clamp_line_ends_to_visibility_windows(jobs)
    pull_lines_earlier_after_visibility_transitions(jobs)
    retime_short_interstitial_lines_between_anchors(jobs)
    rebalance_middle_lines_in_four_line_shared_visibility_runs(jobs)
