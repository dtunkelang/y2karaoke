"""Low-FPS visual timing refinement pipeline orchestration."""

from __future__ import annotations

import os
from typing import Any, Callable, List, Optional, Tuple

import numpy as np

from ..models import TargetLine
from .extractor_mode import ResolvedVisualExtractorMode


def run_low_fps_refinement_pipeline(  # noqa: C901
    *,
    cap: Any,
    target_lines: List[TargetLine],
    roi_rect: tuple[int, int, int, int],
    sample_fps: float,
    extractor_mode: ResolvedVisualExtractorMode,
    is_block_first_multi_cycle_line: Callable[[TargetLine], bool],
    build_line_refinement_jobs: Callable[..., List[Tuple[TargetLine, float, float]]],
    merge_line_refinement_jobs: Callable[
        ..., List[Tuple[float, float, List[Tuple[TargetLine, float, float]]]]
    ],
    read_window_frames_sampled: Callable[
        ..., List[Tuple[float, np.ndarray, np.ndarray]]
    ],
    slice_frames_for_window: Callable[..., List[Tuple[float, np.ndarray, np.ndarray]]],
    compute_line_min_start_time: Callable[..., Optional[float]],
    detect_line_highlight_with_confidence: Callable[
        ..., Tuple[Optional[float], Optional[float], float]
    ],
    maybe_adjust_detected_line_start_with_visibility_hint: Callable[
        ..., Tuple[Optional[float], Optional[float], float]
    ],
    assign_line_level_word_timings: Callable[
        [TargetLine, Optional[float], Optional[float], float], None
    ],
    apply_persistent_block_highlight_order: Callable[..., None],
    assign_surrogate_timings_for_unresolved_overlap_blocks: Callable[..., None],
    retime_late_first_lines_in_shared_visibility_blocks: Callable[..., None],
    retime_compressed_shared_visibility_blocks: Callable[..., None],
    promote_unresolved_first_repeated_lines: Callable[..., None],
    compress_overlong_sparse_line_timings: Callable[..., None],
    trace_refinement_snapshot: Callable[
        [str, List[Tuple[TargetLine, float, float]]], None
    ],
    retime_large_gaps_with_early_visibility: Callable[..., None],
    retime_followups_in_short_lead_shared_visibility_runs: Callable[..., None],
    rebalance_two_followups_after_short_lead: Callable[..., None],
    rebalance_early_lead_shared_visibility_runs: Callable[..., None],
    shrink_overlong_leads_in_dense_shared_visibility_runs: Callable[..., None],
    retime_dense_runs_after_overlong_lead: Callable[..., None],
    pull_dense_short_runs_toward_previous_anchor: Callable[..., None],
    retime_repeated_blocks_with_long_tail_gap: Callable[..., None],
    pull_late_first_lines_in_alternating_repeated_blocks: Callable[..., None],
    clamp_line_ends_to_visibility_windows: Callable[..., None],
    pull_lines_earlier_after_visibility_transitions: Callable[..., None],
    retime_short_interstitial_lines_between_anchors: Callable[..., None],
    rebalance_middle_lines_in_four_line_shared_visibility_runs: Callable[..., None],
    trace_clean_blocks: Callable[[str, List[TargetLine]], None],
    apply_block_first_prototype_ordering: Callable[[List[TargetLine]], bool | None],
    merge_prefix_fragment_rows_in_clean_blocks: Callable[[List[TargetLine]], None],
    demote_fragment_lines_within_clean_blocks: Callable[[List[TargetLine]], None],
    retime_clean_screen_blocks_by_vertical_order: Callable[..., None],
    reorder_clean_screen_blocks_target_lines: Callable[[List[TargetLine]], None],
    assign_block_sequence_hints_from_visibility: Callable[[List[TargetLine]], None],
) -> None:
    jobs = build_line_refinement_jobs(
        target_lines,
        lead_in_sec=10.0,
        tail_sec=1.0,
    )
    groups = merge_line_refinement_jobs(jobs, max_group_duration_sec=90.0)
    last_assigned_start: Optional[float] = None
    last_assigned_visibility_end: Optional[float] = None
    has_multi_cycle_block_first = any(
        is_block_first_multi_cycle_line(ln) for ln in target_lines
    )

    for g_start, g_end, g_jobs in groups:
        group_frames = read_window_frames_sampled(
            cap,
            v_start=g_start,
            v_end=g_end,
            roi_rect=roi_rect,
            sample_fps=sample_fps,
        )
        if len(group_frames) < 6:
            continue
        group_times = [frame[0] for frame in group_frames]

        for ln, v_start, v_end in g_jobs:
            if is_block_first_multi_cycle_line(ln):
                assign_line_level_word_timings(
                    ln,
                    float(ln.start),
                    float(ln.end) if ln.end is not None else float(ln.start) + 0.3,
                    0.35,
                )
                if ln.word_starts and ln.word_starts[0] is not None:
                    last_assigned_start = float(ln.word_starts[0])
                    if ln.visibility_end is not None:
                        last_assigned_visibility_end = float(ln.visibility_end)
                continue
            line_frames = slice_frames_for_window(
                group_frames,
                group_times,
                v_start=v_start,
                v_end=v_end,
            )
            if len(line_frames) < 6:
                continue
            line_min_start = compute_line_min_start_time(
                ln,
                last_assigned_start=last_assigned_start,
                last_assigned_visibility_end=last_assigned_visibility_end,
            )
            c_bg_line = np.mean(
                [
                    np.mean(f[1], axis=(0, 1))
                    for f in line_frames[: min(10, len(line_frames))]
                ],
                axis=0,
            )
            line_s, line_e, line_conf = detect_line_highlight_with_confidence(
                ln,
                line_frames,
                c_bg_line,
                min_start_time=line_min_start,
                require_full_cycle=True,
            )
            if line_s is None and line_e is None:
                line_s, line_e, line_conf = detect_line_highlight_with_confidence(
                    ln,
                    line_frames,
                    c_bg_line,
                    min_start_time=line_min_start,
                    require_full_cycle=False,
                )
                if line_s is None and line_e is None:
                    continue
            line_s, line_e, line_conf = (
                maybe_adjust_detected_line_start_with_visibility_hint(
                    ln,
                    detected_start=line_s,
                    detected_end=line_e,
                    detected_confidence=line_conf,
                    group_frames=group_frames,
                    group_times=group_times,
                )
            )
            assign_line_level_word_timings(ln, line_s, line_e, line_conf)
            if ln.word_starts and ln.word_starts[0] is not None:
                last_assigned_start = float(ln.word_starts[0])
                if ln.visibility_end is not None:
                    last_assigned_visibility_end = float(ln.visibility_end)
            elif line_s is not None:
                last_assigned_start = float(line_s)
                if ln.visibility_end is not None:
                    last_assigned_visibility_end = float(ln.visibility_end)

        mutable_g_jobs = [
            job for job in g_jobs if not is_block_first_multi_cycle_line(job[0])
        ]
        if mutable_g_jobs:
            apply_persistent_block_highlight_order(
                mutable_g_jobs, group_frames, group_times
            )
            assign_surrogate_timings_for_unresolved_overlap_blocks(
                mutable_g_jobs,
                group_frames=group_frames,
                group_times=group_times,
            )
        trace_refinement_snapshot("group_after_overlap", g_jobs)
        if (
            os.environ.get("Y2K_VISUAL_DISABLE_REFINE_SHARED_VIS_HEURISTICS", "0")
            != "1"
        ):
            if mutable_g_jobs:
                retime_late_first_lines_in_shared_visibility_blocks(mutable_g_jobs)
                retime_compressed_shared_visibility_blocks(mutable_g_jobs)
        if os.environ.get("Y2K_VISUAL_DISABLE_REFINE_REPEAT_PROMOTION", "0") != "1":
            if mutable_g_jobs:
                promote_unresolved_first_repeated_lines(mutable_g_jobs)
        if mutable_g_jobs:
            compress_overlong_sparse_line_timings(mutable_g_jobs)
        trace_refinement_snapshot("group_after_legacy_passes", g_jobs)
        for ln, _, _ in g_jobs:
            if ln.word_starts and ln.word_starts[0] is not None:
                if last_assigned_start is None:
                    last_assigned_start = float(ln.word_starts[0])
                else:
                    last_assigned_start = max(
                        last_assigned_start, float(ln.word_starts[0])
                    )
                if ln.visibility_end is not None:
                    vis_end = float(ln.visibility_end)
                    if last_assigned_visibility_end is None:
                        last_assigned_visibility_end = vis_end
                    else:
                        last_assigned_visibility_end = max(
                            last_assigned_visibility_end, vis_end
                        )

    trace_refinement_snapshot("pre_global_postpasses", jobs)
    mutable_jobs = [job for job in jobs if not is_block_first_multi_cycle_line(job[0])]
    if mutable_jobs:
        retime_large_gaps_with_early_visibility(mutable_jobs)
    if os.environ.get("Y2K_VISUAL_DISABLE_REFINE_SHARED_VIS_HEURISTICS", "0") != "1":
        if mutable_jobs:
            retime_followups_in_short_lead_shared_visibility_runs(mutable_jobs)
            rebalance_two_followups_after_short_lead(mutable_jobs)
            rebalance_early_lead_shared_visibility_runs(mutable_jobs)
            shrink_overlong_leads_in_dense_shared_visibility_runs(mutable_jobs)
            retime_dense_runs_after_overlong_lead(mutable_jobs)
            pull_dense_short_runs_toward_previous_anchor(mutable_jobs)
    if os.environ.get("Y2K_VISUAL_DISABLE_REFINE_REPEAT_HEURISTICS", "0") != "1":
        if mutable_jobs:
            retime_repeated_blocks_with_long_tail_gap(mutable_jobs)
            pull_late_first_lines_in_alternating_repeated_blocks(mutable_jobs)
    trace_refinement_snapshot("post_global_legacy", jobs)
    if mutable_jobs:
        clamp_line_ends_to_visibility_windows(mutable_jobs)
        pull_lines_earlier_after_visibility_transitions(mutable_jobs)
        retime_short_interstitial_lines_between_anchors(mutable_jobs)
    if os.environ.get("Y2K_VISUAL_DISABLE_REFINE_SHARED_VIS_HEURISTICS", "0") != "1":
        if mutable_jobs:
            rebalance_middle_lines_in_four_line_shared_visibility_runs(mutable_jobs)
    trace_clean_blocks("pre_clean_block", target_lines)
    use_block_first_proto = extractor_mode == "block-first" or (
        os.environ.get("Y2K_VISUAL_BLOCK_FIRST_PROTOTYPE", "0") == "1"
    )
    if use_block_first_proto:
        if not has_multi_cycle_block_first:
            apply_block_first_prototype_ordering(target_lines)
    else:
        merge_prefix_fragment_rows_in_clean_blocks(target_lines)
        demote_fragment_lines_within_clean_blocks(target_lines)
        retime_clean_screen_blocks_by_vertical_order(jobs)
        reorder_clean_screen_blocks_target_lines(target_lines)
        assign_block_sequence_hints_from_visibility(target_lines)
    trace_clean_blocks("post_clean_block", target_lines)
    trace_refinement_snapshot("post_clean_block", jobs)
