"""Tail-end dense-run retiming helpers for visibility-based refinement."""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple, cast

from ..models import TargetLine


def shrink_overlong_leads_in_dense_shared_visibility_runs(  # noqa: C901
    g_jobs: List[Tuple[TargetLine, float, float]],
    *,
    line_start_fn: Callable[[TargetLine], Optional[float]],
    line_end_fn: Callable[[TargetLine], Optional[float]],
    canonical_line_text_fn: Callable[[TargetLine], str],
    assign_line_level_word_timings_fn: Callable[
        [TargetLine, Optional[float], Optional[float], float], None
    ],
) -> None:
    line_order = [ln for ln, _, _ in g_jobs]
    n = len(line_order)
    i = 0
    while i < n:
        base = line_order[i]
        if base.visibility_start is None or base.visibility_end is None:
            i += 1
            continue
        run = [base]
        j = i + 1
        while j < n:
            ln = line_order[j]
            if ln.visibility_start is None or ln.visibility_end is None:
                break
            if (
                abs(float(ln.visibility_start) - float(base.visibility_start)) <= 1.0
                and abs(float(ln.visibility_end) - float(base.visibility_end)) <= 2.0
            ):
                run.append(ln)
                j += 1
                continue
            break

        if len(run) >= 4:
            starts = [line_start_fn(ln) for ln in run]
            ends = [line_end_fn(ln) for ln in run]
            if all(s is not None for s in starts) and all(e is not None for e in ends):
                texts = [canonical_line_text_fn(ln) for ln in run]
                nonempty = [t for t in texts if t]
                if len(set(nonempty)) < len(nonempty):
                    i = j if j > i else i + 1
                    continue
                starts_f = [float(s) for s in starts if s is not None]
                ends_f = [float(e) for e in ends if e is not None]
                vis_start = float(base.visibility_start)

                # If the whole run starts late relative to a nearby previous anchor,
                # pull the block earlier while respecting visibility floors.
                if i > 0:
                    prev_end = line_end_fn(line_order[i - 1])
                    if prev_end is not None:
                        vis_gap = vis_start - float(prev_end)
                        target_start = max(float(prev_end) + 0.2, vis_start - 1.0)
                        early_shift = starts_f[0] - target_start
                        if 0.4 <= early_shift <= 1.2 and -0.2 <= vis_gap <= 1.6:
                            max_shift_vis = early_shift
                            for ln, s in zip(run, starts_f):
                                if ln.visibility_start is None:
                                    continue
                                vis_floor = float(ln.visibility_start) - 1.0
                                max_shift_vis = min(max_shift_vis, s - vis_floor)
                            if max_shift_vis >= 0.4:
                                early_shift = min(early_shift, max_shift_vis)
                                starts_f = [s - early_shift for s in starts_f]
                                ends_f = [e - early_shift for e in ends_f]

                lead_dur = ends_f[0] - starts_f[0]
                follow_durs = [e - s for s, e in zip(starts_f[1:], ends_f[1:])]
                dense_follow = all(
                    (starts_f[k + 1] - starts_f[k]) <= 1.5
                    for k in range(1, min(len(starts_f) - 1, 3))
                )
                if (
                    lead_dur >= 2.0
                    and follow_durs
                    and dense_follow
                    and lead_dur >= (max(follow_durs[:2]) + 1.2)
                ):
                    lead_target = max(
                        0.85,
                        min(1.25, 0.22 * float(max(len(base.words), 1)) + 0.2),
                    )
                    new_lead_end = starts_f[0] + lead_target
                    shift = starts_f[1] - (new_lead_end + 0.2)

                    max_shift = shift
                    for ln, s in zip(run[1:], starts_f[1:]):
                        if ln.visibility_start is None:
                            continue
                        vis_floor = float(ln.visibility_start) - 1.0
                        max_shift = min(max_shift, s - vis_floor)
                    if shift >= 0.4 and max_shift >= 0.4:
                        shift = min(shift, max_shift)
                        assign_line_level_word_timings_fn(
                            base, starts_f[0], new_lead_end, 0.42
                        )
                        for ln, s, e in zip(run[1:], starts_f[1:], ends_f[1:]):
                            assign_line_level_word_timings_fn(
                                ln, s - shift, e - shift, 0.42
                            )
                elif i > 0:
                    # Persist the early-shift-only adjustment even if lead shrink is skipped.
                    for ln, s, e in zip(run, starts_f, ends_f):
                        assign_line_level_word_timings_fn(ln, s, e, 0.42)

        i = j if j > i else i + 1


def retime_dense_runs_after_overlong_lead(
    g_jobs: List[Tuple[TargetLine, float, float]],
    *,
    line_start_fn: Callable[[TargetLine], Optional[float]],
    line_end_fn: Callable[[TargetLine], Optional[float]],
    canonical_line_text_fn: Callable[[TargetLine], str],
    assign_line_level_word_timings_fn: Callable[
        [TargetLine, Optional[float], Optional[float], float], None
    ],
) -> None:
    line_order = [ln for ln, _, _ in g_jobs]
    for i in range(1, len(line_order) - 3):
        a = line_order[i]
        b = line_order[i + 1]
        c = line_order[i + 2]
        d = line_order[i + 3]
        prev = line_order[i - 1]

        a_s = line_start_fn(a)
        a_e = line_end_fn(a)
        b_s = line_start_fn(b)
        b_e = line_end_fn(b)
        c_s = line_start_fn(c)
        c_e = line_end_fn(c)
        d_s = line_start_fn(d)
        d_e = line_end_fn(d)
        p_e = line_end_fn(prev)
        if None in (a_s, a_e, b_s, b_e, c_s, c_e, d_s, d_e, p_e):
            continue

        a_sf = float(cast(float, a_s))
        a_ef = float(cast(float, a_e))
        b_sf = float(cast(float, b_s))
        b_ef = float(cast(float, b_e))
        c_sf = float(cast(float, c_s))
        c_ef = float(cast(float, c_e))
        d_sf = float(cast(float, d_s))
        d_ef = float(cast(float, d_e))
        p_ef = float(cast(float, p_e))

        lead_dur = a_ef - a_sf
        if lead_dur < 2.0:
            continue
        if abs(b_sf - a_ef) > 0.35:
            continue
        if (c_sf - b_sf) > 2.0 or (d_sf - c_sf) > 2.0:
            continue
        if (b_ef - b_sf) > 2.2 or (c_ef - c_sf) > 2.2 or (d_ef - d_sf) > 2.2:
            continue
        if (a_sf - p_ef) > 2.0:
            continue

        word_counts = [
            max(len(a.words), 1),
            max(len(b.words), 1),
            max(len(c.words), 1),
            max(len(d.words), 1),
        ]
        if max(word_counts) > 1.8 * min(word_counts):
            continue
        texts = [canonical_line_text_fn(x) for x in [a, b, c, d]]
        nonempty = [t for t in texts if t]
        if len(set(nonempty)) < len(nonempty):
            continue

        target_start = max(p_ef + 0.05, a_sf - 1.2)
        block_shift = min(1.2, max(0.0, a_sf - target_start))
        new_a_start = a_sf - block_shift
        target_lead_dur = max(0.85, min(1.1, 0.2 * float(word_counts[0]) + 0.1))
        new_a_end = new_a_start + target_lead_dur

        follower_shift = b_sf - (new_a_end + 0.1)
        follower_shift = min(1.8, max(0.0, follower_shift))
        if block_shift < 0.2 and follower_shift < 0.2:
            continue

        assign_line_level_word_timings_fn(a, new_a_start, new_a_end, 0.42)
        for ln, s_old, e_old in (
            (b, b_sf, b_ef),
            (c, c_sf, c_ef),
            (d, d_sf, d_ef),
        ):
            dur = max(0.6, e_old - s_old)
            proposed_start = s_old - follower_shift
            vis_floor: Optional[float] = None
            if ln.visibility_start is not None and ln.visibility_end is not None:
                vis_span = float(ln.visibility_end) - float(ln.visibility_start)
                # Long-lived lines often stay unhighlighted for a noticeable lead-in.
                # Keep aggressive dense-run pull-forwards from snapping to appearance.
                lead_lag_floor = 0.6 if vis_span >= 8.0 else 0.2
                vis_floor = float(ln.visibility_start) + lead_lag_floor
            if vis_floor is not None:
                proposed_start = max(proposed_start, vis_floor)
            assign_line_level_word_timings_fn(
                ln, proposed_start, proposed_start + dur, 0.42
            )


def pull_dense_short_runs_toward_previous_anchor(
    g_jobs: List[Tuple[TargetLine, float, float]],
    *,
    line_start_fn: Callable[[TargetLine], Optional[float]],
    line_end_fn: Callable[[TargetLine], Optional[float]],
    canonical_line_text_fn: Callable[[TargetLine], str],
    assign_line_level_word_timings_fn: Callable[
        [TargetLine, Optional[float], Optional[float], float], None
    ],
) -> None:
    line_order = [ln for ln, _, _ in g_jobs]
    for i in range(1, len(line_order) - 3):
        prev = line_order[i - 1]
        a = line_order[i]
        b = line_order[i + 1]
        c = line_order[i + 2]
        d = line_order[i + 3]

        p_s, p_e = line_start_fn(prev), line_end_fn(prev)
        a_s, a_e = line_start_fn(a), line_end_fn(a)
        b_s, b_e = line_start_fn(b), line_end_fn(b)
        c_s, c_e = line_start_fn(c), line_end_fn(c)
        d_s, d_e = line_start_fn(d), line_end_fn(d)
        if None in (p_s, p_e, a_s, a_e, b_s, b_e, c_s, c_e, d_s, d_e):
            continue

        p_sf = float(cast(float, p_s))
        p_ef = float(cast(float, p_e))
        a_sf = float(cast(float, a_s))
        a_ef = float(cast(float, a_e))
        b_sf = float(cast(float, b_s))
        b_ef = float(cast(float, b_e))
        c_sf = float(cast(float, c_s))
        c_ef = float(cast(float, c_e))
        d_sf = float(cast(float, d_s))
        d_ef = float(cast(float, d_e))

        prev_dur = p_ef - p_sf
        first_gap = a_sf - p_ef
        if prev_dur < 2.5 or not (0.7 <= first_gap <= 1.6):
            continue
        if any(
            (e - s) > 2.2
            for s, e in [(a_sf, a_ef), (b_sf, b_ef), (c_sf, c_ef), (d_sf, d_ef)]
        ):
            continue
        if not (
            0.8 <= (b_sf - a_sf) <= 2.2
            and 0.8 <= (c_sf - b_sf) <= 2.2
            and 0.8 <= (d_sf - c_sf) <= 2.2
        ):
            continue

        lengths = [max(len(x.words), 1) for x in [a, b, c, d]]
        if max(lengths) > 1.8 * min(lengths):
            continue
        texts = [canonical_line_text_fn(x) for x in [a, b, c, d]]
        nonempty = [t for t in texts if t]
        # Repeated-text blocks (e.g. repeated chorus/outro lines) can have
        # long on-screen dwell before highlight; avoid compressing them early.
        if len(set(nonempty)) < len(nonempty):
            continue

        shift = min(1.2, max(0.0, first_gap - 0.05))
        if shift < 0.35:
            continue

        assign_line_level_word_timings_fn(a, a_sf - shift, a_ef - shift, 0.42)
        assign_line_level_word_timings_fn(b, b_sf - shift, b_ef - shift, 0.42)
        assign_line_level_word_timings_fn(c, c_sf - shift, c_ef - shift, 0.42)
        assign_line_level_word_timings_fn(d, d_sf - shift, d_ef - shift, 0.42)
