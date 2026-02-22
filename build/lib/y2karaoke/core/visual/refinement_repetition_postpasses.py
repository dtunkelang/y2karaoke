"""Repeated-block timing post-pass helpers."""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple, cast

from ..models import TargetLine


def _retime_repeated_blocks_with_long_tail_gap(
    g_jobs: List[Tuple[TargetLine, float, float]],
    *,
    line_start_fn: Callable[[TargetLine], Optional[float]],
    line_end_fn: Callable[[TargetLine], Optional[float]],
    canonical_line_text_fn: Callable[[TargetLine], str],
    assign_line_level_word_timings_fn: Callable[
        [TargetLine, Optional[float], Optional[float], float], None
    ],
) -> None:
    """Delay repeated-line sub-blocks when a long tail gap indicates late highlights."""
    line_order = [ln for ln, _, _ in g_jobs]
    for i in range(len(line_order) - 3):
        a = line_order[i]
        b = line_order[i + 1]
        c = line_order[i + 2]
        d = line_order[i + 3]
        a_s, a_e = line_start_fn(a), line_end_fn(a)
        b_s, b_e = line_start_fn(b), line_end_fn(b)
        c_s, c_e = line_start_fn(c), line_end_fn(c)
        d_s, d_e = line_start_fn(d), line_end_fn(d)
        if None in (a_s, a_e, b_s, b_e, c_s, c_e, d_s, d_e):
            continue

        a_sf = float(cast(float, a_s))
        a_ef = float(cast(float, a_e))
        b_sf = float(cast(float, b_s))
        b_ef = float(cast(float, b_e))
        c_sf = float(cast(float, c_s))
        c_ef = float(cast(float, c_e))
        d_sf = float(cast(float, d_s))

        texts = [canonical_line_text_fn(x) for x in [a, b, c, d]]
        if texts[0] != texts[1]:
            continue
        if len(set(t for t in texts if t)) >= 4:
            continue
        if not ((b_sf - a_sf) <= 3.0 and (c_sf - b_sf) <= 1.5):
            continue
        tail_gap = d_sf - c_ef
        if tail_gap < 6.0:
            continue

        dur_a = max(0.8, a_ef - a_sf)
        dur_b = max(0.7, b_ef - b_sf)
        dur_c = max(0.7, c_ef - c_sf)
        a_target = a_sf
        if i > 0:
            prev = line_order[i - 1]
            p_e = line_end_fn(prev)
            if p_e is not None and (a_sf - float(p_e)) < 1.2:
                a_target = max(a_target, float(p_e) + 2.5)
        a_target_end = a_target + dur_a
        b_target = max(a_target_end + 1.0, d_sf - 4.0)
        c_target = max(b_target + 1.5, d_sf - 2.0)
        b_target = max(b_target, a_target + 0.8)
        c_target = min(c_target, d_sf - 0.4)
        b_target = min(b_target, c_target - 0.8)
        if a_target <= a_sf + 0.3 and b_target <= b_sf + 0.4 and c_target <= c_sf + 0.4:
            continue

        assign_line_level_word_timings_fn(a, a_target, a_target + dur_a, 0.42)
        assign_line_level_word_timings_fn(b, b_target, b_target + dur_b, 0.42)
        assign_line_level_word_timings_fn(c, c_target, c_target + dur_c, 0.42)


def _pull_late_first_lines_in_alternating_repeated_blocks(
    g_jobs: List[Tuple[TargetLine, float, float]],
    *,
    line_start_fn: Callable[[TargetLine], Optional[float]],
    line_end_fn: Callable[[TargetLine], Optional[float]],
    canonical_line_text_fn: Callable[[TargetLine], str],
    assign_line_level_word_timings_fn: Callable[
        [TargetLine, Optional[float], Optional[float], float], None
    ],
) -> None:
    """Correct late first onsets in A/B/A/B repeated blocks with shared visibility."""
    line_order = [ln for ln, _, _ in g_jobs]
    for i in range(len(line_order) - 3):
        a = line_order[i]
        b = line_order[i + 1]
        c = line_order[i + 2]
        d = line_order[i + 3]
        a_s, a_e = line_start_fn(a), line_end_fn(a)
        b_s, b_e = line_start_fn(b), line_end_fn(b)
        c_s, c_e = line_start_fn(c), line_end_fn(c)
        d_s, d_e = line_start_fn(d), line_end_fn(d)
        if None in (a_s, a_e, b_s, b_e, c_s, c_e, d_s, d_e):
            continue
        if a.visibility_start is None or a.visibility_end is None:
            continue

        a_sf = float(cast(float, a_s))
        a_ef = float(cast(float, a_e))
        b_sf = float(cast(float, b_s))
        b_ef = float(cast(float, b_e))
        c_sf = float(cast(float, c_s))
        vis_start = float(a.visibility_start)
        vis_end = float(a.visibility_end)
        if (vis_end - vis_start) < 10.0:
            continue

        ta = canonical_line_text_fn(a)
        tb = canonical_line_text_fn(b)
        tc = canonical_line_text_fn(c)
        td = canonical_line_text_fn(d)
        if not ta or not tb or ta != tc or tb != td or ta == tb:
            continue

        if (a_sf - vis_start) < 1.2:
            continue
        if (c_sf - b_ef) < 4.0:
            continue

        a_dur = max(0.8, a_ef - a_sf)
        target_a = max(vis_start + 0.05, a_sf - 1.8)
        target_a = min(target_a, b_sf - a_dur - 0.1)
        if target_a >= a_sf - 0.25:
            continue
        assign_line_level_word_timings_fn(a, target_a, target_a + a_dur, 0.42)
