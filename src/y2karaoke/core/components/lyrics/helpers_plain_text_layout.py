"""Plain-text clip layout helpers shared by lyrics helpers."""

from __future__ import annotations

from typing import List

from ...models import Line
from .lyrics_clip_layout_helpers import (
    apply_short_setup_repeated_hook_layout as _apply_short_setup_repeated_hook_layout,
    apply_special_plain_text_layout as _apply_special_plain_text_layout,
    is_alternating_hook_chorus_clip as _is_alternating_hook_chorus_clip,
    is_dominant_repetition_run_clip as _is_dominant_repetition_run_clip,
    is_four_line_late_tail_chorus_clip as _is_four_line_late_tail_chorus_clip,
    is_four_line_staggered_chorus_clip as _is_four_line_staggered_chorus_clip,
    is_repeated_five_line_chorus_clip as _is_repeated_five_line_chorus_clip,
    is_seven_line_repeated_hook_bridge_clip as _is_seven_line_repeated_hook_bridge_clip,
    is_short_setup_repeated_hook_clip as _is_short_setup_repeated_hook_clip,
    is_short_title_chorus_clip as _is_short_title_chorus_clip,
    is_staggered_compact_hook_clip as _is_staggered_compact_hook_clip,
    is_three_line_call_response_clip as _is_three_line_call_response_clip,
    is_three_line_subset_hook_clip as _is_three_line_subset_hook_clip,
)


def _plain_text_layout_shape_flags(
    *,
    populated_lines: List[Line],
    duration: float,
) -> dict[str, bool]:
    return {
        "short_title_chorus_clip": _is_short_title_chorus_clip(
            populated_lines=populated_lines,
            duration=duration,
        ),
        "four_line_staggered_chorus_clip": _is_four_line_staggered_chorus_clip(
            populated_lines=populated_lines,
            duration=duration,
        ),
        "four_line_late_tail_chorus_clip": _is_four_line_late_tail_chorus_clip(
            populated_lines=populated_lines,
            duration=duration,
        ),
        "alternating_hook_chorus_clip": _is_alternating_hook_chorus_clip(
            populated_lines=populated_lines,
            duration=duration,
        ),
        "repeated_five_line_chorus_clip": _is_repeated_five_line_chorus_clip(
            populated_lines=populated_lines,
            duration=duration,
        ),
        "seven_line_repeated_hook_bridge_clip": (
            _is_seven_line_repeated_hook_bridge_clip(
                populated_lines=populated_lines,
                duration=duration,
            )
        ),
        "staggered_compact_hook_clip": _is_staggered_compact_hook_clip(
            populated_lines=populated_lines,
            duration=duration,
        ),
        "three_line_call_response_clip": _is_three_line_call_response_clip(
            populated_lines=populated_lines,
            duration=duration,
        ),
        "three_line_subset_hook_clip": _is_three_line_subset_hook_clip(
            populated_lines=populated_lines,
            duration=duration,
        ),
        "short_setup_repeated_hook_clip": _is_short_setup_repeated_hook_clip(
            populated_lines=populated_lines,
            duration=duration,
        ),
        "dominant_repetition_run_clip": _is_dominant_repetition_run_clip(
            populated_lines=populated_lines,
            duration=duration,
        ),
    }


def _apply_special_plain_text_layout_if_any(
    *,
    lines: List[Line],
    populated_lines: List[Line],
    duration: float,
    dense_short_verse_clip: bool,
    layout_flags: dict[str, bool],
    estimate_singing_duration_fn,
    apply_weighted_line_layout_fn,
) -> List[Line] | None:
    return _apply_special_plain_text_layout(
        lines=lines,
        populated_lines=populated_lines,
        duration=duration,
        short_title_chorus_clip=layout_flags["short_title_chorus_clip"],
        four_line_staggered_chorus_clip=layout_flags["four_line_staggered_chorus_clip"],
        four_line_late_tail_chorus_clip=layout_flags["four_line_late_tail_chorus_clip"],
        alternating_hook_chorus_clip=layout_flags["alternating_hook_chorus_clip"],
        repeated_five_line_chorus_clip=layout_flags["repeated_five_line_chorus_clip"],
        seven_line_repeated_hook_bridge_clip=layout_flags[
            "seven_line_repeated_hook_bridge_clip"
        ],
        staggered_compact_hook_clip=layout_flags["staggered_compact_hook_clip"],
        three_line_call_response_clip=layout_flags["three_line_call_response_clip"],
        three_line_subset_hook_clip=layout_flags["three_line_subset_hook_clip"],
        short_setup_repeated_hook_clip=layout_flags["short_setup_repeated_hook_clip"],
        dense_short_verse_clip=dense_short_verse_clip,
        estimate_singing_duration_fn=estimate_singing_duration_fn,
        apply_weighted_line_layout_fn=apply_weighted_line_layout_fn,
    )


def _apply_anchor_specific_plain_text_layout(
    *,
    lines: List[Line],
    populated_lines: List[Line],
    duration: float,
    anchor_start: float,
    desired_end: float,
    layout_flags: dict[str, bool],
    estimate_singing_duration_fn,
    apply_weighted_line_layout_fn,
) -> List[Line] | None:
    anchor_special_flags = {
        **layout_flags,
        "short_title_chorus_clip": False,
        "short_setup_repeated_hook_clip": False,
        "dominant_repetition_run_clip": False,
    }
    special_layout = _apply_special_plain_text_layout_if_any(
        lines=lines,
        populated_lines=populated_lines,
        duration=duration,
        dense_short_verse_clip=False,
        layout_flags=anchor_special_flags,
        estimate_singing_duration_fn=estimate_singing_duration_fn,
        apply_weighted_line_layout_fn=apply_weighted_line_layout_fn,
    )
    if special_layout is not None:
        return special_layout
    if layout_flags["short_setup_repeated_hook_clip"]:
        return _apply_short_setup_repeated_hook_layout(
            lines=lines,
            populated_lines=populated_lines,
            anchor_start=anchor_start,
            desired_end=desired_end,
            apply_weighted_line_layout_fn=apply_weighted_line_layout_fn,
        )
    return None


def _special_plain_text_or_spread(
    *,
    lines: List[Line],
    populated_lines: List[Line],
    duration: float,
    target_duration: float,
    dense_short_verse_clip: bool,
    layout_flags: dict[str, bool],
    estimate_singing_duration_fn,
    apply_weighted_line_layout_fn,
    spread_lines_across_target_duration_fn,
) -> List[Line]:
    special_layout = _apply_special_plain_text_layout_if_any(
        lines=lines,
        populated_lines=populated_lines,
        duration=duration,
        dense_short_verse_clip=dense_short_verse_clip,
        layout_flags=layout_flags,
        estimate_singing_duration_fn=estimate_singing_duration_fn,
        apply_weighted_line_layout_fn=apply_weighted_line_layout_fn,
    )
    if special_layout is not None:
        return special_layout
    return spread_lines_across_target_duration_fn(lines, target_duration)
