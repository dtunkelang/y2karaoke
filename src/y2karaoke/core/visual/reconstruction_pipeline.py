from __future__ import annotations

from typing import Any, Callable

from ..models import TargetLine
from .reconstruction_frame_accumulation import accumulate_persistent_lines_from_frames
from .reconstruction_deduplication import deduplicate_persistent_lines
from .reconstruction_target_conversion import convert_persistent_lines_to_target_lines

EntriesPass = Callable[[list[dict[str, Any]]], list[dict[str, Any]]]
FrameFilter = Callable[[list[dict[str, Any]]], list[dict[str, Any]]]
SnapFn = Callable[[float], float]


def reconstruct_lyrics_from_visuals(  # noqa: C901
    raw_frames: list[dict[str, Any]],
    *,
    filter_static_overlay_words: FrameFilter,
    merge_overlapping_same_lane_duplicates: EntriesPass,
    merge_short_same_lane_reentries: EntriesPass,
    expand_overlapped_same_text_repetitions: EntriesPass,
    extrapolate_mirrored_lane_cycles: EntriesPass,
    split_persistent_line_epochs_from_context_transitions: EntriesPass,
    suppress_short_duplicate_reentries: EntriesPass,
    collapse_short_refrain_noise: EntriesPass,
    filter_intro_non_lyrics: Callable[[list[dict[str, Any]], Optional[str]], list[dict[str, Any]]],
    suppress_bottom_fragment_families: EntriesPass,
    snap_fn: SnapFn,
    artist: Optional[str] = None,
) -> list[TargetLine]:
    committed = accumulate_persistent_lines_from_frames(
        raw_frames, filter_static_overlay_words=filter_static_overlay_words
    )

    unique = deduplicate_persistent_lines(committed)

    unique = merge_overlapping_same_lane_duplicates(unique)
    unique = merge_short_same_lane_reentries(unique)
    unique = expand_overlapped_same_text_repetitions(unique)
    unique = extrapolate_mirrored_lane_cycles(unique)
    unique = split_persistent_line_epochs_from_context_transitions(unique)
    unique = suppress_short_duplicate_reentries(unique)
    unique = collapse_short_refrain_noise(unique)
    unique = filter_intro_non_lyrics(unique, artist)
    unique = suppress_bottom_fragment_families(unique)

    return convert_persistent_lines_to_target_lines(unique, snap_fn=snap_fn)
