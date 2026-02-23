from __future__ import annotations

from typing import Any, Optional

from ..models import TargetLine
from .reconstruction_mirrored_cycles import (
    extrapolate_mirrored_lane_cycles as _extrapolate_mirrored_lane_cycles_impl,
)
from .reconstruction_mirrored_cycles import (
    is_candidate_for_mirrored_cycle as _is_candidate_for_mirrored_cycle_impl,
)
from .reconstruction_mirrored_cycles import (
    mirrored_cycle_candidate as _mirrored_cycle_candidate_impl,
)
from .reconstruction_intro_filters import (
    filter_intro_non_lyrics as _filter_intro_non_lyrics_impl,
)
from .reconstruction_intro_filters import (
    is_intro_artifact as _is_intro_artifact_impl,
)
from .reconstruction_intro_filters import (
    suppress_bottom_fragment_families as _suppress_bottom_fragment_families_impl,
)
from .reconstruction_context_transitions import (
    split_persistent_line_epochs_from_context_transitions as _split_persistent_line_epochs_from_context_transitions_impl,
)
from .reconstruction_lane_merge import (
    EntryPairPredicate,
    is_same_lane as _is_same_lane_impl,
)
from .reconstruction_lane_merge import (
    merge_dim_fade_in_fragments as _merge_dim_fade_in_fragments_impl,
)
from .reconstruction_lane_merge import (
    merge_overlapping_same_lane_duplicates as _merge_overlapping_same_lane_duplicates_impl,
)
from .reconstruction_overlap_repetitions import (
    expand_overlapped_same_text_repetitions as _expand_overlapped_same_text_repetitions_impl,
)
from .reconstruction_reentry import (
    merge_short_same_lane_reentries as _merge_short_same_lane_reentries_impl,
)
from .reconstruction_reentry import (
    suppress_short_duplicate_reentries as _suppress_short_duplicate_reentries_impl,
)
from .reconstruction_refrain import (
    collapse_short_refrain_noise as _collapse_short_refrain_noise_impl,
)
from .reconstruction_refrain import (
    is_short_refrain_entry as _is_short_refrain_entry_impl,
)
from .reconstruction_overlay import _filter_static_overlay_words
from .reconstruction_pipeline import (
    reconstruct_lyrics_from_visuals as _reconstruct_lyrics_from_visuals_impl,
)

_LANE_PROXIMITY_PX = 18.0


def snap(value: float) -> float:
    # Assuming 0.05s snap from original tool
    return round(round(float(value) / 0.05) * 0.05, 3)


def _suppress_short_duplicate_reentries(  # noqa: C901
    entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return _suppress_short_duplicate_reentries_impl(
        entries,
        is_same_lane=_is_same_lane,
    )


def _merge_short_same_lane_reentries(
    entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return _merge_short_same_lane_reentries_impl(
        entries,
        is_same_lane=_is_same_lane,
        is_short_refrain_entry=_is_short_refrain_entry,
    )


def _is_same_lane(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return _is_same_lane_impl(a, b, lane_proximity_px=_LANE_PROXIMITY_PX)


def _merge_overlapping_same_lane_duplicates(
    entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return _merge_overlapping_same_lane_duplicates_impl(
        entries,
        is_short_refrain_entry=_is_short_refrain_entry,
        is_same_lane=_is_same_lane,
    )


def _merge_dim_fade_in_fragments(
    entries: list[dict[str, Any]],
    is_same_lane: EntryPairPredicate,
) -> list[dict[str, Any]]:
    return _merge_dim_fade_in_fragments_impl(
        entries,
        is_same_lane=is_same_lane,
    )


def _is_short_refrain_entry(entry: dict[str, Any]) -> bool:
    return _is_short_refrain_entry_impl(entry)


def _collapse_short_refrain_noise(
    entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return _collapse_short_refrain_noise_impl(
        entries,
        is_short_refrain_entry=_is_short_refrain_entry,
        is_same_lane=_is_same_lane,
    )


def _is_intro_artifact(entry: dict[str, Any]) -> bool:
    return _is_intro_artifact_impl(entry)


def _filter_intro_non_lyrics(
    entries: list[dict[str, Any]], artist: Optional[str] = None
) -> list[dict[str, Any]]:
    return _filter_intro_non_lyrics_impl(entries, artist)


def _suppress_bottom_fragment_families(
    entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return _suppress_bottom_fragment_families_impl(entries)


def _expand_overlapped_same_text_repetitions(  # noqa: C901
    entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return _expand_overlapped_same_text_repetitions_impl(
        entries,
        lane_proximity_px=_LANE_PROXIMITY_PX,
        is_short_refrain_entry=_is_short_refrain_entry,
        is_same_lane=_is_same_lane,
    )


def _extrapolate_mirrored_lane_cycles(
    entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return _extrapolate_mirrored_lane_cycles_impl(
        entries,
        lane_proximity_px=_LANE_PROXIMITY_PX,
        is_candidate_for_mirrored_cycle=_is_candidate_for_mirrored_cycle,
        mirrored_cycle_candidate=_mirrored_cycle_candidate,
    )


def _is_candidate_for_mirrored_cycle(entry: dict[str, Any]) -> bool:
    return _is_candidate_for_mirrored_cycle_impl(
        entry,
        is_short_refrain_entry=_is_short_refrain_entry,
    )


def _mirrored_cycle_candidate(
    a: dict[str, Any], b: dict[str, Any]
) -> tuple[float, dict[str, Any]] | None:
    return _mirrored_cycle_candidate_impl(
        a,
        b,
        is_candidate_for_mirrored_cycle=_is_candidate_for_mirrored_cycle,
        is_same_lane=_is_same_lane,
    )


def _split_persistent_line_epochs_from_context_transitions(  # noqa: C901
    entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return _split_persistent_line_epochs_from_context_transitions_impl(
        entries,
        lane_proximity_px=_LANE_PROXIMITY_PX,
        is_short_refrain_entry=_is_short_refrain_entry,
    )


def reconstruct_lyrics_from_visuals(  # noqa: C901
    raw_frames: list[dict[str, Any]], visual_fps: float, artist: Optional[str] = None
) -> list[TargetLine]:
    return _reconstruct_lyrics_from_visuals_impl(
        raw_frames,
        visual_fps,
        filter_static_overlay_words=_filter_static_overlay_words,
        merge_overlapping_same_lane_duplicates=_merge_overlapping_same_lane_duplicates,
        merge_dim_fade_in_fragments=_merge_dim_fade_in_fragments,
        is_same_lane=_is_same_lane,
        merge_short_same_lane_reentries=_merge_short_same_lane_reentries,
        expand_overlapped_same_text_repetitions=_expand_overlapped_same_text_repetitions,
        extrapolate_mirrored_lane_cycles=_extrapolate_mirrored_lane_cycles,
        split_persistent_line_epochs_from_context_transitions=_split_persistent_line_epochs_from_context_transitions,
        suppress_short_duplicate_reentries=_suppress_short_duplicate_reentries,
        collapse_short_refrain_noise=_collapse_short_refrain_noise,
        filter_intro_non_lyrics=_filter_intro_non_lyrics,
        suppress_bottom_fragment_families=_suppress_bottom_fragment_families,
        snap_fn=snap,
        artist=artist,
    )
