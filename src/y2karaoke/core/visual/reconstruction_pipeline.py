from __future__ import annotations

import os
import logging
import hashlib
import json
from typing import Any, Callable, Optional

from ..models import TargetLine
from ..text_utils import normalize_text_basic, text_similarity
from .extractor_mode import ResolvedVisualExtractorMode
from .reconstruction_metadata_filters import (
    looks_global_metadata_noise as _looks_global_metadata_noise_impl,
    suppress_global_metadata_noise as _suppress_global_metadata_noise_impl,
)
from .reconstruction_fragment_filters import (
    suppress_short_lane_fragments as _suppress_short_lane_fragments_impl,
)
from .reconstruction_frame_accumulation import accumulate_persistent_lines_from_frames
from .reconstruction_deduplication import deduplicate_persistent_lines
from .reconstruction_block_first import (
    apply_block_first_ordering_to_persistent_entries,
)
from .reconstruction_block_first_frames import (
    reconstruct_lyrics_from_visuals_block_first_frames,
)
from .reconstruction_target_conversion import convert_persistent_lines_to_target_lines
from .reconstruction_sequencing import (
    band_fragment_indices as _band_fragment_indices_impl,
    has_significant_overlap as _has_significant_overlap_impl,
    is_band_fragment_subphrase as _is_band_fragment_subphrase_impl,
    log_sequence_blocks as _log_sequence_blocks_impl,
    order_visual_block_locally as _order_visual_block_locally_impl,
    sequence_by_visual_neighborhood as _sequence_by_visual_neighborhood_impl,
    sequence_by_visual_neighborhood_legacy as _sequence_by_visual_neighborhood_legacy_impl,
    should_disable_sequencing_for_blocks as _should_disable_sequencing_for_blocks_impl,
    tokens_contiguous_subphrase as _tokens_contiguous_subphrase_impl,
)

logger = logging.getLogger(__name__)

EntriesPass = Callable[[list[dict[str, Any]]], list[dict[str, Any]]]
EntryPairPredicate = Callable[[dict[str, Any], dict[str, Any]], bool]
FrameFilter = Callable[[list[dict[str, Any]]], list[dict[str, Any]]]
SnapFn = Callable[[float], float]
TraceFn = Callable[[str, list[dict[str, Any]]], None]


def _build_reconstruction_trace_fn(*, trace_enabled: bool) -> TraceFn:
    def _trace(stage: str, entries: list[dict[str, Any]]) -> None:
        if not trace_enabled:
            return
        token_count = sum(len(e.get("words", [])) for e in entries)
        trace_hash = hashlib.sha256(
            json.dumps(
                [
                    (
                        round(float(e.get("first_visible", e.get("first", 0.0))), 3),
                        round(float(e.get("first", 0.0)), 3),
                        round(float(e.get("last", 0.0)), 3),
                        int(e.get("lane", -1) if e.get("lane") is not None else -1),
                        str(e.get("text", "")),
                    )
                    for e in entries
                ],
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:12]
        logger.info(
            "recon_trace stage=%s lines=%d tokens=%d hash=%s",
            stage,
            len(entries),
            token_count,
            trace_hash,
        )

    return _trace


def reconstruct_lyrics_from_visuals(  # noqa: C901
    raw_frames: list[dict[str, Any]],
    visual_fps: float,
    *,
    filter_static_overlay_words: FrameFilter,
    merge_overlapping_same_lane_duplicates: EntriesPass,
    merge_dim_fade_in_fragments: Callable[
        [list[dict[str, Any]], EntryPairPredicate], list[dict[str, Any]]
    ],
    is_same_lane: EntryPairPredicate,
    merge_short_same_lane_reentries: EntriesPass,
    expand_overlapped_same_text_repetitions: EntriesPass,
    extrapolate_mirrored_lane_cycles: EntriesPass,
    split_persistent_line_epochs_from_context_transitions: EntriesPass,
    suppress_short_duplicate_reentries: EntriesPass,
    collapse_short_refrain_noise: EntriesPass,
    filter_intro_non_lyrics: Callable[
        [list[dict[str, Any]], Optional[str]], list[dict[str, Any]]
    ],
    suppress_bottom_fragment_families: EntriesPass,
    snap_fn: SnapFn,
    artist: Optional[str] = None,
    extractor_mode: ResolvedVisualExtractorMode = "line-first",
) -> list[TargetLine]:
    trace_enabled = os.environ.get("Y2K_VISUAL_RECON_TRACE", "0") == "1"
    trace = _build_reconstruction_trace_fn(trace_enabled=trace_enabled)
    use_block_first_proto = extractor_mode == "block-first" or (
        os.environ.get("Y2K_VISUAL_BLOCK_FIRST_PROTOTYPE", "0") == "1"
    )

    if use_block_first_proto:
        block_first_lines = reconstruct_lyrics_from_visuals_block_first_frames(
            raw_frames,
            filter_static_overlay_words=filter_static_overlay_words,
            snap_fn=snap_fn,
        )
        if block_first_lines:
            if trace_enabled:
                logger.info(
                    "recon_trace stage=block_first_frame_extractor lines=%d",
                    len(block_first_lines),
                )
            return block_first_lines

    unique = _run_line_first_reconstruction_passes(
        raw_frames,
        visual_fps=visual_fps,
        filter_static_overlay_words=filter_static_overlay_words,
        is_same_lane=is_same_lane,
        merge_overlapping_same_lane_duplicates=merge_overlapping_same_lane_duplicates,
        merge_dim_fade_in_fragments=merge_dim_fade_in_fragments,
        merge_short_same_lane_reentries=merge_short_same_lane_reentries,
        expand_overlapped_same_text_repetitions=expand_overlapped_same_text_repetitions,
        extrapolate_mirrored_lane_cycles=extrapolate_mirrored_lane_cycles,
        split_persistent_line_epochs_from_context_transitions=split_persistent_line_epochs_from_context_transitions,
        suppress_short_duplicate_reentries=suppress_short_duplicate_reentries,
        collapse_short_refrain_noise=collapse_short_refrain_noise,
        filter_intro_non_lyrics=filter_intro_non_lyrics,
        suppress_bottom_fragment_families=suppress_bottom_fragment_families,
        artist=artist,
        trace=trace,
    )

    if use_block_first_proto:
        unique = apply_block_first_ordering_to_persistent_entries(unique)
        trace("block_first_persistent_order", unique)
    # Final logic-based sequencing: Group lines by temporal overlap and sort by Y.
    elif os.environ.get("Y2K_VISUAL_DISABLE_SEQUENCING", "0") != "1":
        if os.environ.get("Y2K_VISUAL_SEQUENCE_SWEEP", "1") == "0":
            unique = _sequence_by_visual_neighborhood_legacy(unique)
        else:
            unique = _sequence_by_visual_neighborhood(unique)
    trace("sequence_visual_neighborhood", unique)

    return convert_persistent_lines_to_target_lines(unique, snap_fn=snap_fn)


def _run_line_first_reconstruction_passes(
    raw_frames: list[dict[str, Any]],
    *,
    visual_fps: float,
    filter_static_overlay_words: FrameFilter,
    is_same_lane: EntryPairPredicate,
    merge_overlapping_same_lane_duplicates: EntriesPass,
    merge_dim_fade_in_fragments: Callable[
        [list[dict[str, Any]], EntryPairPredicate], list[dict[str, Any]]
    ],
    merge_short_same_lane_reentries: EntriesPass,
    expand_overlapped_same_text_repetitions: EntriesPass,
    extrapolate_mirrored_lane_cycles: EntriesPass,
    split_persistent_line_epochs_from_context_transitions: EntriesPass,
    suppress_short_duplicate_reentries: EntriesPass,
    collapse_short_refrain_noise: EntriesPass,
    filter_intro_non_lyrics: Callable[
        [list[dict[str, Any]], Optional[str]], list[dict[str, Any]]
    ],
    suppress_bottom_fragment_families: EntriesPass,
    artist: Optional[str],
    trace: TraceFn,
) -> list[dict[str, Any]]:
    committed = accumulate_persistent_lines_from_frames(
        raw_frames,
        filter_static_overlay_words=filter_static_overlay_words,
        visual_fps=visual_fps,
    )
    trace("accumulate", committed)

    if os.environ.get("Y2K_VISUAL_DISABLE_DEDUP", "0") == "1":
        unique = list(committed)
    else:
        unique = deduplicate_persistent_lines(committed)
    trace("dedup", unique)
    if os.environ.get("Y2K_VISUAL_DISABLE_GHOST_REENTRY_GUARDS", "0") != "1":
        unique = _suppress_never_visible_ghost_reentries(
            unique, is_same_lane=is_same_lane
        )
    trace("suppress_never_visible_ghost_reentries", unique)

    unique = merge_overlapping_same_lane_duplicates(unique)
    trace("merge_overlap_same_lane", unique)
    unique = merge_dim_fade_in_fragments(unique, is_same_lane)
    trace("merge_dim_fade", unique)
    unique = merge_short_same_lane_reentries(unique)
    trace("merge_short_reentries", unique)
    unique = expand_overlapped_same_text_repetitions(unique)
    trace("expand_overlapped_repeats", unique)
    unique = extrapolate_mirrored_lane_cycles(unique)
    trace("extrapolate_mirrored_cycles", unique)
    if os.environ.get("Y2K_VISUAL_DISABLE_CONTEXT_SPLIT", "0") != "1":
        unique = split_persistent_line_epochs_from_context_transitions(unique)
    trace("split_context_transitions", unique)
    unique = suppress_short_duplicate_reentries(unique)
    trace("suppress_short_dup_reentries", unique)
    unique = collapse_short_refrain_noise(unique)
    trace("collapse_short_refrain_noise", unique)
    if os.environ.get("Y2K_VISUAL_DISABLE_RECON_INTRO_FILTER", "0") != "1":
        unique = filter_intro_non_lyrics(unique, artist)
    trace("filter_intro_non_lyrics", unique)
    if os.environ.get("Y2K_VISUAL_SUPPRESS_GLOBAL_METADATA", "1") != "0":
        unique = _suppress_global_metadata_noise(unique)
        trace("suppress_global_metadata_noise", unique)
    unique = suppress_bottom_fragment_families(unique)
    trace("suppress_bottom_fragment_families", unique)
    unique = _suppress_short_lane_fragments(unique)
    trace("suppress_short_lane_fragments", unique)
    unique = _suppress_repeated_short_fragment_clusters(unique)
    trace("suppress_repeated_short_fragments", unique)
    return unique


def _suppress_never_visible_ghost_reentries(
    lines: list[dict[str, Any]],
    *,
    is_same_lane: EntryPairPredicate,
) -> list[dict[str, Any]]:
    """Drop later never-visible reentries that mirror an earlier visible line.

    These are typically dim OCR ghost trails that persist after a lyric line fades out
    and can incorrectly bridge instrumental gaps.
    """
    if len(lines) < 2:
        return lines

    kept: list[dict[str, Any]] = []
    for ent in lines:
        if bool(ent.get("visible_yet", False)):
            kept.append(ent)
            continue

        suppress = False
        ent_first = float(ent.get("first", 0.0))
        ent_last = float(ent.get("last", ent_first))
        ent_dur = max(0.0, ent_last - ent_first)
        if ent_dur >= 1.0:
            for prev in reversed(kept[-16:]):
                if not bool(prev.get("visible_yet", False)):
                    continue
                if not is_same_lane(prev, ent):
                    continue
                if (
                    text_similarity(str(prev.get("text", "")), str(ent.get("text", "")))
                    < 0.9
                ):
                    continue
                prev_last = float(prev.get("last", prev.get("first", 0.0)))
                if ent_first >= prev_last + 0.8:
                    suppress = True
                    break

        if not suppress:
            kept.append(ent)

    return kept


def _suppress_short_lane_fragments(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _suppress_short_lane_fragments_impl(lines)


def _suppress_repeated_short_fragment_clusters(  # noqa: C901
    lines: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Remove repeated short fragment shards backed by nearby fuller lyric lines."""
    if len(lines) < 4:
        return lines

    token_lists = [
        [t for t in normalize_text_basic(str(line.get("text", ""))).split() if t]
        for line in lines
    ]
    groups: dict[tuple[str, ...], list[int]] = {}
    for idx, (line, toks) in enumerate(zip(lines, token_lists)):
        if not (1 <= len(toks) <= 3):
            continue
        dur = float(line["last"]) - float(line["first"])
        if dur > 1.6:
            continue
        groups.setdefault(tuple(toks), []).append(idx)

    suppressed: set[int] = set()
    for key, idxs in groups.items():
        if len(idxs) < 2:
            continue
        avg_dur = sum(
            max(0.0, float(lines[i]["last"]) - float(lines[i]["first"])) for i in idxs
        ) / max(len(idxs), 1)
        min_count = 3
        if len(key) <= 2 and avg_dur <= 1.0:
            min_count = 2
        if len(idxs) < min_count:
            continue

        supported: list[int] = []
        for idx in idxs:
            line = lines[idx]
            start = float(line["first"])
            end = float(line["last"])
            for j, other in enumerate(lines):
                if j == idx:
                    continue
                other_toks = token_lists[j]
                if len(other_toks) <= len(key):
                    continue
                other_start = float(other["first"])
                other_end = float(other["last"])
                if other_end < start - 3.0 or other_start > end + 3.0:
                    continue
                if _is_band_fragment_subphrase(list(key), other_toks):
                    supported.append(idx)
                    break

        if len(supported) < min_count:
            continue
        suppressed.update(supported)

    if not suppressed:
        return lines
    return [line for idx, line in enumerate(lines) if idx not in suppressed]


_looks_global_metadata_noise = _looks_global_metadata_noise_impl
_suppress_global_metadata_noise = _suppress_global_metadata_noise_impl


def _sequence_by_visual_neighborhood(
    lines: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return _sequence_by_visual_neighborhood_impl(
        lines,
        has_significant_overlap_fn=_has_significant_overlap,
        log_sequence_blocks_fn=_log_sequence_blocks,
        should_disable_sequencing_for_blocks_fn=_should_disable_sequencing_for_blocks,
        order_visual_block_locally_fn=_order_visual_block_locally,
    )


def _sequence_by_visual_neighborhood_legacy(
    lines: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return _sequence_by_visual_neighborhood_legacy_impl(
        lines,
        has_significant_overlap_fn=_has_significant_overlap,
        log_sequence_blocks_fn=_log_sequence_blocks,
        should_disable_sequencing_for_blocks_fn=_should_disable_sequencing_for_blocks,
        order_visual_block_locally_fn=_order_visual_block_locally,
    )


_log_sequence_blocks = _log_sequence_blocks_impl


_should_disable_sequencing_for_blocks = _should_disable_sequencing_for_blocks_impl


def _order_visual_block_locally(
    block: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return _order_visual_block_locally_impl(
        block,
        has_significant_overlap_fn=_has_significant_overlap,
        band_fragment_indices_fn=_band_fragment_indices,
    )


def _band_fragment_indices(band: list[dict[str, Any]]) -> set[int]:
    return _band_fragment_indices_impl(
        band,
        normalize_text_basic_fn=normalize_text_basic,
        is_band_fragment_subphrase_fn=_is_band_fragment_subphrase,
    )


def _is_band_fragment_subphrase(fragment: list[str], full: list[str]) -> bool:
    return _is_band_fragment_subphrase_impl(
        fragment,
        full,
        tokens_contiguous_subphrase_fn=_tokens_contiguous_subphrase,
    )


_tokens_contiguous_subphrase = _tokens_contiguous_subphrase_impl


_has_significant_overlap = _has_significant_overlap_impl
