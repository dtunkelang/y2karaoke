from __future__ import annotations

import os
import logging
import hashlib
import json
from typing import Any, Callable, Optional

from ..models import TargetLine
from ..text_utils import normalize_text_basic, text_similarity
from .extractor_mode import ResolvedVisualExtractorMode
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

    committed = accumulate_persistent_lines_from_frames(
        raw_frames,
        filter_static_overlay_words=filter_static_overlay_words,
        visual_fps=visual_fps,
    )
    _trace("accumulate", committed)

    if os.environ.get("Y2K_VISUAL_DISABLE_DEDUP", "0") == "1":
        unique = list(committed)
    else:
        unique = deduplicate_persistent_lines(committed)
    _trace("dedup", unique)
    if os.environ.get("Y2K_VISUAL_DISABLE_GHOST_REENTRY_GUARDS", "0") != "1":
        unique = _suppress_never_visible_ghost_reentries(
            unique, is_same_lane=is_same_lane
        )
    _trace("suppress_never_visible_ghost_reentries", unique)

    unique = merge_overlapping_same_lane_duplicates(unique)
    _trace("merge_overlap_same_lane", unique)

    unique = merge_dim_fade_in_fragments(unique, is_same_lane)
    _trace("merge_dim_fade", unique)

    unique = merge_short_same_lane_reentries(unique)
    _trace("merge_short_reentries", unique)

    unique = expand_overlapped_same_text_repetitions(unique)
    _trace("expand_overlapped_repeats", unique)
    unique = extrapolate_mirrored_lane_cycles(unique)
    _trace("extrapolate_mirrored_cycles", unique)
    if os.environ.get("Y2K_VISUAL_DISABLE_CONTEXT_SPLIT", "0") != "1":
        unique = split_persistent_line_epochs_from_context_transitions(unique)
    _trace("split_context_transitions", unique)
    unique = suppress_short_duplicate_reentries(unique)
    _trace("suppress_short_dup_reentries", unique)

    unique = collapse_short_refrain_noise(unique)
    _trace("collapse_short_refrain_noise", unique)
    if os.environ.get("Y2K_VISUAL_DISABLE_RECON_INTRO_FILTER", "0") != "1":
        unique = filter_intro_non_lyrics(unique, artist)
    _trace("filter_intro_non_lyrics", unique)
    if os.environ.get("Y2K_VISUAL_SUPPRESS_GLOBAL_METADATA", "1") != "0":
        unique = _suppress_global_metadata_noise(unique)
        _trace("suppress_global_metadata_noise", unique)

    unique = suppress_bottom_fragment_families(unique)
    _trace("suppress_bottom_fragment_families", unique)

    # Suppress short-lived, low-word-count fragments that are overshadowed by stable lines
    unique = _suppress_short_lane_fragments(unique)
    _trace("suppress_short_lane_fragments", unique)
    unique = _suppress_repeated_short_fragment_clusters(unique)
    _trace("suppress_repeated_short_fragments", unique)

    if use_block_first_proto:
        unique = apply_block_first_ordering_to_persistent_entries(unique)
        _trace("block_first_persistent_order", unique)
    # Final logic-based sequencing: Group lines by temporal overlap and sort by Y.
    elif os.environ.get("Y2K_VISUAL_DISABLE_SEQUENCING", "0") != "1":
        if os.environ.get("Y2K_VISUAL_SEQUENCE_SWEEP", "1") == "0":
            unique = _sequence_by_visual_neighborhood_legacy(unique)
        else:
            unique = _sequence_by_visual_neighborhood(unique)
    _trace("sequence_visual_neighborhood", unique)

    return convert_persistent_lines_to_target_lines(unique, snap_fn=snap_fn)


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
    """Identify and remove transient OCR noise overshadowed by stable lines."""
    if not lines:
        return []

    by_lane: dict[Any, list[tuple[int, dict[str, Any]]]] = {}
    for idx, line in enumerate(lines):
        by_lane.setdefault(line.get("lane"), []).append((idx, line))
    for lane_items in by_lane.values():
        lane_items.sort(key=lambda item: float(item[1].get("first", 0.0)))

    suppressed_indices = set()
    for i, line in enumerate(lines):
        dur = line["last"] - line["first"]
        wc = len(line["words"])

        # Candidate for suppression if short and thin
        if dur < 1.0 and wc < 4:
            lane_items = by_lane.get(line.get("lane"), [])
            line_first = float(line["first"])
            for j, other in lane_items:
                if i == j:
                    continue
                other_first = float(other["first"])
                if other_first < line_first - 3.0:
                    continue
                if other_first > line_first + 3.0:
                    break

                other_dur = other["last"] - other["first"]
                other_wc = len(other["words"])

                # If the other line is significantly more 'dominant'
                if other_dur > dur * 2 and other_wc >= wc + 2:
                    suppressed_indices.add(i)
                    break

    return [l for idx, l in enumerate(lines) if idx not in suppressed_indices]


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


_GLOBAL_META_KEYWORDS = {
    "karaoke",
    "karafun",
    "entertainer",
    "digitop",
    "rights",
    "reserved",
    "copyright",
    "produced",
    "association",
    "global",
    "ltd",
    "www",
    "http",
}


def _looks_global_metadata_noise(line: dict[str, Any]) -> bool:
    text = normalize_text_basic(str(line.get("text", ""))).strip().lower()
    if not text:
        return False
    tokens = [tok for tok in text.split() if tok]
    if not tokens:
        return False

    compact_tokens = ["".join(ch for ch in tok if ch.isalnum()) for tok in tokens]
    compact_tokens = [tok for tok in compact_tokens if tok]
    if not compact_tokens:
        return False

    metadata_hits = 0
    providerish = 0
    urlish = 0
    for tok in compact_tokens:
        if any(key in tok for key in _GLOBAL_META_KEYWORDS):
            metadata_hits += 1
        if tok.startswith(("kara", "xara", "xora")) and len(tok) >= 5:
            providerish += 1
        if "www" in tok or tok.endswith(("com", "couk", "net")):
            urlish += 1

    # Strong indicators: URLs/legal/provider spam.
    if urlish >= 1 and (metadata_hits >= 1 or providerish >= 1):
        return True
    if metadata_hits >= 3:
        return True
    if providerish >= 2 and len(compact_tokens) >= 2:
        return True

    # Long legal/credit lines are almost never lyrics in this pipeline context.
    if len(compact_tokens) >= 10 and (metadata_hits >= 2 or providerish >= 2):
        return True

    return False


def _suppress_global_metadata_noise(
    lines: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not lines:
        return []
    return [line for line in lines if not _looks_global_metadata_noise(line)]


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


def _log_sequence_blocks(mode: str, blocks: list[list[dict[str, Any]]]) -> None:
    _log_sequence_blocks_impl(mode, blocks)


def _should_disable_sequencing_for_blocks(blocks: list[list[dict[str, Any]]]) -> bool:
    return _should_disable_sequencing_for_blocks_impl(blocks)


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


def _tokens_contiguous_subphrase(needle: list[str], haystack: list[str]) -> bool:
    return _tokens_contiguous_subphrase_impl(needle, haystack)


def _has_significant_overlap(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return _has_significant_overlap_impl(a, b)
