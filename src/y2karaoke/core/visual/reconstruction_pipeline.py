from __future__ import annotations

import os
import logging
from typing import Any, Callable, Optional

from ..models import TargetLine
from ..text_utils import normalize_text_basic
from .reconstruction_frame_accumulation import accumulate_persistent_lines_from_frames
from .reconstruction_deduplication import deduplicate_persistent_lines
from .reconstruction_target_conversion import convert_persistent_lines_to_target_lines

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
) -> list[TargetLine]:
    trace_enabled = os.environ.get("Y2K_VISUAL_RECON_TRACE", "0") == "1"

    def _trace(stage: str, entries: list[dict[str, Any]]) -> None:
        if not trace_enabled:
            return
        token_count = sum(len(e.get("words", [])) for e in entries)
        logger.info(
            "recon_trace stage=%s lines=%d tokens=%d",
            stage,
            len(entries),
            token_count,
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

    # Final logic-based sequencing: Group lines by temporal overlap and sort by Y.
    if os.environ.get("Y2K_VISUAL_DISABLE_SEQUENCING", "0") != "1":
        if os.environ.get("Y2K_VISUAL_SEQUENCE_SWEEP", "1") == "0":
            unique = _sequence_by_visual_neighborhood_legacy(unique)
        else:
            unique = _sequence_by_visual_neighborhood(unique)
    _trace("sequence_visual_neighborhood", unique)

    return convert_persistent_lines_to_target_lines(unique, snap_fn=snap_fn)


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
    """Group truly simultaneous lines and sort Top-to-Bottom."""
    if not lines:
        return []
    input_order = list(lines)

    # 1. Sort by first detection to establish temporal baseline
    lines.sort(key=lambda x: x["first"])

    # 2. Build connected components with a sweep-line + union-find.
    # This preserves behavior while avoiding large adjacency sets on dense overlap cases.
    parent = list(range(len(lines)))
    rank = [0] * len(lines)

    def _find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def _union(a_idx: int, b_idx: int) -> None:
        ra = _find(a_idx)
        rb = _find(b_idx)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
            return
        if rank[ra] > rank[rb]:
            parent[rb] = ra
            return
        parent[rb] = ra
        rank[ra] += 1

    active: list[int] = []
    for j, line_j in enumerate(lines):
        start_j = float(line_j["first"])
        next_active: list[int] = []
        for i in active:
            # Same pruning condition as the original nested-loop break.
            if start_j <= float(lines[i]["last"]) + 1.5:
                next_active.append(i)
                if _has_significant_overlap(lines[i], line_j):
                    _union(i, j)
        next_active.append(j)
        active = next_active

    # 3. Collect connected components (visual blocks)
    blocks_by_root: dict[int, list[dict[str, Any]]] = {}
    for i, line in enumerate(lines):
        blocks_by_root.setdefault(_find(i), []).append(line)
    unsorted_blocks = list(blocks_by_root.values())
    _log_sequence_blocks("sweep", unsorted_blocks)
    if _should_disable_sequencing_for_blocks(unsorted_blocks):
        return input_order

    # 4. Sort blocks externally by earliest 'first', and internally by Y
    unsorted_blocks.sort(key=lambda b: min(x["first"] for x in b))

    ordered = []
    for block in unsorted_blocks:
        ordered.extend(_order_visual_block_locally(block))

    return ordered


def _sequence_by_visual_neighborhood_legacy(
    lines: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Previous adjacency-list implementation kept for regression A/B toggles."""
    if not lines:
        return []
    input_order = list(lines)

    lines.sort(key=lambda x: x["first"])
    adj: dict[int, set[int]] = {i: set() for i in range(len(lines))}
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            if _has_significant_overlap(lines[i], lines[j]):
                adj[i].add(j)
                adj[j].add(i)
            if lines[j]["first"] > lines[i]["last"] + 1.5:
                break

    visited = set()
    unsorted_blocks = []
    for i in range(len(lines)):
        if i not in visited:
            block_indices = []
            stack = [i]
            while stack:
                node = stack.pop()
                if node not in visited:
                    visited.add(node)
                    block_indices.append(node)
                    stack.extend(adj[node] - visited)
            unsorted_blocks.append([lines[idx] for idx in block_indices])
    _log_sequence_blocks("legacy", unsorted_blocks)
    if _should_disable_sequencing_for_blocks(unsorted_blocks):
        return input_order

    unsorted_blocks.sort(key=lambda b: min(x["first"] for x in b))
    ordered = []
    for block in unsorted_blocks:
        ordered.extend(_order_visual_block_locally(block))
    return ordered


def _log_sequence_blocks(mode: str, blocks: list[list[dict[str, Any]]]) -> None:
    if not _SEQUENCE_TRACE_ENABLED or not blocks:
        return
    stats = []
    for block in blocks:
        starts = [float(x.get("first_visible", x["first"])) for x in block]
        ends = [float(x["last"]) for x in block]
        stats.append(
            (
                len(block),
                max(ends) - min(starts),
                min(starts),
                max(ends),
            )
        )
    stats.sort(key=lambda x: (x[0], x[1]), reverse=True)
    top = ", ".join(
        f"size={size} span={span:.1f}s [{start:.1f},{end:.1f}]"
        for size, span, start, end in stats[:5]
    )
    logger.info(
        "sequence_blocks mode=%s count=%d top=%s",
        mode,
        len(blocks),
        top,
    )


def _should_disable_sequencing_for_blocks(blocks: list[list[dict[str, Any]]]) -> bool:
    """Disable global sequencing when overlap components indicate pathological chains."""
    for block in blocks:
        if len(block) < 4:
            continue
        starts = [float(x.get("first_visible", x["first"])) for x in block]
        ends = [float(x["last"]) for x in block]
        span = max(ends) - min(starts)
        # Large components spanning long intervals are strongly associated with
        # chronology-scrambling on repeated/refrain-heavy karaoke videos.
        if span >= 20.0:
            if _SEQUENCE_TRACE_ENABLED:
                logger.info(
                    "sequence_blocks global_fallback size=%d span=%.1fs",
                    len(block),
                    span,
                )
            return True
    return False


def _order_visual_block_locally(
    block: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Preserve chronology across long overlap chains; lane-sort only local simultaneous bands."""
    if len(block) <= 1:
        return list(block)

    chrono = sorted(
        block,
        key=lambda x: (
            float(x.get("first_visible", x["first"])),
            float(x["first"]),
            float(x["last"]),
        ),
    )
    block_starts = [float(x.get("first_visible", x["first"])) for x in chrono]
    block_ends = [float(x["last"]) for x in chrono]
    block_span = max(block_ends) - min(block_starts)
    # Long transitive overlap chains can scramble refinement-sensitive chronology (e.g. repeated choruses).
    if len(chrono) >= 3 and block_span >= 6.0:
        return chrono

    ordered: list[dict[str, Any]] = []
    band: list[dict[str, Any]] = []
    band_anchor = 0.0
    band_last = 0.0

    def _flush() -> None:
        nonlocal band
        if not band:
            return
        # Stable sort keeps same-lane lines in chronological order.
        band.sort(
            key=lambda x: (x.get("lane", 0), float(x.get("first_visible", x["first"])))
        )
        ordered.extend(band)
        band = []

    for line in chrono:
        start = float(line.get("first_visible", line["first"]))
        end = float(line["last"])
        if not band:
            band = [line]
            band_anchor = start
            band_last = end
            continue

        local_simultaneous = (
            start <= band_last + 0.20
            and start <= band_anchor + 0.90
            and any(_has_significant_overlap(existing, line) for existing in band)
        )
        if not local_simultaneous:
            _flush()
            band = [line]
            band_anchor = start
            band_last = end
            continue

        band.append(line)
        if end > band_last:
            band_last = end

    _flush()
    return ordered


def _has_significant_overlap(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Identify if two lines share enough temporal overlap to be part of the same visual block."""
    # Use first_visible for more stable overlap calculation if available
    start_a = a.get("first_visible", a["first"])
    start_b = b.get("first_visible", b["first"])

    overlap_start = max(start_a, start_b)
    overlap_end = min(a["last"], b["last"])
    overlap_duration = overlap_end - overlap_start
    if overlap_duration <= 0:
        return False

    # Requirement: overlap at least 70% of the duration of BOTH lines
    dur_a = max(0.1, a["last"] - start_a)
    dur_b = max(0.1, b["last"] - start_b)

    return (overlap_duration / dur_a) >= 0.7 and (overlap_duration / dur_b) >= 0.7


_SEQUENCE_TRACE_ENABLED = os.environ.get("Y2K_VISUAL_SEQUENCE_TRACE", "0") == "1"
