from __future__ import annotations

from typing import Any, Callable, Optional

from ..models import TargetLine
from .reconstruction_frame_accumulation import accumulate_persistent_lines_from_frames
from .reconstruction_deduplication import deduplicate_persistent_lines
from .reconstruction_target_conversion import convert_persistent_lines_to_target_lines

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
    committed = accumulate_persistent_lines_from_frames(
        raw_frames,
        filter_static_overlay_words=filter_static_overlay_words,
        visual_fps=visual_fps,
    )

    unique = deduplicate_persistent_lines(committed)

    unique = merge_overlapping_same_lane_duplicates(unique)

    unique = merge_dim_fade_in_fragments(unique, is_same_lane)

    unique = merge_short_same_lane_reentries(unique)

    unique = expand_overlapped_same_text_repetitions(unique)
    unique = extrapolate_mirrored_lane_cycles(unique)
    unique = split_persistent_line_epochs_from_context_transitions(unique)
    unique = suppress_short_duplicate_reentries(unique)

    unique = collapse_short_refrain_noise(unique)
    unique = filter_intro_non_lyrics(unique, artist)

    unique = suppress_bottom_fragment_families(unique)

    # Suppress short-lived, low-word-count fragments that are overshadowed by stable lines
    unique = _suppress_short_lane_fragments(unique)

    # Final logic-based sequencing: Group lines by temporal overlap and sort by Y.
    unique = _sequence_by_visual_neighborhood(unique)

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


def _sequence_by_visual_neighborhood(
    lines: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Group truly simultaneous lines and sort Top-to-Bottom."""
    if not lines:
        return []

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

    # 4. Sort blocks externally by earliest 'first', and internally by Y
    unsorted_blocks.sort(key=lambda b: min(x["first"] for x in b))

    ordered = []
    for block in unsorted_blocks:
        # Top-to-Bottom priority using stabilized lane IDs
        block.sort(key=lambda x: x["lane"])
        ordered.extend(block)

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
