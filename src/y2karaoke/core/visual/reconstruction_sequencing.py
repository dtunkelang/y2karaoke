"""Sequencing helpers for ordering reconstructed visual lyric lines."""

from __future__ import annotations

import logging
import os
from difflib import SequenceMatcher
from typing import Any, Callable

logger = logging.getLogger(__name__)

_SEQUENCE_TRACE_ENABLED = os.environ.get("Y2K_VISUAL_SEQUENCE_TRACE", "0") == "1"


def _dsu_find(parent: list[int], idx: int) -> int:
    while parent[idx] != idx:
        parent[idx] = parent[parent[idx]]
        idx = parent[idx]
    return idx


def _dsu_union(parent: list[int], rank: list[int], a_idx: int, b_idx: int) -> None:
    ra = _dsu_find(parent, a_idx)
    rb = _dsu_find(parent, b_idx)
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


def _connect_overlapping_active_lines(
    lines: list[dict[str, Any]],
    *,
    has_significant_overlap_fn: Callable[[dict[str, Any], dict[str, Any]], bool],
) -> tuple[list[int], list[int]]:
    parent = list(range(len(lines)))
    rank = [0] * len(lines)
    active: list[int] = []
    for j, line_j in enumerate(lines):
        start_j = float(line_j["first"])
        next_active: list[int] = []
        for i in active:
            if start_j <= float(lines[i]["last"]) + 1.5:
                next_active.append(i)
                if has_significant_overlap_fn(lines[i], line_j):
                    _dsu_union(parent, rank, i, j)
        next_active.append(j)
        active = next_active
    return parent, rank


def _blocks_from_dsu(
    lines: list[dict[str, Any]], parent: list[int]
) -> list[list[dict[str, Any]]]:
    blocks_by_root: dict[int, list[dict[str, Any]]] = {}
    for i, line in enumerate(lines):
        blocks_by_root.setdefault(_dsu_find(parent, i), []).append(line)
    return list(blocks_by_root.values())


def _order_blocks_locally(
    blocks: list[list[dict[str, Any]]],
    *,
    order_visual_block_locally_fn: Callable[
        [list[dict[str, Any]]], list[dict[str, Any]]
    ],
) -> list[dict[str, Any]]:
    blocks.sort(key=lambda b: min(x["first"] for x in b))
    ordered: list[dict[str, Any]] = []
    for block in blocks:
        ordered.extend(order_visual_block_locally_fn(block))
    return ordered


def sequence_by_visual_neighborhood(
    lines: list[dict[str, Any]],
    *,
    has_significant_overlap_fn: Callable[[dict[str, Any], dict[str, Any]], bool],
    log_sequence_blocks_fn: Callable[[str, list[list[dict[str, Any]]]], None],
    should_disable_sequencing_for_blocks_fn: Callable[
        [list[list[dict[str, Any]]]], bool
    ],
    order_visual_block_locally_fn: Callable[
        [list[dict[str, Any]]], list[dict[str, Any]]
    ],
) -> list[dict[str, Any]]:
    if not lines:
        return []
    input_order = list(lines)
    lines.sort(key=lambda x: x["first"])
    parent, _rank = _connect_overlapping_active_lines(
        lines, has_significant_overlap_fn=has_significant_overlap_fn
    )
    unsorted_blocks = _blocks_from_dsu(lines, parent)
    log_sequence_blocks_fn("sweep", unsorted_blocks)
    if should_disable_sequencing_for_blocks_fn(unsorted_blocks):
        return input_order
    return _order_blocks_locally(
        unsorted_blocks, order_visual_block_locally_fn=order_visual_block_locally_fn
    )


def sequence_by_visual_neighborhood_legacy(
    lines: list[dict[str, Any]],
    *,
    has_significant_overlap_fn: Callable[[dict[str, Any], dict[str, Any]], bool],
    log_sequence_blocks_fn: Callable[[str, list[list[dict[str, Any]]]], None],
    should_disable_sequencing_for_blocks_fn: Callable[
        [list[list[dict[str, Any]]]], bool
    ],
    order_visual_block_locally_fn: Callable[
        [list[dict[str, Any]]], list[dict[str, Any]]
    ],
) -> list[dict[str, Any]]:
    if not lines:
        return []
    input_order = list(lines)

    lines.sort(key=lambda x: x["first"])
    adj = _build_legacy_overlap_adjacency(
        lines, has_significant_overlap_fn=has_significant_overlap_fn
    )
    unsorted_blocks = _legacy_blocks_from_adjacency(lines, adj)
    log_sequence_blocks_fn("legacy", unsorted_blocks)
    if should_disable_sequencing_for_blocks_fn(unsorted_blocks):
        return input_order

    return _order_legacy_blocks(
        unsorted_blocks, order_visual_block_locally_fn=order_visual_block_locally_fn
    )


def _build_legacy_overlap_adjacency(
    lines: list[dict[str, Any]],
    *,
    has_significant_overlap_fn: Callable[[dict[str, Any], dict[str, Any]], bool],
) -> dict[int, set[int]]:
    adjacency: dict[int, set[int]] = {i: set() for i in range(len(lines))}
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            if has_significant_overlap_fn(lines[i], lines[j]):
                adjacency[i].add(j)
                adjacency[j].add(i)
            if lines[j]["first"] > lines[i]["last"] + 1.5:
                break
    return adjacency


def _legacy_blocks_from_adjacency(
    lines: list[dict[str, Any]], adjacency: dict[int, set[int]]
) -> list[list[dict[str, Any]]]:
    visited: set[int] = set()
    unsorted_blocks: list[list[dict[str, Any]]] = []
    for i in range(len(lines)):
        if i in visited:
            continue
        block_indices: list[int] = []
        stack = [i]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            block_indices.append(node)
            stack.extend(adjacency[node] - visited)
        unsorted_blocks.append([lines[idx] for idx in block_indices])
    return unsorted_blocks


def _order_legacy_blocks(
    unsorted_blocks: list[list[dict[str, Any]]],
    *,
    order_visual_block_locally_fn: Callable[
        [list[dict[str, Any]]], list[dict[str, Any]]
    ],
) -> list[dict[str, Any]]:
    unsorted_blocks.sort(key=lambda block: min(line["first"] for line in block))
    ordered: list[dict[str, Any]] = []
    for block in unsorted_blocks:
        ordered.extend(order_visual_block_locally_fn(block))
    return ordered


def log_sequence_blocks(mode: str, blocks: list[list[dict[str, Any]]]) -> None:
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


def should_disable_sequencing_for_blocks(blocks: list[list[dict[str, Any]]]) -> bool:
    for block in blocks:
        if len(block) < 4:
            continue
        starts = [float(x.get("first_visible", x["first"])) for x in block]
        ends = [float(x["last"]) for x in block]
        span = max(ends) - min(starts)
        if span >= 20.0:
            if _SEQUENCE_TRACE_ENABLED:
                logger.info(
                    "sequence_blocks global_fallback size=%d span=%.1fs",
                    len(block),
                    span,
                )
            return True
    return False


def order_visual_block_locally(
    block: list[dict[str, Any]],
    *,
    has_significant_overlap_fn: Callable[[dict[str, Any], dict[str, Any]], bool],
    band_fragment_indices_fn: Callable[[list[dict[str, Any]]], set[int]],
) -> list[dict[str, Any]]:
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
        fragment_indices = band_fragment_indices_fn(band)
        band.sort(
            key=lambda x: (
                1 if id(x) in fragment_indices else 0,
                x.get("lane", 0),
                float(x.get("first_visible", x["first"])),
            )
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
            and any(has_significant_overlap_fn(existing, line) for existing in band)
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


def band_fragment_indices(
    band: list[dict[str, Any]],
    *,
    normalize_text_basic_fn: Callable[[str], str],
    is_band_fragment_subphrase_fn: Callable[[list[str], list[str]], bool],
) -> set[int]:
    if len(band) < 2:
        return set()

    def _tokens(line: dict[str, Any]) -> list[str]:
        return [
            t for t in normalize_text_basic_fn(str(line.get("text", ""))).split() if t
        ]

    token_lists = [_tokens(line) for line in band]
    out: set[int] = set()
    for i, toks_i in enumerate(token_lists):
        if len(toks_i) < 1 or len(toks_i) > 4:
            continue
        if all(t in {"oh", "ooh", "ah", "la", "na", "mm", "mmm"} for t in toks_i):
            continue
        dur_i = max(
            0.1,
            float(band[i]["last"])
            - float(band[i].get("first_visible", band[i]["first"])),
        )
        for j, toks_j in enumerate(token_lists):
            if i == j or len(toks_j) <= len(toks_i):
                continue
            if not is_band_fragment_subphrase_fn(toks_i, toks_j):
                continue
            dur_j = max(
                0.1,
                float(band[j]["last"])
                - float(band[j].get("first_visible", band[j]["first"])),
            )
            if dur_j < dur_i:
                continue
            out.add(id(band[i]))
            break
    return out


def is_band_fragment_subphrase(
    fragment: list[str],
    full: list[str],
    *,
    tokens_contiguous_subphrase_fn: Callable[[list[str], list[str]], bool],
) -> bool:
    if tokens_contiguous_subphrase_fn(fragment, full):
        return True
    return _matches_singular_plural_variant(fragment, full) or _matches_merged_fragment(
        fragment, full
    )


def _matches_singular_plural_variant(fragment: list[str], full: list[str]) -> bool:
    if len(fragment) != 1:
        return False
    token = fragment[0]
    if len(token) < 4:
        return False
    singular = token[:-1] if token.endswith("s") else token
    plural = token if token.endswith("s") else f"{token}s"
    return any(full_token == singular or full_token == plural for full_token in full)


def _matches_merged_fragment(fragment: list[str], full: list[str]) -> bool:
    if not (2 <= len(fragment) <= 3):
        return False
    if not all(1 <= len(token) <= 4 for token in fragment):
        return False
    merged = "".join(fragment)
    if len(merged) < 5:
        return False
    for token in full:
        if len(token) < len(merged):
            continue
        if token == merged:
            return True
        if SequenceMatcher(None, merged, token).ratio() >= 0.84:
            return True
    return False


def tokens_contiguous_subphrase(needle: list[str], haystack: list[str]) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    n = len(needle)
    for idx in range(0, len(haystack) - n + 1):
        if haystack[idx : idx + n] == needle:
            return True
    return False


def has_significant_overlap(a: dict[str, Any], b: dict[str, Any]) -> bool:
    start_a = a.get("first_visible", a["first"])
    start_b = b.get("first_visible", b["first"])

    overlap_start = max(start_a, start_b)
    overlap_end = min(a["last"], b["last"])
    overlap_duration = overlap_end - overlap_start
    if overlap_duration <= 0:
        return False

    dur_a = max(0.1, a["last"] - start_a)
    dur_b = max(0.1, b["last"] - start_b)
    return (overlap_duration / dur_a) >= 0.7 and (overlap_duration / dur_b) >= 0.7
